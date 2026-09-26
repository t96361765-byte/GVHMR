"""Circle-only closed-leg refinement, with exact preservation outside the stage."""

import copy
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteV437Coco17
from hmr4d.utils.mushroom_refine import BODY_MAP, Y_TO_Z, fk, project, cycle_metrics, reprojection_metrics
from hmr4d.utils.mushroom_contact import palm_surface_gaps, arm_body_regions, ArmBodyCollision
from hmr4d.utils.mushroom_priors import trajectory_statistics, circle_envelope
from hmr4d.utils.mushroom_config import resolve_constraints, pixel_scale, weighted_loss, constraint_hash

# Body-pose indices (SMPL-X joint index minus the root).
LEG_POSE_IDS = [0, 1, 3, 4, 6, 7]


def leg_statistics(joints, selection):
    j = np.asarray(joints)[selection]

    def angle(u, v):
        dot = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))
        return np.degrees(np.arccos(np.clip(dot, -1, 1)))

    def summary(x):
        return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))

    thigh = j[:, [4, 5]] - j[:, [1, 2]]
    shin = j[:, [7, 8]] - j[:, [4, 5]]
    whole = j[:, [7, 8]] - j[:, [1, 2]]
    return dict(
        ankle_gap_cm=summary(np.linalg.norm(j[:, 7] - j[:, 8], axis=-1) * 100),
        knee_gap_cm=summary(np.linalg.norm(j[:, 4] - j[:, 5], axis=-1) * 100),
        thigh_angle_deg=summary(angle(thigh[:, 0], thigh[:, 1])),
        shin_angle_deg=summary(angle(shin[:, 0], shin[:, 1])),
        leg_angle_deg=summary(angle(whole[:, 0], whole[:, 1])),
        knee_flexion_deg=summary(angle(thigh, shin)),
    )


def bvh_leg_limits(bvh, reference_range, skeleton, options=None):
    options = options or resolve_constraints()["legs"]
    percentile = options["prior_percentile"]
    j = bvh["joints"][slice(*reference_range)]
    j = {name: j[:, bvh["names"].index(name)] for name in BODY_MAP.values()}
    thigh = [j[s + "Knee"] - j[s + "Hip"] for s in ["Left", "Right"]]
    shin = [j[s + "Ankle"] - j[s + "Knee"] for s in ["Left", "Right"]]
    length = np.mean([np.linalg.norm(t, axis=-1) + np.linalg.norm(s, axis=-1) for t, s in zip(thigh, shin)])
    target_length = float(
        np.mean(
            [
                np.linalg.norm(skeleton[k] - skeleton[h]) + np.linalg.norm(skeleton[a] - skeleton[k])
                for h, k, a in [(1, 4, 7), (2, 5, 8)]
            ]
        )
    )

    def angle(u, v):
        return np.degrees(
            np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1))
        )

    flex = np.stack([angle(t, s) for t, s in zip(thigh, shin)], -1)
    return dict(
        ankle_max_m=float(
            np.percentile(np.linalg.norm(j["LeftAnkle"] - j["RightAnkle"], axis=-1) / length, percentile)
            * target_length
            + options["ankle_gap_margin_m"]
        ),
        knee_max_m=float(
            np.percentile(np.linalg.norm(j["LeftKnee"] - j["RightKnee"], axis=-1) / length, percentile) * target_length
            + options["knee_gap_margin_m"]
        ),
        thigh_max_deg=float(np.percentile(angle(*thigh), percentile) + options["thigh_angle_margin_deg"]),
        shin_max_deg=float(np.percentile(angle(*shin), percentile) + options["shin_angle_margin_deg"]),
        flexion_max_deg=float(np.percentile(flex, percentile) + options["knee_flexion_margin_deg"]),
        flexion_difference_max_deg=float(
            np.percentile(abs(flex[:, 0] - flex[:, 1]), percentile) + options["knee_asymmetry_margin_deg"]
        ),
        source_leg_length_m=float(length),
        target_leg_length_m=target_length,
        note="Unpaired BVH percentiles with margins, scaled by leg length; not framewise targets.",
    )


def refine_closed_legs(prediction, arrays, metrics, bvh, cfg, iterations=None, device="cuda", callback=None):
    constraints = resolve_constraints(cfg)
    options = constraints["legs"]
    weights = options["weights"]
    if not options["enabled"]:
        met = copy.deepcopy(metrics)
        met.setdefault("active_refinement", {}).setdefault("closed_legs", False)
        return copy.deepcopy(prediction), dict(arrays), met
    iterations = options["iterations"] if iterations is None else iterations
    options["iterations"] = iterations
    torch.set_num_threads(4)
    torch.manual_seed(constraints["shared"]["seed"])
    ss, se = cfg["circle_range"]
    ks, ke = cfg["keep_range"]
    fps = cfg.get("fps", 30.0)
    base = {k: v.to(device) for k, v in prediction["smpl_params_global"].items()}
    count = len(base["body_pose"])
    if not 0 <= ks <= ss < se <= ke <= count:
        raise ValueError("Circle range must be inside keep_range and motion length")
    if iterations <= 0:
        raise ValueError("Closed-leg iterations must be positive")
    envelope = circle_envelope(count, ss, se, fps, options["fade_seconds"])
    active = np.flatnonzero(envelope > 0)
    core = envelope >= 1 - 1e-8
    if not len(active) or not core.any():
        raise ValueError("Circle segment is too short for closed-leg refinement")
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    N = lambda x: x.detach().cpu().numpy()
    model = SmplxLiteV437Coco17().eval()
    dense = SmplxLite().eval()
    body_faces, arm_ids = arm_body_regions(dense)
    vids = torch.tensor(arrays["surface_vertex_ids"], dtype=torch.long)
    for name in ["v_template", "shapedirs", "lbs_weights"]:
        setattr(model, name, torch.cat([getattr(model, name)[:132], getattr(dense, name)[vids]], 0))
    model.posedirs = torch.cat([model.posedirs[:, :132], dense.posedirs[:, vids]], 1)
    del dense
    model = model.to(device)
    hand_options = constraints["hands"]
    arm_weight = hand_options["weights"]["self_collision"] if hand_options["enabled"] else 0.0
    arm_collision = ArmBodyCollision(body_faces, arm_ids, vids.numpy(), device, hand_options) if arm_weight else None
    B = T(Y_TO_Z)
    indices = torch.as_tensor(active, dtype=torch.long, device=device)
    gate = T(envelope[active])
    beta = base["betas"][indices]
    trans = base["transl"][indices]
    root = axis_angle_to_matrix(base["global_orient"][indices])
    body0 = axis_angle_to_matrix(base["body_pose"][indices].reshape(-1, 21, 3))
    raw = torch.nn.Parameter(torch.zeros(len(active), 6, 3, device=device))
    optimizer = torch.optim.Adam([raw], lr=options["learning_rate"])
    limits = bvh_leg_limits(bvh, cfg["bvh_circle_range"], N(model.get_skeleton(beta[:1]))[0], options)
    radius = T(metrics["apparatus"]["radius_m"])
    top = T(metrics["apparatus"]["top_m"])
    dome = T(metrics["apparatus"]["dome_m"])
    image_sigma = constraints["shared"]["image_sigma_px"] * pixel_scale(cfg, arrays["K"])
    K = T(arrays["K"])
    camera = T(arrays["camera_R_zup_to_camera"][active])
    camera_t = T(arrays["camera_t"][active])
    kp = T(arrays["keypoints"][active, :, :2])
    confidence = T(arrays["keypoints"][active, :, 2].clip(0, 1) ** 2)
    # Only the observed legs contribute: root, torso and arms are immutable.
    confidence[:, :13] = 0
    for begin, finish, joint_ids in cfg.get("occluded_keypoints", []):
        selected = torch.as_tensor((active >= begin) & (active < finish), device=device)
        confidence[
            selected[:, None]
            & torch.isin(torch.arange(17, device=device)[None], torch.tensor(joint_ids, device=device))
        ] *= constraints["shared"]["occluded_keypoint_multiplier"]
    baseline_j = T(arrays["corrected_joints_zup"][active])
    baseline_mid = baseline_j[:, [7, 8]].mean(1)
    baseline_direction = torch.nn.functional.normalize(baseline_mid - baseline_j[:, [1, 2]].mean(1), dim=-1)
    lateral = torch.nn.functional.normalize(baseline_j[:, 2] - baseline_j[:, 1], dim=-1)
    collision_ids = torch.where(model.lbs_weights[132:, [1, 2, 4, 5, 7, 8, 10, 11]].sum(-1) > 0.5)[0]

    def average(value):
        return (value * gate).sum() / gate.sum().clamp_min(1)

    def hinges(value, maximum, scale):
        return average((torch.relu(value - maximum) / scale).square())

    def directions(j):
        thigh = j[:, [4, 5]] - j[:, [1, 2]]
        shin = j[:, [7, 8]] - j[:, [4, 5]]
        return torch.nn.functional.normalize(thigh, dim=-1), torch.nn.functional.normalize(shin, dim=-1)

    history = []
    for step in range(iterations):
        optimizer.param_groups[0]["lr"] = options["learning_rate"] * max(0.12, 1 - step / iterations)
        optimizer.zero_grad(set_to_none=True)
        # Bound the correction before gating: the optimizer cannot cancel the fade.
        size = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = (
            raw
            * (options["max_correction_rad"] * torch.tanh(size / options["max_correction_rad"]) / size)
            * gate[:, None, None]
        )
        body = body0.clone()
        body[:, LEG_POSE_IDS] = axis_angle_to_matrix(delta) @ body0[:, LEG_POSE_IDS]
        verts = super(SmplxLiteV437Coco17, model).forward(
            matrix_to_rotation_6d(body).flatten(1), beta, matrix_to_rotation_6d(root), trans, rotation_type="r6d"
        )
        coco = torch.einsum("vj,tvc->tjc", model.smplx2coco17_interestd, verts[:, :132]) @ B.T
        v = verts[:, 132:] @ B.T
        j = fk(model, body, root, trans, beta) @ B.T
        thigh, shin = directions(j)
        gap = (j[:, 7] - j[:, 8]).norm(dim=-1)
        knee_gap = (j[:, 4] - j[:, 5]).norm(dim=-1)
        pair_loss = hinges(gap, limits["ankle_max_m"], 0.06) + options["knee_gap_factor"] * hinges(
            knee_gap, limits["knee_max_m"], 0.06
        )
        parallel = 0
        for directions_, name in [(thigh, "thigh_max_deg"), (shin, "shin_max_deg")]:
            difference = (directions_[:, 0] - directions_[:, 1]).norm(dim=-1)
            parallel += hinges(difference, 2 * np.sin(np.radians(limits[name]) / 2), 0.2)
        # Soft knee straightness and symmetry permit BVH's natural asymmetry.
        cosflex = (thigh * shin).sum(-1)
        flex = average(torch.relu(np.cos(np.radians(limits["flexion_max_deg"])) - cosflex).square().mean(1) / 0.1**2)
        flex_angles = torch.acos(cosflex.clamp(-0.99999, 0.99999))
        flex += hinges(
            (flex_angles[:, 0] - flex_angles[:, 1]).abs(), np.radians(limits["flexion_difference_max_deg"]), 0.2
        )
        # Coarse leg-leg capsule samples and left/right ordering avoid overlap/crossing.
        samples = []
        for side in range(2):
            hip = j[:, 1 + side]
            knee = j[:, 4 + side]
            ankle = j[:, 7 + side]
            samples.append(
                torch.stack(
                    [hip + t * (knee - hip) for t in [0.3, 0.55, 0.8, 1.0]]
                    + [knee + t * (ankle - knee) for t in [0.25, 0.5, 0.75, 1.0]],
                    1,
                )
            )
        distance = torch.cdist(samples[0], samples[1])
        radii = T([0.034] * 4 + [0.027] * 4) * (limits["target_leg_length_m"] / 0.77)
        self_loss = average(
            (torch.relu(radii[:, None] + radii[None, :] - 0.005 - distance) / 0.02).square().mean((1, 2))
        )
        for l, r in [(4, 5), (7, 8)]:
            side_sep = ((j[:, r] - j[:, l]) * lateral).sum(-1)
            self_loss += average((torch.relu(0.035 - side_sep) / 0.04).square())
        uv = project(coco, camera, camera_t, K)
        r2 = ((uv - kp) / image_sigma).square().sum(-1)
        image_loss = ((torch.sqrt(1 + r2) - 1) * confidence).sum() / confidence.sum().clamp_min(1)
        mid = j[:, [7, 8]].mean(1)
        direction = torch.nn.functional.normalize(mid - j[:, [1, 2]].mean(1), dim=-1)
        sweep = average((direction - baseline_direction).square().sum(-1) / 0.2**2)
        center = average((torch.relu((mid - baseline_mid).norm(dim=-1) - 0.05) / 0.15).square())
        vv = v[:, collision_ids]
        vr = vv[:, :, :2].norm(dim=-1)
        cap = top - dome * (vr / radius).square()
        depth = torch.minimum(radius - vr, torch.minimum(cap - vv[:, :, 2], vv[:, :, 2] - 0.08))
        collision = (torch.relu(depth - 0.005) / 0.015).square().topk(20, dim=1).values.mean()
        floor = (torch.relu(-vv[:, :, 2] - 0.003) / 0.015).square().topk(20, dim=1).values.mean()
        padded = torch.zeros(count, 6, 3, device=device).index_copy(0, indices, delta)
        smooth = ((padded[2:] - 2 * padded[1:-1] + padded[:-2]) / 0.055).square().mean()
        joint_delta = torch.zeros(count, 6, 3, device=device).index_copy(
            0, indices, j[:, [4, 5, 7, 8, 10, 11]] - baseline_j[:, [4, 5, 7, 8, 10, 11]]
        )
        pos_smooth = ((joint_delta[2:] - 2 * joint_delta[1:-1] + joint_delta[:-2]) / 0.015).square().mean()
        preservation = (delta / 0.5).square().mean()
        terms = dict(
            gap=pair_loss,
            parallel=parallel,
            knee_flexion=flex,
            self_collision=self_loss,
            image=image_loss,
            sweep_direction=sweep,
            ankle_midpoint=center,
            apparatus_collision=collision,
            ground_collision=floor,
            rotation_smoothness=smooth,
            position_smoothness=pos_smooth,
            pose_preservation=preservation,
        )
        loss = weighted_loss(terms, weights)
        if arm_collision is not None:
            # Legs may move toward an immutable hand during this stage.
            loss = loss + arm_weight * arm_collision(v)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite closed-leg loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10)
        optimizer.step()
        if step % 50 == 0 or step == iterations - 1:
            row = dict(
                iteration=step,
                loss=float(loss.detach()),
                pair=float(pair_loss.detach()),
                parallel=float(parallel.detach()),
                image=float(image_loss.detach()),
                collision=float(collision.detach()),
                self_collision=float(self_loss.detach()),
            )
            history.append(row)
            if callback:
                callback(row)
    result = copy.deepcopy(prediction)
    with torch.no_grad():
        size = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = (
            raw
            * (options["max_correction_rad"] * torch.tanh(size / options["max_correction_rad"]) / size)
            * gate[:, None, None]
        )
        final_rot = axis_angle_to_matrix(delta) @ body0[:, LEG_POSE_IDS]
        final_pose = matrix_to_axis_angle(final_rot).cpu()
        # Copy only the six leg joints at strictly interior active frames. All
        # other stored floats, including both incam/global stage boundaries, stay exact.
        for system in ["smpl_params_global", "smpl_params_incam"]:
            pose = result[system]["body_pose"].reshape(count, 21, 3)
            for k, joint in enumerate(LEG_POSE_IDS):
                pose[active, joint] = final_pose[:, k]
        updated = {k: x.to(device) for k, x in result["smpl_params_global"].items()}
        vc, cc = model(**updated)
        body_all = axis_angle_to_matrix(updated["body_pose"].reshape(-1, 21, 3))
        jj = (
            fk(model, body_all, axis_angle_to_matrix(updated["global_orient"]), updated["transl"], updated["betas"])
            @ B.T
        )
        cv, coco_incam = model(**{k: x.to(device) for k, x in result["smpl_params_incam"].items()})
        arr = dict(arrays)
        met = copy.deepcopy(metrics)
        for key, val in [
            ("corrected_joints_zup", N(jj)),
            ("corrected_coco_zup", N(cc @ B.T)),
            ("corrected_coco_incam", N(coco_incam)),
            ("corrected_surface_zup", N(vc @ B.T)),
        ]:
            new_array = arr[key].copy() if key in arr else val.copy()
            new_array[active] = val[active]
            arr[key] = new_array
        # Compose rotation corrections; never add axis-angle vectors.
        old_delta = axis_angle_to_matrix(T(arrays["body_delta"][active]))
        composed = old_delta.clone()
        composed[:, LEG_POSE_IDS] = axis_angle_to_matrix(delta) @ old_delta[:, LEG_POSE_IDS]
        arr["body_delta"] = arrays["body_delta"].copy()
        composed_aa = N(matrix_to_axis_angle(composed))
        for joint in LEG_POSE_IDS:
            arr["body_delta"][active, joint] = composed_aa[:, joint]
        arr["closed_leg_envelope"] = envelope
        arr["closed_leg_delta"] = np.zeros((count, 6, 3), np.float32)
        arr["closed_leg_delta"][active] = N(delta)
        arr["pre_closed_leg_joints_zup"] = arrays["corrected_joints_zup"].copy()
        arr["pre_closed_leg_body_pose"] = prediction["smpl_params_global"]["body_pose"].numpy().copy()
        met["corrected_reprojection"] = reprojection_metrics(
            arr["corrected_coco_incam"],
            arrays["K"],
            arrays["keypoints"],
            cfg,
            constraints["shared"]["occluded_keypoint_multiplier"],
        )
        met["corrected"] = cycle_metrics(arr["corrected_joints_zup"], arr["cycle_boundaries"])
        met["trajectory_corrected"] = trajectory_statistics(
            arr["corrected_joints_zup"], arr["phase"], arr["cycle_boundaries"]
        )
        transform = np.einsum("tij,tvj->tvi", arr["camera_R_zup_to_camera"], N(vc @ B.T)) + arr["camera_t"][:, None]
        met["camera_world_max_vertex_difference_m"] = float(np.abs(transform - N(cv)).max())
        # Hand joint orientations are invariant; refresh sampled skin gaps to
        # account for the model's tiny nonlocal pose blend-shape effects.
        palm_indices = [
            np.flatnonzero(np.isin(arr["surface_vertex_ids"], arr[f"palm_{side}_vertex_ids"]))
            for side in ["left", "right"]
        ]
        arr["palm_surface_gap_m"] = palm_surface_gaps(
            arr["corrected_surface_zup"], palm_indices, float(radius), float(top), float(dome)
        )
        prep_sel = arr["stage_contact"] > 0.8
        if prep_sel.any():
            met["stage_contact_quality"]["palm_surface_gap_mae_cm"] = float(
                np.abs(arr["palm_surface_gap_m"][prep_sel]).mean() * 100
            )
        met["body_correction_degrees"] = dict(
            median=float(np.median(np.linalg.norm(arr["body_delta"][ks:ke], axis=-1)) * 180 / np.pi),
            max=float(np.linalg.norm(arr["body_delta"][ks:ke], axis=-1).max() * 180 / np.pi),
        )
        unchanged = np.ones(count, bool)
        unchanged[active] = False
        exact = all(
            torch.equal(result[space][key][unchanged], prediction[space][key][unchanged])
            for space in ["smpl_params_global", "smpl_params_incam"]
            for key in base
        )
        if not exact:
            raise AssertionError("Closed-leg stage changed protected frames")
        report = dict(
            limits=limits,
            circle_range=[ss, se],
            full_weight_range=[int(np.flatnonzero(core)[0]), int(np.flatnonzero(core)[-1] + 1)],
            fade_seconds=options["fade_seconds"],
            boundary_hold_frames=2,
            protected_frames_exact=exact,
            before_core=leg_statistics(arrays["corrected_joints_zup"], core),
            after_core=leg_statistics(arr["corrected_joints_zup"], core),
            before_circle=leg_statistics(arrays["corrected_joints_zup"], slice(ss, se)),
            after_circle=leg_statistics(arr["corrected_joints_zup"], slice(ss, se)),
            max_wrist_joint_change_m=float(
                np.abs(arr["corrected_joints_zup"][:, [20, 21]] - arrays["corrected_joints_zup"][:, [20, 21]]).max()
            ),
            history=history,
            iterations=iterations,
            self_collision_model="Coarse leg capsule samples, not exact mesh self-intersection detection",
        )
        met["constraints_sha256"] = constraint_hash(constraints)
        met["closed_legs"] = report
        met["active_refinement"]["closed_legs"] = True
    return result, arr, met
