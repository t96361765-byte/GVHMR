"""Circle-only ankle orientation refinement using an anatomical BVH foot prior."""

import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.vertex_ids import vertex_ids
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteV437Coco17
from hmr4d.utils.mushroom_refine import Y_TO_Z, fk, project, reprojection_metrics
from hmr4d.utils.mushroom_contact import palm_surface_gaps
from hmr4d.utils.mushroom_priors import circle_envelope
from hmr4d.utils.mushroom_config import resolve_constraints, pixel_scale, weighted_loss, constraint_hash

ANKLE_POSE_IDS = [6, 7]
FOOT_LANDMARK_IDS = np.array([[vertex_ids["smplx"][s + k] for k in ["BigToe", "SmallToe", "Heel"]] for s in ["L", "R"]])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-10)


def shin_frames(joints, hips=(1, 2), knees=(4, 5), ankles=(7, 8)):
    """Right, forward, proximal axes; independent of global body heading."""
    proximal = unit(joints[:, knees] - joints[:, ankles])
    right = unit(joints[:, hips[1]] - joints[:, hips[0]])[:, None]
    right = unit(right - (right * proximal).sum(-1, keepdims=True) * proximal)
    forward = unit(np.cross(proximal, right))
    frame = np.stack([right, forward, proximal], -1)
    if np.any(np.linalg.det(frame) < 0.99):
        raise ValueError("Degenerate shin anatomical frame")
    return frame


def bvh_foot_prior(bvh, span, options=None):
    options = options or resolve_constraints()["feet"]
    names = bvh["names"]
    j = bvh["joints"][slice(*span)]
    ids = lambda suffix: tuple(names.index(s + suffix) for s in ["Left", "Right"])
    frame = shin_frames(j, ids("Hip"), ids("Knee"), ids("Ankle"))
    world = []
    references = []
    spread = []
    for side, name in enumerate(["LeftToe", "RightToe"]):
        if name not in bvh["ends"]:
            raise ValueError(f"BVH needs a nonzero {name} end site")
        forward = unit(np.asarray(bvh["ends"][name], float))
        normal = unit(np.array([0.0, 0.0, 1.0]) - forward * forward[2])
        if np.linalg.norm(forward) < 0.99 or np.linalg.norm(normal) < 0.99:
            raise ValueError("Invalid BVH toe end direction")
        rest = np.stack([np.cross(forward, normal), forward, normal], -1)
        foot = bvh["rotations"][slice(*span), names.index(name)] @ rest
        world.append(foot)
        relative = np.swapaxes(frame[:, side], -1, -2) @ foot
        mean = Rotation.from_matrix(relative).mean().as_matrix()
        errors = np.degrees(Rotation.from_matrix(relative @ mean.T).magnitude())
        inliers = errors <= np.percentile(errors, 90)
        mean = Rotation.from_matrix(relative[inliers]).mean().as_matrix()
        references.append(mean)
        spread.append(float(np.percentile(errors, 95)))
    world = np.stack(world, 1)
    pair = np.degrees(np.arccos(np.clip((world[:, 0, :, 1] * world[:, 1, :, 1]).sum(-1), -1, 1)))
    return np.stack(references), dict(
        reference_range=list(span),
        relative_orientation_p95_deg=spread,
        pair_tolerance_deg=float(np.percentile(pair, 95) + options["pair_angle_margin_deg"]),
        note="Toe end-site direction and rest Z-up dorsal axis; unpaired anatomical orientation prior, not frame targets.",
    )


def foot_geometry(vertices):
    """Vertices: (..., side, [big toe, small toe, heel], xyz)."""
    normalize = lambda x: torch.nn.functional.normalize(x, dim=-1)
    forward = normalize(vertices[..., :2, :].mean(-2) - vertices[..., 2, :])
    normal = normalize(
        torch.cross(vertices[..., 0, :] - vertices[..., 2, :], vertices[..., 1, :] - vertices[..., 2, :], dim=-1)
    )
    # Vertex winding differs between left and right; calibrate in the rest mesh.
    return forward, normal


def foot_statistics(points, joints, selection):
    p = points[selection]
    j = joints[selection]
    forward = unit(p[:, :, :2].mean(2) - p[:, :, 2])
    shin = unit(j[:, [7, 8]] - j[:, [4, 5]])
    angle = lambda a, b: np.degrees(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))
    summary = lambda x: dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
    return dict(
        foot_pair_angle_deg=summary(angle(forward[:, 0], forward[:, 1])),
        shin_foot_angle_deg=[summary(angle(forward[:, i], shin[:, i])) for i in range(2)],
        toe_center_gap_cm=summary(np.linalg.norm(p[:, 0, :2].mean(1) - p[:, 1, :2].mean(1), axis=-1) * 100),
    )


def refine_circle_feet(prediction, arrays, metrics, bvh, cfg, iterations=None, device="cuda", callback=None):
    constraints = resolve_constraints(cfg)
    options = constraints["feet"]
    weights = options["weights"]
    if not options["enabled"]:
        met = copy.deepcopy(metrics)
        met.setdefault("active_refinement", {}).setdefault("feet", False)
        return copy.deepcopy(prediction), dict(arrays), met
    iterations = options["iterations"] if iterations is None else iterations
    options["iterations"] = iterations
    torch.set_num_threads(4)
    torch.manual_seed(constraints["shared"]["seed"])
    n = len(prediction["smpl_params_global"]["body_pose"])
    ss, se = cfg["circle_range"]
    ks, ke = cfg["keep_range"]
    if not 0 <= ks <= ss < se <= ke <= n or iterations <= 0:
        raise ValueError("Invalid stage range or iterations")
    envelope = circle_envelope(n, ss, se, cfg["fps"], options["fade_seconds"])
    active = np.flatnonzero(envelope > 0)
    core = envelope >= 1 - 1e-8
    if not core.any():
        raise ValueError("Circle interval too short")
    N = lambda x: x.detach().cpu().numpy()
    T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    params = {k: v.to(device) for k, v in prediction["smpl_params_global"].items()}
    model = SmplxLiteV437Coco17().eval()
    dense = SmplxLite().eval()
    # Foot surfaces, COCO regressor and anatomical markers only during optimization.
    foot_weights = dense.lbs_weights[:, [7, 8, 10, 11]].sum(-1)
    foot_ids = (
        torch.unique(torch.cat([torch.where(foot_weights > 0.35)[0], torch.tensor(FOOT_LANDMARK_IDS.ravel())]))
        .sort()
        .values
    )
    for name in ["v_template", "shapedirs", "lbs_weights"]:
        setattr(model, name, torch.cat([getattr(model, name)[:132], getattr(dense, name)[foot_ids]], 0))
    model.posedirs = torch.cat([model.posedirs[:, :132], dense.posedirs[:, foot_ids]], 1)
    model = model.to(device)
    marker_ids = np.array([[int(torch.where(foot_ids == i)[0][0]) for i in row] for row in FOOT_LANDMARK_IDS])
    B = T(Y_TO_Z)
    gate = T(envelope[active])
    idx = torch.tensor(active, device=device)
    beta = params["betas"][idx]
    trans = params["transl"][idx]
    root = axis_angle_to_matrix(params["global_orient"][idx])
    body0 = axis_angle_to_matrix(params["body_pose"][idx].reshape(-1, 21, 3))
    rest = model.v_template[132:] + torch.einsum("vck,k->vc", model.shapedirs[132:], beta[0])
    _, rest_normals = foot_geometry(rest[marker_ids])
    signs = torch.sign(rest_normals[:, 1])[None, :, None]
    reference, info = bvh_foot_prior(bvh, cfg["bvh_circle_range"], options)
    frame = shin_frames(arrays["corrected_joints_zup"])
    target = T(frame[active] @ reference[None])
    target_forward = target[:, :, :, 1]
    target_normal = target[:, :, :, 2]
    raw = torch.nn.Parameter(torch.zeros(len(active), 2, 3, device=device))
    optimizer = torch.optim.Adam([raw], lr=options["learning_rate"])
    image_sigma = constraints["shared"]["image_sigma_px"] * pixel_scale(cfg, arrays["K"])
    K = T(arrays["K"])
    cam = T(arrays["camera_R_zup_to_camera"][active])
    cam_t = T(arrays["camera_t"][active])
    kp = T(arrays["keypoints"][active, :, :2])
    confidence = T(arrays["keypoints"][active, :, 2].clip(0, 1) ** 2)
    confidence[:, :15] = 0
    for begin, finish, joint_ids in cfg.get("occluded_keypoints", []):
        selected = torch.as_tensor((active >= begin) & (active < finish), device=device)
        confidence[
            selected[:, None]
            & torch.isin(torch.arange(17, device=device)[None], torch.tensor(joint_ids, device=device))
        ] *= constraints["shared"]["occluded_keypoint_multiplier"]
    radius = metrics["apparatus"]["radius_m"]
    top = metrics["apparatus"]["top_m"]
    dome = metrics["apparatus"]["dome_m"]
    tolerance = options["orientation_tolerance_deg"]
    if not 0 < tolerance < 45:
        raise ValueError("foot_orientation_tolerance_deg must be between 0 and 45")
    average = lambda v: (v * gate).sum() / gate.sum()

    def pose():
        size = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = (
            raw
            * (options["max_correction_rad"] * torch.tanh(size / options["max_correction_rad"]) / size)
            * gate[:, None, None]
        )
        body = body0.clone()
        body[:, ANKLE_POSE_IDS] = axis_angle_to_matrix(delta) @ body0[:, ANKLE_POSE_IDS]
        return body, delta

    def cone(a, b, degrees):
        d = (a - b).norm(dim=-1)
        return average((torch.relu(d - 2 * np.sin(np.radians(degrees) / 2)) / 0.15).square().mean(-1))

    history = []
    for step in range(iterations):
        optimizer.param_groups[0]["lr"] = options["learning_rate"] * max(0.15, 1 - step / iterations)
        optimizer.zero_grad(set_to_none=True)
        body, delta = pose()
        vv = super(SmplxLiteV437Coco17, model).forward(
            matrix_to_rotation_6d(body).flatten(1), beta, matrix_to_rotation_6d(root), trans, rotation_type="r6d"
        )
        v = vv[:, 132:] @ B.T
        points = v[:, marker_ids]
        forward, normal = foot_geometry(points)
        normal = normal * signs
        orientation = cone(forward, target_forward, tolerance) + options["normal_factor"] * cone(
            normal, target_normal, tolerance + options["normal_tolerance_extra_deg"]
        )
        # A small central attraction suppresses flutter inside the allowed cone.
        orientation += options["central_attraction_factor"] * average(
            ((forward - target_forward) / 0.2).square().mean((1, 2))
        )
        pair = cone(forward[:, :1], forward[:, 1:], info["pair_tolerance_deg"])
        toe_mid = points[:, :, :2].mean(2)
        heel = points[:, :, 2]
        centerline = torch.stack([heel + t * (toe_mid - heel) for t in np.linspace(0.1, 0.95, 8)], 2)
        distance = torch.cdist(centerline[:, 0], centerline[:, 1])
        # Capsule widths are a conservative proxy, not exact mesh collision.
        widths = (rest[marker_ids[:, 0]] - rest[marker_ids[:, 1]]).norm(dim=-1)
        clearance = 0.78 * widths.mean()
        self_collision = average((torch.relu(clearance - distance) / 0.02).square().mean((1, 2)))
        lateral = T(frame[active, 0, :, 0])
        sep = ((toe_mid[:, 1] - toe_mid[:, 0]) * lateral).sum(-1)
        self_collision += average((torch.relu(0.04 - sep) / 0.025).square())
        ankle_gap = T(
            np.linalg.norm(
                arrays["corrected_joints_zup"][active, 7] - arrays["corrected_joints_zup"][active, 8], axis=-1
            )
        )
        gap = average(
            (
                torch.relu((toe_mid[:, 0] - toe_mid[:, 1]).norm(dim=-1) - ankle_gap - options["toe_gap_margin_m"])
                / 0.04
            ).square()
        )
        rad = v[:, :, :2].norm(dim=-1)
        cap = top - dome * (rad / radius).square()
        depth = torch.minimum(radius - rad, torch.minimum(cap - v[:, :, 2], v[:, :, 2] - 0.08))
        collision = (torch.relu(depth - 0.004) / 0.012).square().topk(20, dim=1).values.mean()
        # Late-circle feet may already approach the floor: pointing must yield
        # before the dismount, without moving the ankle or changing ending poses.
        floor = (
            (torch.relu(-v[:, :, 2] - options["ground_tolerance_m"]) / options["ground_sigma_m"])
            .square()
            .topk(20, dim=1)
            .values.mean()
        )
        coco = torch.einsum("vj,tvc->tjc", model.smplx2coco17_interestd, vv[:, :132]) @ B.T
        uv = project(coco, cam, cam_t, K)
        r2 = ((uv - kp) / image_sigma).square().sum(-1)
        image = ((torch.sqrt(1 + r2) - 1) * confidence).sum() / confidence.sum().clamp_min(1)
        padded = torch.zeros(n, 2, 3, device=device).index_copy(0, idx, delta)
        smooth = ((padded[2:] - 2 * padded[1:-1] + padded[:-2]) / 0.065).square().mean()
        local = torch.einsum("tsji,tsj->tsi", T(frame[active]), forward)
        local_smooth = ((local[1:] - local[:-1]) / 0.07).square().mean(-1)
        stable = (local_smooth * torch.minimum(gate[1:], gate[:-1])[:, None]).mean()
        preserve = (delta / 1.0).square().mean()
        terms = dict(
            orientation=orientation,
            parallel=pair,
            toe_gap=gap,
            self_collision=self_collision,
            apparatus_collision=collision,
            ground_collision=floor,
            image=image,
            rotation_smoothness=smooth,
            relative_foot_stability=stable,
            pose_preservation=preserve,
        )
        loss = weighted_loss(terms, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite foot refinement loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 10)
        optimizer.step()
        if step % 50 == 0 or step == iterations - 1:
            row = dict(
                iteration=step,
                loss=float(loss.detach()),
                orientation=float(orientation.detach()),
                pair=float(pair.detach()),
                self_collision=float(self_collision.detach()),
                collision=float(collision.detach()),
                floor=float(floor.detach()),
            )
            history.append(row)
            if callback:
                callback(row)
    result = copy.deepcopy(prediction)
    arr = dict(arrays)
    met = copy.deepcopy(metrics)
    with torch.no_grad():
        body, delta = pose()
        final = matrix_to_axis_angle(body[:, ANKLE_POSE_IDS]).cpu()
        for space in ["smpl_params_global", "smpl_params_incam"]:
            for k, joint in enumerate(ANKLE_POSE_IDS):
                result[space]["body_pose"].reshape(n, 21, 3)[active, joint] = final[:, k]
        # Evaluate complete surfaces in small batches; retain only small arrays by default.
        dense = dense.to(device)
        all_points = {}
        cocos = {}
        surfaces = {}
        for name, p in [
            ("before", prediction["smpl_params_global"]),
            ("after", result["smpl_params_global"]),
            ("incam", result["smpl_params_incam"]),
        ]:
            p = {k: x.to(device) for k, x in p.items()}
            verts = []
            cc = []
            for s in range(0, n, 24):
                part = {k: x[s : s + 24] for k, x in p.items()}
                verts.append(N(dense(**part)))
                cc.append(N(model(**part)[1]))
            verts = np.concatenate(verts)
            cocos[name] = np.concatenate(cc)
            if name != "incam":
                verts = verts @ Y_TO_Z.T
            all_points[name] = verts[:, FOOT_LANDMARK_IDS]
            surfaces[name] = verts[:, arrays["surface_vertex_ids"]]
        p = {k: x.to(device) for k, x in result["smpl_params_global"].items()}
        jj = N(
            fk(
                model,
                axis_angle_to_matrix(p["body_pose"].reshape(-1, 21, 3)),
                axis_angle_to_matrix(p["global_orient"]),
                p["transl"],
                p["betas"],
            )
            @ B.T
        )
        arr["corrected_joints_zup"] = arrays["corrected_joints_zup"].copy()
        for joint in [10, 11]:
            arr["corrected_joints_zup"][active, joint] = jj[active, joint]
        for key, val in [
            ("corrected_coco_zup", cocos["after"] @ Y_TO_Z.T),
            ("corrected_coco_incam", cocos["incam"]),
            ("corrected_surface_zup", surfaces["after"]),
        ]:
            arr[key] = arrays[key].copy() if key in arrays else val.copy()
            arr[key][active] = val[active]
        arr["body_delta"] = arrays["body_delta"].copy()
        old_delta = axis_angle_to_matrix(T(arrays["body_delta"][active][:, ANKLE_POSE_IDS]))
        composed = N(matrix_to_axis_angle(axis_angle_to_matrix(delta) @ old_delta))
        for k, joint in enumerate(ANKLE_POSE_IDS):
            arr["body_delta"][active, joint] = composed[:, k]
        arr["foot_envelope"] = envelope
        arr["foot_delta"] = np.zeros((n, 2, 3), np.float32)
        arr["foot_delta"][active] = N(delta)
        arr["pre_foot_landmarks_zup"] = all_points["before"]
        arr["foot_landmarks_zup"] = all_points["after"]
        arr["foot_target_frames_zup"] = frame @ reference[None]
        palm_indices = [
            np.flatnonzero(np.isin(arr["surface_vertex_ids"], arr[f"palm_{side}_vertex_ids"]))
            for side in ["left", "right"]
        ]
        arr["palm_surface_gap_m"] = palm_surface_gaps(
            surfaces["after"], palm_indices, float(radius), float(top), float(dome)
        )
        met["corrected_reprojection"] = reprojection_metrics(
            cocos["incam"], arrays["K"], arrays["keypoints"], cfg, constraints["shared"]["occluded_keypoint_multiplier"]
        )
        tr = np.einsum("tij,tvj->tvi", arr["camera_R_zup_to_camera"], surfaces["after"]) + arr["camera_t"][:, None]
        met["camera_world_max_vertex_difference_m"] = float(np.abs(tr - surfaces["incam"]).max())
        prep = arr["stage_contact"] > 0.8
        if prep.any():
            met["stage_contact_quality"]["palm_surface_gap_mae_cm"] = float(
                np.abs(arr["palm_surface_gap_m"][prep]).mean() * 100
            )
        magnitudes = np.linalg.norm(arr["body_delta"][ks:ke], axis=-1) * 180 / np.pi
        met["body_correction_degrees"] = dict(median=float(np.median(magnitudes)), max=float(magnitudes.max()))
        unchanged = envelope == 0
        other = [i for i in range(21) if i not in ANKLE_POSE_IDS]
        for space in ["smpl_params_global", "smpl_params_incam"]:
            for key in prediction[space]:
                assert torch.equal(result[space][key][unchanged], prediction[space][key][unchanged])
                if key != "body_pose":
                    assert torch.equal(result[space][key], prediction[space][key])
            assert torch.equal(
                result[space]["body_pose"].reshape(n, 21, 3)[:, other],
                prediction[space]["body_pose"].reshape(n, 21, 3)[:, other],
            )
        met["constraints_sha256"] = constraint_hash(constraints)
        met["feet"] = dict(
            prior=info,
            circle_range=[ss, se],
            full_weight_range=[int(np.flatnonzero(core)[0]), int(np.flatnonzero(core)[-1] + 1)],
            fade_seconds=options["fade_seconds"],
            orientation_tolerance_deg=tolerance,
            changed_body_pose_indices=ANKLE_POSE_IDS,
            protected_parameters_exact=True,
            knee_ankle_positions_max_change_m=float(
                np.abs(jj[:, [4, 5, 7, 8]] - arrays["corrected_joints_zup"][:, [4, 5, 7, 8]]).max()
            ),
            before_core=foot_statistics(all_points["before"], arrays["corrected_joints_zup"], core),
            after_core=foot_statistics(all_points["after"], jj, core),
            iterations=iterations,
            history=history,
            collision_note="Foot capsules and fitted apparatus proxy; not exact self-intersection guarantee.",
        )
        met["active_refinement"]["feet"] = True
    return result, arr, met
