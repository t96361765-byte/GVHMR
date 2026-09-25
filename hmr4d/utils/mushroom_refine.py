"""Offline, fixed-camera mushroom refinement. Network weights are never changed.

The optimizer uses the source performer's shape and timing. An unpaired BVH
supplies periodic relative geometry and contact timing, not framewise targets.
All internal scene coordinates are metres, Z-up; exported SMPL-X stays Y-up.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.body_model.smplx_lite import SmplxLite, SmplxLiteV437Coco17, batch_rigid_transform_v2
from hmr4d.utils.mushroom_priors import prepare_stage_prior, trajectory_statistics
from hmr4d.utils.mushroom_contact import (
    support_weights,
    palm_geometry,
    cap_inward_normal,
    palm_orientation_loss,
    palm_surface_gaps,
)
from hmr4d.utils.mushroom_config import resolve_constraints, pixel_scale, weighted_loss, constraint_hash

Y_TO_Z = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
BODY_MAP = {
    0: "Hips",
    1: "LeftHip",
    2: "RightHip",
    4: "LeftKnee",
    5: "RightKnee",
    7: "LeftAnkle",
    8: "RightAnkle",
    10: "LeftToe",
    11: "RightToe",
    12: "Neck",
    15: "Head",
    16: "LeftShoulder",
    17: "RightShoulder",
    18: "LeftElbow",
    19: "RightElbow",
    20: "LeftWrist",
    21: "RightWrist",
}


def read_bvh(path):
    """Preserve source hierarchy; honor declared intrinsic Euler channel order."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    mi = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    names, parents, offsets, channels, ends = [], [], [], [], {}
    active, in_end = -1, False
    for line in lines[:mi]:
        s = line.split()
        if not s:
            continue
        if s[0] in ("ROOT", "JOINT"):
            names.append(s[1])
            parents.append(active)
            offsets.append([0.0, 0, 0])
            channels.append([])
            active = len(names) - 1
        elif s[:2] == ["End", "Site"]:
            in_end = True
        elif s[0] == "OFFSET":
            if in_end:
                ends[names[active]] = list(map(float, s[1:]))
            else:
                offsets[active] = list(map(float, s[1:]))
        elif s[0] == "CHANNELS":
            channels[active] = s[2:]
            if len(channels[active]) != int(s[1]):
                raise ValueError("Invalid BVH channel count")
        elif s[0] == "}":
            if in_end:
                in_end = False
            else:
                active = parents[active]
    n = int(lines[mi + 1].split(":")[1])
    dt = float(lines[mi + 2].split(":")[1])
    values = np.fromstring(" ".join(lines[mi + 3 :]), sep=" ")
    if dt <= 0 or len(values) != n * sum(map(len, channels)) or not np.isfinite(values).all():
        raise ValueError("Invalid BVH motion data")
    values = values.reshape(n, -1)
    positions = []
    rotations = []
    cursor = 0
    for j, ch in enumerate(channels):
        pos = np.tile(offsets[j], (n, 1))
        angles, order = [], ""
        for c in ch:
            if c.endswith("position"):
                pos[:, "XYZ".index(c[0])] += values[:, cursor]
            elif c.endswith("rotation"):
                order += c[0]
                angles.append(values[:, cursor])
            else:
                raise ValueError(f"Unsupported BVH channel {c}")
            cursor += 1
        rot = Rotation.from_euler(order, np.stack(angles, -1), degrees=True).as_matrix()
        if parents[j] >= 0:
            par = parents[j]
            pos = positions[par] + np.einsum("tij,tj->ti", rotations[par], pos)
            rot = rotations[par] @ rot
        positions.append(pos)
        rotations.append(rot)
    if not set(BODY_MAP.values()).issubset(names):
        raise ValueError("BVH is missing required named body joints")
    offsets = np.array(offsets)
    if offsets[names.index("Chest"), 2] <= 0 or not 0.1 < np.linalg.norm(offsets[names.index("LeftKnee")]) < 0.8:
        raise ValueError("This reference must be metre-valued and Z-up; convert other conventions explicitly")
    return dict(
        names=names,
        parents=parents,
        offsets=offsets,
        ends=ends,
        joints=np.stack(positions, 1),
        rotations=np.stack(rotations, 1),
        fps=1 / dt,
    )


def phase_and_cuts(joints, start, end, wrists=(20, 21), ankles=(7, 8)):
    vector = joints[:, ankles].mean(1) - joints[:, wrists].mean(1)
    angle = np.unwrap(np.arctan2(vector[:, 1], vector[:, 0]))
    angle = gaussian_filter1d(angle, 0.6)
    direction = np.sign(np.median(np.diff(angle[start:end]))) or 1
    progress = direction * angle
    # Each boundary has the same apparatus-relative angle (positive X axis).
    levels = (
        np.arange(np.ceil(progress[start] / (2 * np.pi)), np.floor(progress[end - 1] / (2 * np.pi)) + 1) * 2 * np.pi
    )
    cuts = []
    for level in levels:
        hits = np.flatnonzero((progress[start : end - 1] < level) & (progress[start + 1 : end] >= level)) + start + 1
        if len(hits):
            cuts.append(int(hits[0]))
    return angle, cuts, int(direction)


def fk(model, body_matrices, root_matrix, transl, betas, return_rotations=False):
    sk = model.get_skeleton(betas)
    full = torch.cat([root_matrix[:, None], body_matrices], 1)
    # Only the first 22 joints are needed for kinematic/contact losses.
    j, transforms = batch_rigid_transform_v2(full, sk[:, :22], model.parents[:22])
    j = j + transl[:, None]
    return (j, transforms[:, :, :3, :3]) if return_rotations else j


def transform_camera(points, rotation, translation):
    if rotation.ndim == 3:
        return torch.einsum("tij,tnj->tni", rotation, points) + translation[:, None]
    return points @ rotation.T + translation


def project(points, rotation, translation, K):
    cam = transform_camera(points, rotation, translation)
    uv = cam[..., :2] / cam[..., 2:].clamp_min(0.2)
    return uv * torch.stack([K[0, 0], K[1, 1]]) + K[:2, 2]


def reprojection_metrics(coco, K, keypoints, cfg, occluded_multiplier):
    """Common reporting convention for base, leg and foot stages."""
    uv = coco[..., :2] / coco[..., 2:] * K[[0, 1], [0, 1]] + K[:2, 2]
    error = np.linalg.norm(uv - keypoints[:, :, :2], axis=-1)
    weight = keypoints[:, :, 2].clip(0, 1) ** 2
    start, end = cfg.get("keep_range", [0, len(keypoints)])
    weight[:start] = weight[end:] = 0
    weight[:, :5] = 0
    for start, end, ids in cfg.get("occluded_keypoints", []):
        weight[start:end, ids] *= occluded_multiplier
    return dict(
        weighted_mean_px=float((error * weight).sum() / weight.sum()),
        median_confident_px=float(np.median(error[weight > 0.25])),
        p95_confident_px=float(np.percentile(error[weight > 0.25], 95)),
    )


def periodic_reference(bvh, source_joints, cfg, phase):
    """Phase averaging repeats cycles, never stretches an entire BVH to the video."""
    names = bvh["names"]
    bj = bvh["joints"]
    fps = bvh["fps"]
    wr = [names.index("LeftWrist"), names.index("RightWrist")]
    an = [names.index("LeftAnkle"), names.index("RightAnkle")]
    bs, be = cfg["bvh_circle_range"]
    bphase, cuts, direction = phase_and_cuts(bj, bs, be, wr, an)
    if len(cuts) < 3:
        raise ValueError("BVH reference interval must contain at least two complete cycles")
    mapped = np.zeros((len(bj), 22, 3))
    for j, name in BODY_MAP.items():
        mapped[:, j] = bj[:, names.index(name)]
    ss, se = cfg["circle_range"]
    leg = lambda x: np.linalg.norm(x[:, 1] - x[:, 4], axis=-1) + np.linalg.norm(x[:, 4] - x[:, 7], axis=-1)
    scale = np.median(leg(source_joints[ss:se])) / np.median(leg(mapped[bs:be]))
    center = mapped[cuts[0] : cuts[-1], [20, 21]].mean((0, 1))
    speed = np.linalg.norm(np.gradient(bj[:, wr], 1 / fps, axis=0), axis=-1)
    contact = np.exp(-((speed / resolve_constraints(cfg)["hands"]["detection"]["bvh_contact_speed_m_s"]) ** 2))
    grid = np.linspace(0, 2 * np.pi, 129)
    samples, contacts = [], []
    for s, e in zip(cuts[:-1], cuts[1:]):
        xp = direction * (bphase[s : e + 1] - bphase[s])
        xp = np.maximum.accumulate(xp)
        # Preserve absolute horizontal phase, including the small crossing offset.
        yaw = -bphase[s]
        rz = Rotation.from_euler("z", yaw).as_matrix()
        rel = (mapped[s : e + 1] - center) @ rz.T
        arr = np.stack([np.interp(grid, xp, col) for col in rel.reshape(len(rel), -1).T], -1).reshape(129, 22, 3)
        samples.append(arr)
        contacts.append(np.stack([np.interp(grid, xp, col) for col in contact[s : e + 1].T], -1))
    template = np.mean(samples, 0)
    ctemplate = np.mean(contacts, 0)
    template[-1] = template[0]
    ctemplate[-1] = ctemplate[0]
    query = np.mod(direction * phase, 2 * np.pi)
    ref = (
        np.stack([np.interp(query, grid, col) for col in template.reshape(129, -1).T], -1).reshape(len(phase), 22, 3)
        * scale
    )
    contact = np.stack([np.interp(query, grid, col) for col in ctemplate.T], -1)
    # A reference can have two moving wrists briefly; ensure at least one soft support.
    contact /= np.maximum(contact.max(1, keepdims=True), 0.05)
    return ref, contact, dict(scale=float(scale), cycle_boundaries=cuts, direction=direction)


def cycle_metrics(joints, cuts):
    centers = np.array([joints[s:e, [20, 21]].mean((0, 1)) for s, e in zip(cuts[:-1], cuts[1:])])
    pelvis = np.array([joints[s:e, 0].mean(0) for s, e in zip(cuts[:-1], cuts[1:])])
    return dict(
        wrist_centers_m=centers.tolist(),
        pelvis_centers_m=pelvis.tolist(),
        wrist_horizontal_first_last_cm=float(np.linalg.norm((centers[-1] - centers[0])[:2]) * 100),
        wrist_vertical_first_last_cm=float((centers[-1] - centers[0])[2] * 100),
        pelvis_horizontal_first_last_cm=float(np.linalg.norm((pelvis[-1] - pelvis[0])[:2]) * 100),
    )


def fit_camera(coco, kp, K, initial_rotation, initial_translation, frames, scale=1.0):
    def residual(x):
        pts = coco[frames] @ Rotation.from_rotvec(x[:3]).as_matrix().T + x[3:]
        uv = pts[..., :2] / np.maximum(pts[..., 2:], 0.2) * K[[0, 1], [0, 1]] + K[:2, 2]
        return ((uv[:, 5:] - kp[frames, 5:, :2]) * np.sqrt(np.clip(kp[frames, 5:, 2:3], 0, 1))).ravel()

    result = least_squares(
        residual,
        np.r_[Rotation.from_matrix(initial_rotation).as_rotvec(), initial_translation],
        loss="soft_l1",
        f_scale=15 * scale,
        max_nfev=100,
    )
    return Rotation.from_rotvec(result.x[:3]).as_matrix(), result.x[3:]


def refine(prediction, kp, bvh, cfg, iterations=None, device="cuda", callback=None, camera_motion=None):
    constraints = resolve_constraints(cfg)
    base_options = constraints["base"]
    hand_options = constraints["hands"]
    detection = hand_options["detection"]
    base_weights = base_options["weights"]
    hand_weights = {k: (v if hand_options["enabled"] else 0.0) for k, v in hand_options["weights"].items()}
    iterations = base_options["iterations"] if iterations is None else iterations
    if iterations <= 0:
        raise ValueError("Base iterations must be positive")
    base_options["iterations"] = iterations
    torch.set_num_threads(4)
    torch.manual_seed(constraints["shared"]["seed"])
    model = SmplxLiteV437Coco17().eval()
    # Dense surface coverage is needed near wrists/feet: the 437 preview vertices
    # miss narrow but deep penetrations. Retain the exact 132-vertex COCO regressor.
    dense = SmplxLite()
    extremity = dense.lbs_weights[:, [7, 8, 10, 11, 20, 21] + list(range(25, 55))].sum(-1) > 0.25
    vids = torch.unique(
        torch.cat(
            [torch.arange(0, len(dense.v_template), int(cfg.get("surface_stride", 3))), torch.where(extremity)[0]]
        )
    )
    for name in ["v_template", "shapedirs", "lbs_weights"]:
        setattr(model, name, torch.cat([getattr(model, name)[:132], getattr(dense, name)[vids]], 0))
    model.posedirs = torch.cat([model.posedirs[:, :132], dense.posedirs[:, vids]], 1)
    del dense
    model = model.to(device)

    def T(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    def N(a):
        return a.detach().cpu().numpy()

    pg = {k: v.to(device) for k, v in prediction["smpl_params_global"].items()}
    pc = {k: v.to(device) for k, v in prediction["smpl_params_incam"].items()}
    length = len(pg["body_pose"])
    fps = float(cfg.get("fps", 30))
    ss, se = cfg["circle_range"]
    ks, ke = cfg.get("keep_range", [0, length])
    if not 0 <= ks <= ss < se <= ke <= length:
        raise ValueError("Invalid keep/circle frame ranges (zero-based, end exclusive)")
    B = T(Y_TO_Z)
    K = prediction["K_fullimg"][0].to(device)
    px_scale = pixel_scale(cfg, N(K))
    image_sigma = constraints["shared"]["image_sigma_px"] * px_scale
    relative_camera = T(np.tile(np.eye(3), (length, 1, 1)) if camera_motion is None else camera_motion)
    if relative_camera.shape != (length, 3, 3):
        raise ValueError("Background camera track length mismatch")
    rays = np.concatenate([kp[:, :, :2], np.ones((*kp.shape[:2], 1))], -1) @ np.linalg.inv(N(K)).T
    stabilized_rays = np.einsum("tji,tnj->tni", N(relative_camera), rays)
    stable_kp = kp.copy()
    stable_kp[:, :, :2] = stabilized_rays[:, :, :2] / stabilized_rays[:, :, 2:] * N(K)[[0, 1], [0, 1]] + N(K)[:2, 2]
    if not torch.allclose(prediction["K_fullimg"], prediction["K_fullimg"][:1].expand_as(prediction["K_fullimg"])):
        raise ValueError("Changing intrinsics are not supported by this fixed-camera refinement")
    beta = pg["betas"]
    if not torch.allclose(beta, beta[:1].expand_as(beta), atol=1e-5):
        raise ValueError("Expected constant source shape")
    with torch.no_grad():
        source_v, source_c = model(**pg)
        _, cam_c = model(**pc)
        body0 = axis_angle_to_matrix(pg["body_pose"].reshape(length, 21, 3))
        root0 = axis_angle_to_matrix(pg["global_orient"])
        j0 = fk(model, body0, root0, pg["transl"], beta) @ B.T
        source_c = source_c @ B.T
        source_v = source_v @ B.T
    jn = N(j0)
    phase, cuts, direction = phase_and_cuts(jn, ss, se)
    if len(cuts) < 3:
        raise ValueError("Circle range needs at least two complete revolutions for drift estimation")
    ref, contact, ref_info = periodic_reference(bvh, jn, cfg, phase)
    if direction != ref_info["direction"]:
        raise ValueError("Video and BVH turn in opposite directions; an explicit left/right mirror is needed")
    centers = np.array([jn[s:e, [20, 21]].mean((0, 1)) for s, e in zip(cuts[:-1], cuts[1:])])
    times = np.array([(s + e - 1) / 2 for s, e in zip(cuts[:-1], cuts[1:])])
    center = centers.mean(0)
    drift = np.stack([np.interp(np.arange(length), times, centers[:, i]) for i in range(3)], -1) - center
    # Normalize apparatus horizontal origin once, never align cycles independently.
    support_height = float(cfg.get("initial_wrist_height_m", 0.70))
    shift = np.array([center[0], center[1], center[2] - support_height])
    initial_j = jn - drift[:, None] - shift
    initial_c = N(source_c) - drift[:, None] - shift
    p0 = initial_j[:, 0]
    Rmean = (
        Rotation.from_matrix(N(axis_angle_to_matrix(pc["global_orient"]) @ root0.transpose(-1, -2) @ B.T)[ks:ke])
        .mean()
        .as_matrix()
    )
    tmean = (N(cam_c)[ks:ke] - initial_c[ks:ke] @ Rmean.T).mean((0, 1))
    rcam, tcam = fit_camera(initial_c, stable_kp, N(K), Rmean, tmean, np.arange(ks, ke), px_scale)
    pelvis = torch.nn.Parameter(T(p0))
    cam_rot = torch.nn.Parameter(T(Rotation.from_matrix(rcam).as_rotvec()))
    cam_t = torch.nn.Parameter(T(tcam))
    root_delta = torch.nn.Parameter(torch.zeros(length, 3, device=device))
    body_delta = torch.nn.Parameter(torch.zeros(length, 21, 3, device=device))
    # Apparatus scale is inferred in the source SMPL-X scale, not claimed metric calibration.
    radius = torch.nn.Parameter(T(float(cfg.get("initial_radius_m", 0.37))))
    top = torch.nn.Parameter(T(float(cfg.get("initial_top_m", 0.64))))
    wrist_offset = float(cfg.get("wrist_surface_offset_m", 0.055))
    stage_contact, stage_ground, prep_ref, prep_weight, stage_info = prepare_stage_prior(
        bvh, jn, stable_kp, cfg, BODY_MAP
    )
    if not hand_options["auto_stage_contacts"]:
        stage_contact[:] = 0
        stage_ground[:] = 0
        prep_weight[:] = 0
    preparation_only = hand_options["preparation_only"]
    if preparation_only:
        stage_contact[ss:] = 0
        stage_ground[ss:] = 0
    stage_info["applied_scope"] = "preparation_only" if preparation_only else "preparation_and_dismount"
    ref += np.array([0, 0, support_height])
    if cfg.get("apparatus_pixels"):
        # Visible hand position and speed override the unpaired BVH support timing.
        ap = np.asarray(cfg["apparatus_pixels"])
        wr = stable_kp[:, [9, 10], :2]
        cap_width = np.linalg.norm(ap[2] - ap[1])
        xmin, xmax = (
            ap[1, 0] - detection["horizontal_margin_ratio"] * cap_width,
            ap[2, 0] + detection["horizontal_margin_ratio"] * cap_width,
        )
        ymin, ymax = (
            ap[0, 1] - detection["top_margin_ratio"] * cap_width,
            max(ap[1:3, 1]) + detection["bottom_margin_ratio"] * cap_width,
        )
        if preparation_only:
            # Restore the original circle contact detector; prep contacts below
            # are inferred independently and retain their wider wrist margin.
            xmin, xmax = ap[1, 0] - 25 * px_scale, ap[2, 0] + 25 * px_scale
            ymin, ymax = ap[0, 1] - 35 * px_scale, max(ap[1:3, 1]) + 20 * px_scale
        distance = np.maximum.reduce(
            [xmin - wr[:, :, 0], wr[:, :, 0] - xmax, ymin - wr[:, :, 1], wr[:, :, 1] - ymax, np.zeros(wr.shape[:2])]
        )
        gate = np.exp(-((distance / (detection["circle_distance_sigma_px"] * px_scale)) ** 2))
        velocity = np.linalg.norm(
            np.gradient(gaussian_filter1d(wr, detection["video_smoothing_seconds"] * fps, axis=0), axis=0) * fps,
            axis=-1,
        )
        observed = gate * np.exp(-((velocity / (detection["circle_speed_sigma_px_s"] * px_scale)) ** 2))
        contact = (detection["bvh_contact_fraction"] * contact + detection["video_contact_fraction"] * observed) * gate
        contact[ss:se] /= np.maximum(contact[ss:se].max(1, keepdims=True), detection["minimum_circle_contact"])
    contact[:ss] = 0
    contact[se:] = 0
    contact = np.maximum(contact, stage_contact)
    for start, end, side in cfg.get("extra_hand_contacts", []):
        contact[start:end, side] = 1
    contact = T(contact)
    ref = T(ref)
    phase_t = T(phase)
    palm_contact = contact.clone()
    if preparation_only:
        palm_contact[ss:] = 0
    palm_support = T(
        support_weights(
            N(palm_contact),
            fps,
            [ks, ke],
            hand_options["contact_min_confidence"],
            hand_options["support_smoothing_seconds"],
        )
    )
    if not hand_options["enabled"]:
        palm_support.zero_()
    orientation_weight = hand_weights["orientation"]
    orientation_tolerance = hand_options["orientation_tolerance_deg"]
    if orientation_weight < 0:
        raise ValueError("palm_orientation_weight must be nonnegative")
    wrist_height_slack = hand_options["wrist_height_slack_m"]
    if not 0 <= wrist_height_slack <= 0.03:
        raise ValueError("palm_wrist_height_slack_m must be in [0, .03]")
    conf = np.clip(kp[:, :, 2], 0, 1) ** 2
    conf[:ks] = 0
    conf[ke:] = 0
    conf[:, :5] *= 0.25
    for start, end, ids in cfg.get("occluded_keypoints", []):
        conf[start:end, ids] *= constraints["shared"]["occluded_keypoint_multiplier"]
    conf = T(conf)
    kp_t = T(kp[:, :, :2])
    circle = torch.zeros(length, device=device)
    circle[ss:se] = 1
    ground_mask = T(stage_ground)
    for start, end, side in cfg.get("ground_contacts", []):
        ground_mask[start:end, side] = 1
    foot_ids = [10, 11]
    sk = model.get_skeleton(beta)
    rest_pelvis = sk[:, 0]
    # Keep the BVH prior on limb directions weak, with source limb lengths unchanged.
    edges = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21), (0, 12)]
    refdir = torch.stack([torch.nn.functional.normalize(ref[:, b] - ref[:, a], dim=-1) for a, b in edges], 1)
    prep_edges = edges[4:]
    prep_ref = T(prep_ref)
    prep_weight = T(prep_weight)
    prep_refdir = torch.stack(
        [torch.nn.functional.normalize(prep_ref[:, b] - prep_ref[:, a], dim=-1) for a, b in prep_edges], 1
    )
    # Sparse skinning for reliable COCO joints and surface collision proxies.
    with torch.no_grad():
        verts0 = source_v - j0[:, :1]
        coco0 = source_c - j0[:, :1]
        jrel0 = j0 - j0[:, :1]
    history = []
    stage1 = min(200, max(60, iterations // 4))
    optimizer = torch.optim.Adam(
        [
            {"params": [pelvis], "lr": 0.008},
            {"params": [cam_rot, cam_t], "lr": 0.002},
            {"params": [radius, top], "lr": 0.001},
            {"params": [root_delta, body_delta], "lr": 0.0025},
        ]
    )
    pixel_annotations = cfg.get("apparatus_pixels")
    pose_mask = torch.ones(21, 3, device=device)
    pose_mask[[9, 10, 11, 14]] = 0.4  # feet and head retain stronger preservation
    pose_sigma = torch.full((length, 21, 1), 0.20, device=device)
    # Permit wrist corrections for support, preserve the source more in flight.
    pose_sigma[:, [19, 20], 0] = 0.20 + hand_options["support_wrist_pose_relaxation_rad"] * palm_support
    hand_vertex_weights = model.lbs_weights[132:, [20, 21] + list(range(25, 55))].sum(-1)
    collision_vertex_mask = hand_vertex_weights < 0.35
    palm_rest_normals, palm_ids = palm_geometry(model)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    translation_stage = None
    for step in range(iterations):
        full = step >= stage1
        decay = 1.0 if step < iterations * 0.65 else max(0.15, 1 - (step - iterations * 0.65) / (iterations * 0.4))
        for group, lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = lr * decay
        optimizer.zero_grad(set_to_none=True)
        Rd = axis_angle_to_matrix(root_delta if full else root_delta.detach())
        Rz = Rd @ (B @ root0)
        body = axis_angle_to_matrix(body_delta * pose_mask if full else body_delta.detach()) @ body0
        Ry = B.T @ Rz
        translation_y = pelvis @ B - rest_pelvis
        if full:
            # r6d avoids an unnecessary matrix->axis-angle round trip in the loss.
            v, c = (
                super(SmplxLiteV437Coco17, model).forward(
                    matrix_to_rotation_6d(body).flatten(1),
                    beta,
                    matrix_to_rotation_6d(Ry),
                    translation_y,
                    rotation_type="r6d",
                ),
                None,
            )
            c = torch.einsum("vj,tvc->tjc", model.smplx2coco17_interestd, v[:, :132]) @ B.T
            v = v[:, 132:] @ B.T
            j, wrist_rotations = fk(model, body, Ry, translation_y, beta, return_rotations=True)
            j = j @ B.T
        else:
            j = jrel0 + pelvis[:, None]
            c = coco0 + pelvis[:, None]
            v = verts0 + pelvis[:, None]
        camera = axis_angle_to_matrix(cam_rot)
        cameras = relative_camera @ camera
        camera_trans = torch.einsum("tij,j->ti", relative_camera, cam_t)
        uv = project(c, cameras, camera_trans, K)
        # Robust pixel residual, expressed in a reproducible pixel scale.
        r2 = ((uv - kp_t) / image_sigma).square().sum(-1)
        image_loss = ((torch.sqrt(1 + r2) - 1) * conf).sum() / conf.sum()
        wrist = j[:, [20, 21]]
        wr_r = wrist[:, :, :2].norm(dim=-1)
        dome = 0.21 * radius
        surface = top - dome * (wr_r / radius).square().clamp(max=1.5)
        wrist_gap = wrist[:, :, 2] - surface - wrist_offset
        # The fixed wrist offset is only a coarse proxy. In the pose stage let
        # anatomical palm geometry place the hand within a small height band.
        height_residual = torch.relu(wrist_gap.abs() - wrist_height_slack * palm_support) if full else wrist_gap
        height_loss = ((height_residual / 0.025).square() * contact).sum() / contact.sum().clamp_min(1)
        radial_loss = ((torch.relu(wr_r - radius * 0.88) / 0.025).square() * contact).sum() / contact.sum().clamp_min(1)
        vel = (wrist[1:] - wrist[:-1]) * fps
        cm = contact[1:] * contact[:-1]
        slip_loss = ((vel / 0.25).square().sum(-1) * cm).sum() / cm.sum().clamp_min(1)
        cycle_centers = torch.stack([wrist[s:e].mean((0, 1)) for s, e in zip(cuts[:-1], cuts[1:])])
        # Allow natural hand exchange and cycle variation; do not force each frame to the axis.
        center_loss = ((cycle_centers[:, :2] - cycle_centers[:, :2].mean(0)) / 0.015).square().mean()
        pelvis_centers = torch.stack([j[s:e, 0].mean(0) for s, e in zip(cuts[:-1], cuts[1:])])
        pelvis_cycle_loss = ((pelvis_centers[:, :2] - pelvis_centers[:, :2].mean(0)) / 0.03).square().mean()
        axis_loss = (cycle_centers[:, :2].mean(0) / 0.08).square().mean()
        feet = j[:, foot_ids]
        ground_loss = (((feet[:, :, 2] - 0.045) / 0.025).square() * ground_mask).sum() / ground_mask.sum().clamp_min(1)
        # Floor and capped-body proxy are soft; exact mesh penetration is audited separately.
        floor_loss = (torch.relu(-v[ks:ke, :, 2] - 0.003) / 0.015).square().topk(20, dim=1).values.mean()
        vr = v[:, :, :2].norm(dim=-1)
        vtop = top - dome * (vr / radius).square()
        depth = torch.minimum(radius - vr, torch.minimum(vtop - v[:, :, 2], v[:, :, 2] - 0.08))
        penetration = (torch.relu(depth[:, collision_vertex_mask] - 0.005) / 0.015).square()
        collision_loss = penetration[ks:ke].topk(min(20, penetration.shape[1]), dim=1).values.mean()
        hand_penetration = (torch.relu(depth[:, ~collision_vertex_mask] - 0.007) / 0.015).square()
        hand_collision_loss = hand_penetration[ks:ke].topk(20, dim=1).values.mean() if full else v.new_zeros(())
        palm_gap = torch.stack([(vtop[:, ids] - v[:, ids, 2]).topk(6, dim=1).values.mean(1) for ids in palm_ids], 1)
        palm_loss = (
            (((palm_gap + 0.003) / 0.018).square() * palm_support).sum() / palm_support.sum().clamp_min(1)
            if full
            else v.new_zeros(())
        )
        orientation_loss = v.new_zeros(())
        if full:
            palm_normals = torch.einsum("tsij,sj->tsi", B @ wrist_rotations[:, [20, 21]], palm_rest_normals)
            target_normal = cap_inward_normal(wrist, radius, dome)
            orientation_loss = palm_orientation_loss(palm_normals, target_normal, palm_support, orientation_tolerance)
        footvec = j[:, [7, 8]].mean(1) - wrist.mean(1)
        footdir = torch.nn.functional.normalize(footvec[:, :2], dim=-1)
        # Preserve the source phase; no trajectory-template or phase-fitting variables.
        phaseref = torch.stack([torch.cos(phase_t), torch.sin(phase_t)], -1)
        phase_loss = (((footdir - phaseref) / 0.15).square().sum(-1) * circle).sum() / circle.sum()
        direction = torch.stack([torch.nn.functional.normalize(j[:, b] - j[:, a], dim=-1) for a, b in edges], 1)
        bvh_loss = (((direction - refdir) / 0.5).square().sum(-1) * circle[:, None]).sum() / (circle.sum() * len(edges))
        prep_dirs = torch.stack([torch.nn.functional.normalize(j[:, b] - j[:, a], dim=-1) for a, b in prep_edges], 1)
        prep_loss = (((prep_dirs - prep_refdir) / 0.5).square().sum(-1) * prep_weight[:, None]).sum() / (
            prep_weight.sum().clamp_min(1) * len(prep_edges)
        )
        # Preserve performer-specific articulation with small, smooth SO(3) corrections.
        pose_loss = (body_delta / pose_sigma).square().mean()
        root_loss = (root_delta / 0.24).square().mean()
        pos_delta = pelvis - T(p0)
        smooth = ((pos_delta[2:] - 2 * pos_delta[1:-1] + pos_delta[:-2]) / 0.008).square().mean()
        rot_smooth = ((root_delta[2:] - 2 * root_delta[1:-1] + root_delta[:-2]) / 0.025).square().mean()
        pose_smooth = ((body_delta[2:] - 2 * body_delta[1:-1] + body_delta[:-2]) / 0.025).square().mean()
        position_prior = (pos_delta / 0.35).square().mean()
        apparatus_prior = ((radius - float(cfg.get("initial_radius_m", 0.37))) / 0.1).square() + (
            (top - float(cfg.get("initial_top_m", 0.64))) / 0.15
        ).square()
        apparatus_loss = pelvis.new_zeros(())
        if pixel_annotations:
            # Fixed apparatus centre-top, left/right cap silhouette, and base-front.
            viewx = camera[0, :2]
            viewx = viewx / viewx.norm()
            front = -camera[2, :2]
            front = front / front.norm()
            a = torch.cat([pelvis.new_zeros(2), top[None]])
            left = torch.cat([-viewx * radius, (top - dome)[None]])
            right = torch.cat([viewx * radius, (top - dome)[None]])
            base = torch.cat([front * radius * 0.94, pelvis.new_zeros(1)])
            ap = project(torch.stack([a, left, right, base]), camera, cam_t, K)
            apparatus_loss = ((ap - T(pixel_annotations)) / (10 * px_scale)).square().mean()
        terms = dict(
            image=image_loss,
            height=height_loss,
            radial=radial_loss,
            slip=slip_loss,
            cycle_center=center_loss,
            axis_center=axis_loss,
            pelvis_cycle_center=pelvis_cycle_loss,
            ground_contact=ground_loss,
            ground_collision=floor_loss,
            body_collision=collision_loss,
            collision=hand_collision_loss,
            phase=phase_loss,
            bvh_pose=bvh_loss,
            pose_preservation=pose_loss,
            root_preservation=root_loss,
            translation_smoothness=smooth,
            root_smoothness=rot_smooth,
            pose_smoothness=pose_smooth,
            position_preservation=position_prior,
            apparatus_prior=apparatus_prior,
            apparatus_observation=apparatus_loss,
            preparation_pose=prep_loss,
            palm_surface=palm_loss,
            orientation=orientation_loss,
        )
        loss = weighted_loss(terms, {**base_weights, **hand_weights})
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite refinement loss at iteration {step}")
        if step == stage1 - 1:
            translation_stage = dict(
                joints=N(j), coco_incam=N(transform_camera(c, cameras, camera_trans)), metrics=cycle_metrics(N(j), cuts)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([pelvis, cam_rot, cam_t, root_delta, body_delta, radius, top], 10)
        optimizer.step()
        with torch.no_grad():
            radius.clamp_(0.2, 0.65)
            top.clamp_(0.35, 1.0)
        if step % 50 == 0 or step == iterations - 1:
            scalar = lambda x: float(x.detach())
            row = dict(
                iteration=step,
                stage="pose" if full else "translation",
                loss=scalar(loss),
                image=scalar(image_loss),
                contact=scalar(height_loss),
                slip=scalar(slip_loss),
                center=scalar(center_loss),
                phase=scalar(phase_loss),
                collision=scalar(collision_loss),
                radius=scalar(radius),
                top=scalar(top),
                preparation=scalar(prep_loss),
                palm=scalar(palm_loss),
                palm_orientation=scalar(orientation_loss),
                periodic=0.0,
            )
            history.append(row)
            if callback:
                callback(row)
    with torch.no_grad():
        Ry = B.T @ axis_angle_to_matrix(root_delta) @ B @ root0
        body = axis_angle_to_matrix(body_delta * pose_mask) @ body0
        params = dict(
            body_pose=matrix_to_axis_angle(body).flatten(1),
            global_orient=matrix_to_axis_angle(Ry),
            transl=pelvis @ B - rest_pelvis,
            betas=beta,
        )
        v, c = model(**params)
        j, wrist_rotations = fk(model, body, Ry, params["transl"], beta, return_rotations=True)
        j = j @ B.T
        palm_normals = torch.einsum("tsij,sj->tsi", B @ wrist_rotations[:, [20, 21]], palm_rest_normals)
        target_normal = cap_inward_normal(j[:, [20, 21]], radius, 0.21 * radius)
        palm_angles = torch.rad2deg(torch.acos((palm_normals * target_normal).sum(-1).clamp(-1, 1)))
        camera = axis_angle_to_matrix(cam_rot)
        cameras = relative_camera @ camera
        camera_trans = torch.einsum("tij,j->ti", relative_camera, cam_t)
        # Transform the pelvis, not transl alone: SMPL-X rotates around its rest pelvis.
        cam_pelvis = transform_camera(pelvis[:, None], cameras, camera_trans)[:, 0]
        incam = dict(
            body_pose=params["body_pose"],
            global_orient=matrix_to_axis_angle(cameras @ B @ Ry),
            transl=cam_pelvis - rest_pelvis,
            betas=beta,
        )
        vv, cc = model(**incam)
        transformed = transform_camera(v @ B.T, cameras, camera_trans)
        consistency = float((transformed - vv).abs().max())
        arrays = dict(
            original_joints_zup=jn,
            corrected_joints_zup=N(j),
            original_coco_zup=N(source_c),
            corrected_coco_zup=N(c @ B.T),
            original_coco_incam=N(cam_c),
            corrected_coco_incam=N(cc),
            corrected_surface_zup=N(v @ B.T),
            original_surface_zup=N(source_v),
            surface_vertex_ids=vids.numpy(),
            K=N(K),
            keypoints=kp,
            phase=phase,
            cycle_boundaries=np.array(cuts),
            contact=N(contact),
            reference_joints_zup=N(ref),
            camera_R_zup_to_camera=N(cameras),
            camera_t=N(camera_trans),
            base_camera_R_zup_to_camera=N(camera),
            base_camera_t=N(cam_t),
            original_to_initial_shift=shift,
            root_delta=N(root_delta),
            body_delta=N(body_delta * pose_mask),
        )
        arrays.update(
            stage_contact=stage_contact,
            ground_contact=N(ground_mask),
            palm_contact=N(palm_contact),
            preparation_reference_zup=N(prep_ref),
            preparation_weight=N(prep_weight),
            adjusted_phase=phase.copy(),
        )
        arrays.update(
            palm_support_weight=N(palm_support),
            palm_normals_zup=N(palm_normals),
            palm_target_normals_zup=N(target_normal),
            palm_orientation_degrees=N(palm_angles),
            palm_left_vertex_ids=vids[N(palm_ids[0])].numpy(),
            palm_right_vertex_ids=vids[N(palm_ids[1])].numpy(),
        )

        def pixerr(q):
            return reprojection_metrics(q, N(K), kp, cfg, constraints["shared"]["occluded_keypoint_multiplier"])

        metrics = dict(
            original=cycle_metrics(jn, cuts),
            corrected=cycle_metrics(N(j), cuts),
            original_incam_reprojection=pixerr(N(cam_c)),
            corrected_reprojection=pixerr(N(cc)),
            camera_world_max_vertex_difference_m=consistency,
            root_correction_degrees=dict(
                median=float(np.median(np.linalg.norm(N(root_delta)[ks:ke], axis=-1)) * 180 / np.pi),
                max=float(np.linalg.norm(N(root_delta)[ks:ke], axis=-1).max() * 180 / np.pi),
            ),
            body_correction_degrees=dict(
                median=float(np.median(np.linalg.norm(N(body_delta * pose_mask)[ks:ke], axis=-1)) * 180 / np.pi),
                max=float(np.linalg.norm(N(body_delta * pose_mask)[ks:ke], axis=-1).max() * 180 / np.pi),
            ),
            apparatus=dict(
                center_xy_m=[0, 0],
                radius_m=float(radius),
                top_m=float(top),
                dome_m=float(0.21 * radius),
                calibration="Approximate fit in the original SMPL-X scale; no measured physical scale",
            ),
            phase=dict(cuts=cuts, source_turns=float((phase[cuts[-1]] - phase[cuts[0]]) / (2 * np.pi)), bvh=ref_info),
            history=history,
            stage_support=stage_info,
            active_refinement=dict(
                preparation_only=preparation_only,
                periodic_weight=0.0,
                phase_refinement=False,
                image_weight=base_weights["image"],
                hands_enabled=hand_options["enabled"],
                palm_contact_scope="preparation_only" if preparation_only else "all_inferred_contacts",
                palm_orientation_weight=orientation_weight,
                palm_orientation_tolerance_deg=orientation_tolerance,
                palm_wrist_height_slack_m=wrist_height_slack,
                palm_surface="fixed_anatomical_volar_samples",
            ),
            trajectory_original=trajectory_statistics(jn, phase, cuts),
            trajectory_corrected=trajectory_statistics(N(j), phase, cuts),
        )
        prep_sel = stage_contact > 0.8
        wrist_final = N(j)[:, [20, 21]]
        rr = np.linalg.norm(wrist_final[:, :, :2], axis=-1)
        gap = wrist_final[:, :, 2] - (float(top) - float(0.21 * radius) * (rr / float(radius)) ** 2) - wrist_offset
        palm_residual = palm_surface_gaps(
            N(v @ B.T), [N(ids) for ids in palm_ids], float(radius), float(top), float(0.21 * radius)
        )
        arrays["palm_surface_gap_m"] = palm_residual

        def orientation_summary(sel):
            values = N(palm_angles)[sel]
            return dict(
                samples=int(len(values)),
                median_deg=float(np.median(values)) if len(values) else None,
                p95_deg=float(np.percentile(values, 95)) if len(values) else None,
                over_90_samples=int((values > 90).sum()),
            )

        orientation_sel = (N(contact) > 0.8) & (np.arange(length)[:, None] >= ks) & (np.arange(length)[:, None] < ke)
        metrics["palm_orientation"] = dict(
            all_support=orientation_summary(orientation_sel),
            circle_support=orientation_summary(
                orientation_sel & (np.arange(length)[:, None] >= ss) & (np.arange(length)[:, None] < se)
            ),
            zero_support_weight_samples=int((N(palm_support) == 0).sum()),
            note="Anatomical palm normal versus inward fitted-cap normal; inferred support, not video ground truth. Free 30-degree cone by default, no in-plane heading target.",
        )
        metrics["stage_contact_quality"] = dict(
            wrist_surface_offset_mae_cm=float(np.abs(gap[prep_sel]).mean() * 100) if prep_sel.any() else None,
            wrist_surface_offset_p95_cm=(
                float(np.percentile(np.abs(gap[prep_sel]), 95) * 100) if prep_sel.any() else None
            ),
            palm_surface_gap_mae_cm=float(np.abs(palm_residual[prep_sel]).mean() * 100) if prep_sel.any() else None,
            inferred_contact_samples=int(prep_sel.sum()),
            note="Residual to fitted surface plus wrist offset, on inferred preparation/dismount contacts",
        )
        if translation_stage:
            arrays["translation_only_joints_zup"] = translation_stage["joints"]
            metrics["translation_only"] = translation_stage["metrics"]
            metrics["translation_only_reprojection"] = pixerr(translation_stage["coco_incam"])
        metrics["constraints_sha256"] = constraint_hash(constraints)
        metrics["camera_mode"] = "background_rotation" if camera_motion is not None else "fixed"
    result = {
        key: {k: v.detach().cpu() for k, v in value.items()}
        for key, value in [("smpl_params_global", params), ("smpl_params_incam", incam)]
    }
    result["K_fullimg"] = prediction["K_fullimg"].clone()
    # Omit stale net_outputs: they describe the pre-refinement motion, not this result.
    return result, arrays, metrics
