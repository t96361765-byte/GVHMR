"""Observable stage contacts and soft periodic priors for unpaired motion."""

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d
from scipy.spatial.transform import Rotation
from hmr4d.utils.mushroom_config import resolve_constraints


def circle_envelope(length, start, end, fps, fade_seconds=0.4):
    """Shared C2 ramp; hold two circle boundary frames exactly unchanged."""
    if not 0 <= start < end <= length or fade_seconds <= 0 or fps <= 0:
        raise ValueError("Invalid circle range, frame rate or fade duration")
    fade = max(1.0, min(float(fade_seconds) * fps, (end - start - 4) / 2))
    t = np.arange(length)
    x = np.clip(np.minimum(t - (start + 1), (end - 2) - t) / fade, 0.0, 1.0)
    return x * x * x * (10 + x * (-15 + 6 * x))


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [[int(s), int(e)] for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def sustained(mask, minimum, gap=1):
    mask = np.asarray(mask, bool).copy()
    if gap:
        mask = binary_closing(mask, structure=np.ones(gap + 1))
    result = np.zeros_like(mask)
    for start, end in intervals(mask):
        if end - start >= minimum:
            result[start:end] = True
    return result


def infer_bvh_support(bvh, circle_range, options=None):
    """Slow wrists must also occupy the support region observed during circles.

    A stationary hand beside the body is not classified as apparatus contact.
    These are kinematic contact candidates, not force-sensor ground truth.
    """
    options = options or resolve_constraints()["hands"]["detection"]
    names = bvh["names"]
    fps = bvh["fps"]
    wrist = bvh["joints"][:, [names.index("LeftWrist"), names.index("RightWrist")]]
    speed = np.linalg.norm(np.gradient(gaussian_filter1d(wrist, max(0.5, 0.02 * fps), axis=0), axis=0) * fps, axis=-1)
    start, end = circle_range
    low = (speed[start:end] < options["bvh_hand_speed_m_s"]) & (
        wrist[start:end, :, 2] <= np.percentile(wrist[start:end, :, 2], 55)
    )
    samples = wrist[start:end][low]
    if len(samples) < 10:
        raise ValueError("BVH has too few slow, low wrists to infer a support surface")
    center = np.median(samples, axis=0)
    horizontal = np.linalg.norm(wrist[:, :, :2] - center[:2], axis=-1)
    close = (np.abs(wrist[:, :, 2] - center[2]) < options["bvh_height_tolerance_m"]) & (
        horizontal < options["bvh_horizontal_radius_m"]
    )
    candidate = close & (speed < options["bvh_hand_speed_m_s"])
    mask = np.stack(
        [sustained(candidate[:, side], max(3, round(0.10 * fps)), round(0.03 * fps)) for side in range(2)], -1
    )
    double = sustained(mask.all(1) & (np.arange(len(mask)) < start), max(5, round(0.25 * fps)))
    spans = intervals(double)
    # Longest stable double support before the configured circle interval.
    prep = max(spans, key=lambda x: x[1] - x[0]) if spans else None
    return mask, dict(
        support_center_m=center.tolist(),
        prep_double_support=prep,
        left_support_ranges=intervals(mask[:, 0]),
        right_support_ranges=intervals(mask[:, 1]),
        fps=float(fps),
        method="Wrist speed, support height and horizontal proximity; kinematic inference",
    )


def infer_video_support(stable_kp, cfg):
    """Detect sustained, confident wrist/ankle stationarity near the apparatus."""
    options = resolve_constraints(cfg)["hands"]["detection"]
    count = len(stable_kp)
    fps = cfg.get("fps", 30)
    start, end = cfg["keep_range"]
    ss, se = cfg["circle_range"]
    contact = np.zeros((count, 2))
    ground = np.zeros_like(contact)
    if not cfg.get("apparatus_pixels"):
        return contact, ground, {"prep_double_support": None, "reason": "No apparatus landmarks"}
    ap = np.asarray(cfg["apparatus_pixels"])
    width = np.linalg.norm(ap[2] - ap[1])
    wrist = stable_kp[:, [9, 10], :2]
    # Wrists lie above the visible palm; margins scale with apparatus width.
    xmin, xmax = (
        ap[1, 0] - options["horizontal_margin_ratio"] * width,
        ap[2, 0] + options["horizontal_margin_ratio"] * width,
    )
    ymin, ymax = (
        ap[0, 1] - options["top_margin_ratio"] * width,
        max(ap[1:3, 1]) + options["bottom_margin_ratio"] * width,
    )
    close = (wrist[:, :, 0] > xmin) & (wrist[:, :, 0] < xmax) & (wrist[:, :, 1] > ymin) & (wrist[:, :, 1] < ymax)
    speed = np.linalg.norm(
        np.gradient(gaussian_filter1d(wrist, options["video_smoothing_seconds"] * fps, axis=0), axis=0) * fps, axis=-1
    )
    outside = (
        ((np.arange(count) < ss) | (np.arange(count) >= se)) & (np.arange(count) >= start) & (np.arange(count) < end)
    )
    for side in range(2):
        candidate = (
            close[:, side]
            & (speed[:, side] < options["video_hand_speed_widths_s"] * width)
            & (stable_kp[:, 9 + side, 2] > options["hand_confidence"])
            & outside
        )
        contact[:, side] = sustained(candidate, max(3, round(options["minimum_support_seconds"] * fps)), 1)
    spans = intervals((contact > 0.5).all(1) & (np.arange(count) < ss))
    prep = max(spans, key=lambda x: x[1] - x[0]) if spans else None
    # Only confident, low and stationary feet outside circles are attracted to ground.
    feet = stable_kp[:, [15, 16], :2]
    foot_speed = np.linalg.norm(
        np.gradient(gaussian_filter1d(feet, options["video_smoothing_seconds"] * fps, axis=0), axis=0) * fps, axis=-1
    )
    for side in range(2):
        candidate = (feet[:, side, 1] > max(ap[1:3, 1]) + 0.04 * width) & (
            foot_speed[:, side] < options["video_foot_speed_widths_s"] * width
        )
        candidate &= (stable_kp[:, 15 + side, 2] > options["foot_confidence"]) & outside
        ground[:, side] = sustained(candidate, max(3, round(options["minimum_support_seconds"] * fps)), 1)
    info = dict(
        prep_double_support=prep,
        left_support_ranges=intervals(contact[:, 0] > 0.5),
        right_support_ranges=intervals(contact[:, 1] > 0.5),
        left_ground_ranges=intervals(ground[:, 0] > 0.5),
        right_ground_ranges=intervals(ground[:, 1] > 0.5),
        method="Camera-stabilized 2D stationarity and apparatus proximity; not force measurement",
    )
    # Ramp constraints at touch/release while retaining unambiguous plateau contacts.
    contact = gaussian_filter1d(contact, options["event_smoothing_seconds"] * fps, axis=0)
    ground = gaussian_filter1d(ground, options["event_smoothing_seconds"] * fps, axis=0)
    contact[:start] = contact[end:] = 0
    ground[:start] = ground[end:] = 0
    return contact, ground, info


def prepare_stage_prior(bvh, source_joints, stable_kp, cfg, body_map):
    """Align support onset/release events, not the whole unpaired clip duration."""
    bmask, binfo = infer_bvh_support(bvh, cfg["bvh_circle_range"], resolve_constraints(cfg)["hands"]["detection"])
    contact, ground, vinfo = infer_video_support(stable_kp, cfg)
    reference = np.zeros_like(source_joints)
    weight = np.zeros(len(source_joints))
    bs = binfo["prep_double_support"]
    vs = vinfo["prep_double_support"]
    info = dict(bvh=binfo, video=vinfo, preparation_prior_enabled=False)
    if bs is None or vs is None:
        return contact, ground, reference, weight, info
    ss = cfg["circle_range"][0]
    bss = cfg["bvh_circle_range"][0]
    stop = min(ss, len(source_joints))
    ids = np.arange(vs[0], stop)
    query = np.interp(ids, [vs[0], vs[1] - 1, ss], [bs[0], bs[1] - 1, bss])
    mapped = np.zeros((len(bvh["joints"]), 22, 3))
    for joint, name in body_map.items():
        mapped[:, joint] = bvh["joints"][:, bvh["names"].index(name)]
    # Estimate one yaw from the performer's shoulders during stable double support.
    src_axis = np.median(source_joints[vs[0] : vs[1], 16] - source_joints[vs[0] : vs[1], 17], axis=0)
    ref_axis = np.median(mapped[bs[0] : bs[1], 16] - mapped[bs[0] : bs[1], 17], axis=0)
    yaw = np.arctan2(src_axis[1], src_axis[0]) - np.arctan2(ref_axis[1], ref_axis[0])
    aligned = mapped @ Rotation.from_euler("z", yaw).as_matrix().T
    reference[ids] = np.stack(
        [np.interp(query, np.arange(len(mapped)), col) for col in aligned.reshape(len(mapped), -1).T], -1
    ).reshape(-1, 22, 3)
    weight[ids] = 0.35 + 0.65 * contact[ids].min(1)
    weight[ids[:3]] *= np.linspace(0.2, 1, min(3, len(ids)))
    weight[max(vs[0], ss - 4) : ss] *= np.linspace(1, 0.1, min(4, ss - vs[0]))
    info.update(
        preparation_prior_enabled=True,
        alignment_yaw_deg=float(np.rad2deg(yaw)),
        event_alignment=dict(video=[vs[0], vs[1] - 1, ss], bvh=[bs[0], bs[1] - 1, bss]),
    )
    return contact, ground, reference, weight, info


def fourier_basis(phase, harmonics=3):
    return np.stack(
        [np.ones_like(phase)] + [f(k * phase) for k in range(1, harmonics + 1) for f in (np.cos, np.sin)], -1
    )


def trajectory_statistics(joints, phase, cuts):
    """Phase-normalized cycle variation; no requirement that the path be circular."""
    grid = np.linspace(0, 1, 61)
    direction = np.sign(phase[cuts[-1]] - phase[cuts[0]]) or 1
    paths = []
    for s, e in zip(cuts[:-1], cuts[1:]):
        # Cuts are integer frames just AFTER an angular crossing. Interpolate the
        # exact same absolute phase instead of rotating each cycle by that offset.
        lo = max(0, s - 1)
        hi = min(len(joints), e + 2)
        p = joints[lo:hi, [7, 8]].mean(1)
        level = np.floor((direction * phase[s] + 1e-6) / (2 * np.pi)) * 2 * np.pi
        progress = (direction * phase[lo:hi] - level) / (2 * np.pi)
        progress = np.maximum.accumulate(progress)
        paths.append(np.stack([np.interp(grid, progress, col) for col in p.T], -1))
    paths = np.array(paths)
    mean = paths.mean(0)
    basis = fourier_basis(grid * 2 * np.pi)
    fitted = basis @ np.linalg.lstsq(basis, mean, rcond=None)[0]
    return dict(
        cycle_variation_rms_cm=float(np.sqrt(np.mean(np.sum((paths - mean) ** 2, axis=-1))) * 100),
        mean_path_high_frequency_rms_cm=float(np.sqrt(np.mean(np.sum((mean - fitted) ** 2, axis=-1))) * 100),
        note="Ankle midpoint in 3D, phase normalized. Lower is more repeatable/smooth, not proof of true accuracy.",
    )
