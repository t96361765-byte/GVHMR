"""Validated, video-independent constraint parameters and ablation presets.

Per-video frame ranges, camera ROI, apparatus landmarks and reference BVH ranges
stay in the annotation config. A constraint file contains none of those inputs.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

DEFAULT_CONSTRAINTS = {
    "schema_version": 1,
    "shared": {
        "seed": 0,
        "reference_image_width": 1920.0,
        "image_sigma_px": 12.0,
        "scale_pixels_with_image": True,
        "occluded_keypoint_multiplier": 0.12,
    },
    "base": {
        "iterations": 1400,
        "weights": {
            "image": 1.0,
            "ground_contact": 0.8,
            "ground_collision": 0.5,
            "body_collision": 0.8,
            "phase": 0.5,
            "bvh_pose": 0.08,
            "pose_preservation": 0.22,
            "root_preservation": 0.12,
            "translation_smoothness": 0.12,
            "root_smoothness": 0.08,
            "pose_smoothness": 0.08,
            "position_preservation": 0.04,
            "apparatus_prior": 0.03,
            "apparatus_observation": 0.6,
            "pelvis_cycle_center": 0.25,
        },
    },
    "hands": {
        "enabled": True,
        "auto_stage_contacts": True,
        "preparation_only": False,
        "orientation_tolerance_deg": 30.0,
        "contact_min_confidence": 0.2,
        "wrist_height_slack_m": 0.015,
        "support_smoothing_seconds": 0.05,
        "support_wrist_pose_relaxation_rad": 0.60,
        "weights": {
            "height": 1.0,
            "radial": 0.6,
            "slip": 0.07,
            "cycle_center": 2.0,
            "axis_center": 0.1,
            "collision": 0.35,
            "palm_surface": 0.35,
            "orientation": 0.6,
            "preparation_pose": 0.06,
        },
        "detection": {
            "horizontal_margin_ratio": 0.10,
            "top_margin_ratio": 0.30,
            "bottom_margin_ratio": 0.08,
            "circle_distance_sigma_px": 20.0,
            "circle_speed_sigma_px_s": 160.0,
            "bvh_contact_fraction": 0.25,
            "video_contact_fraction": 0.75,
            "minimum_circle_contact": 0.15,
            "bvh_contact_speed_m_s": 0.45,
            "video_smoothing_seconds": 1.0 / 30.0,
            "event_smoothing_seconds": 0.7 / 30.0,
            "video_hand_speed_widths_s": 0.40,
            "video_foot_speed_widths_s": 0.18,
            "hand_confidence": 0.45,
            "foot_confidence": 0.5,
            "minimum_support_seconds": 0.13,
            "bvh_hand_speed_m_s": 0.30,
            "bvh_height_tolerance_m": 0.085,
            "bvh_horizontal_radius_m": 0.32,
        },
    },
    "legs": {
        "enabled": True,
        "iterations": 800,
        "fade_seconds": 0.4,
        "learning_rate": 0.015,
        "max_correction_rad": 1.5,
        "prior_percentile": 95.0,
        "ankle_gap_margin_m": 0.025,
        "knee_gap_margin_m": 0.02,
        "thigh_angle_margin_deg": 5.0,
        "shin_angle_margin_deg": 5.0,
        "knee_flexion_margin_deg": 15.0,
        "knee_asymmetry_margin_deg": 5.0,
        "knee_gap_factor": 0.6,
        "weights": {
            "gap": 2.5,
            "parallel": 1.2,
            "knee_flexion": 0.25,
            "self_collision": 1.2,
            "image": 0.18,
            "sweep_direction": 0.12,
            "ankle_midpoint": 0.05,
            "apparatus_collision": 1.0,
            "ground_collision": 0.5,
            "rotation_smoothness": 0.1,
            "position_smoothness": 0.12,
            "pose_preservation": 0.025,
        },
    },
    "feet": {
        "enabled": True,
        "iterations": 600,
        "fade_seconds": 0.4,
        "learning_rate": 0.018,
        "max_correction_rad": 2.4,
        "orientation_tolerance_deg": 6.0,
        "normal_tolerance_extra_deg": 2.0,
        "normal_factor": 0.5,
        "central_attraction_factor": 0.02,
        "pair_angle_margin_deg": 4.0,
        "toe_gap_margin_m": 0.025,
        "ground_tolerance_m": 0.003,
        "ground_sigma_m": 0.005,
        "weights": {
            "orientation": 1.8,
            "parallel": 1.0,
            "toe_gap": 0.5,
            "self_collision": 0.9,
            "apparatus_collision": 0.8,
            "ground_collision": 4.0,
            "image": 0.06,
            "rotation_smoothness": 0.05,
            "relative_foot_stability": 0.04,
            "pose_preservation": 0.006,
        },
    },
}

PRESETS = {
    "default": {},
    "no-hands": {"hands": {"enabled": False}},
    "no-legs": {"legs": {"enabled": False}},
    "no-feet": {"feet": {"enabled": False}},
    "no-palm-orientation": {"hands": {"weights": {"orientation": 0.0}}},
    "no-local-stages": {"legs": {"enabled": False}, "feet": {"enabled": False}},
}

LEGACY_KEYS = {
    "image_weight": "base.weights.image",
    "auto_stage_contacts": "hands.auto_stage_contacts",
    "preparation_only": "hands.preparation_only",
    "palm_orientation_weight": "hands.weights.orientation",
    "palm_orientation_tolerance_deg": "hands.orientation_tolerance_deg",
    "palm_contact_min_confidence": "hands.contact_min_confidence",
    "palm_wrist_height_slack_m": "hands.wrist_height_slack_m",
    "closed_leg_refinement": "legs.enabled",
    "closed_leg_iterations": "legs.iterations",
    "closed_leg_fade_seconds": "legs.fade_seconds",
    "foot_refinement": "feet.enabled",
    "foot_iterations": "feet.iterations",
    "foot_fade_seconds": "feet.fade_seconds",
    "foot_orientation_tolerance_deg": "feet.orientation_tolerance_deg",
}


def merge_constraints(target, patch, prefix=""):
    """Reject misspelled or per-video keys instead of silently ignoring them."""
    if not isinstance(patch, dict):
        raise ValueError(f'{prefix or "constraints"} must be an object')
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ValueError(f"Unknown constraint parameter: {path}")
        if isinstance(target[key], dict):
            merge_constraints(target[key], value, path)
        else:
            target[key] = value
    return target


def set_parameter(config, path, value):
    parts = path.split(".")
    parent = config
    for part in parts[:-1]:
        if part not in parent or not isinstance(parent[part], dict):
            raise ValueError(f"Unknown constraint parameter: {path}")
        parent = parent[part]
    if parts[-1] not in parent or isinstance(parent[parts[-1]], dict):
        raise ValueError(f"Unknown scalar constraint parameter: {path}")
    parent[parts[-1]] = value


def validate_constraints(config):
    def walk(actual, expected, prefix=""):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f'Invalid constraint keys at {prefix or "root"}')
        for key, default in expected.items():
            value = actual[key]
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(default, dict):
                walk(value, default, path)
            elif isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f"{path} must be boolean")
            elif isinstance(default, int):
                if type(value) is not int or value < 0:
                    raise ValueError(f"{path} must be a nonnegative integer")
            elif (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"{path} must be finite and nonnegative")

    walk(config, DEFAULT_CONSTRAINTS)
    if config["schema_version"] != 1:
        raise ValueError("Unsupported constraint schema_version")
    for group in ["base", "legs", "feet"]:
        if config[group]["iterations"] < 1:
            raise ValueError(f"{group}.iterations must be positive")
    for group in ["legs", "feet"]:
        for key in ["fade_seconds", "learning_rate", "max_correction_rad"]:
            if config[group][key] <= 0:
                raise ValueError(f"{group}.{key} must be positive")
    for key in ["image_sigma_px", "reference_image_width"]:
        if config["shared"][key] <= 0:
            raise ValueError(f"shared.{key} must be positive")
    hands = config["hands"]
    feet = config["feet"]
    det = hands["detection"]
    if not 0 < hands["orientation_tolerance_deg"] < 90:
        raise ValueError("Invalid hand orientation tolerance")
    if not 0 <= hands["contact_min_confidence"] < 0.85:
        raise ValueError("Invalid palm minimum confidence")
    if not 0 <= hands["wrist_height_slack_m"] <= 0.03:
        raise ValueError("Invalid wrist height slack")
    if not 0 < feet["orientation_tolerance_deg"] < 45:
        raise ValueError("Invalid foot orientation tolerance")
    if feet["ground_sigma_m"] <= 0:
        raise ValueError("feet.ground_sigma_m must be positive")
    if not 0 < config["legs"]["prior_percentile"] < 100:
        raise ValueError("Invalid leg prior percentile")
    for key in [
        "circle_distance_sigma_px",
        "circle_speed_sigma_px_s",
        "minimum_circle_contact",
        "bvh_contact_speed_m_s",
        "video_smoothing_seconds",
        "event_smoothing_seconds",
    ]:
        if det[key] <= 0:
            raise ValueError(f"hands.detection.{key} must be positive")
    for key in ["bvh_contact_fraction", "video_contact_fraction", "hand_confidence", "foot_confidence"]:
        if not 0 <= det[key] <= 1:
            raise ValueError(f"hands.detection.{key} must be in [0,1]")
    return config


def resolve_constraints(annotation=None, overrides=None):
    annotation = annotation or {}
    result = deepcopy(DEFAULT_CONSTRAINTS)
    for key, path in LEGACY_KEYS.items():
        if key in annotation:
            set_parameter(result, path, annotation[key])
    if "constraints" in annotation:
        merge_constraints(result, annotation["constraints"])
    if overrides is not None:
        merge_constraints(result, overrides)
    return validate_constraints(result)


def read_constraints(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def apply_constraints(annotation, overrides=None):
    """Return effective run config; synchronize legacy CLI keys for compatibility."""
    result = deepcopy(annotation)
    constraints = resolve_constraints(annotation, overrides)
    result["constraints"] = constraints
    for key, path in LEGACY_KEYS.items():
        value = constraints
        for part in path.split("."):
            value = value[part]
        result[key] = value
    result["periodic_weight"] = 0.0
    return result


def constraint_hash(constraints):
    raw = json.dumps(constraints, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def save_constraint_snapshot(folder, cfg):
    constraints = resolve_constraints(cfg)
    Path(folder, "constraints.json").write_text(
        json.dumps(constraints, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return constraint_hash(constraints)


def pixel_scale(cfg, intrinsics=None):
    shared = resolve_constraints(cfg)["shared"]
    if not shared["scale_pixels_with_image"]:
        return 1.0
    size = cfg.get("image_size")
    width = (
        float(size[0])
        if size
        else (float(intrinsics[0, 2]) * 2 if intrinsics is not None else shared["reference_image_width"])
    )
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Invalid image width")
    return width / shared["reference_image_width"]


def weighted_loss(terms, weights, enabled=True):
    """Ordered named losses; disabled groups contribute exactly zero gradient."""
    if terms.keys() != weights.keys():
        raise ValueError("Loss terms and configured weights differ")
    return sum((weights[key] if enabled else 0.0) * value for key, value in terms.items())
