"""Neutral SMPL-X animation interchange; no Blender or PyTorch3D import needed."""
import subprocess
from pathlib import Path

import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.utils.runtime_paths import find_blender, find_smplx_addon


def as_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Animation contains NaN/Inf")
    return result


def save_smplx_animation(prediction, output, model_path=None, fps=30):
    """Save both coordinate systems, plus a 165D SMPL-X full-pose sequence.

    poses uses absolute axis angles (including the model's mean hand pose).
    betas is the temporal mean for fixed-shape consumers; betas_per_frame is
    lossless. Face/expression are unobserved zeros; fingers use the model mean.
    World coordinates are the original GVHMR Y-up, metre coordinate system.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    params = {k: as_array(prediction["smpl_params_global"][k])
              for k in ("body_pose", "global_orient", "transl", "betas")}
    count = len(params["body_pose"])
    if count < 1:
        raise ValueError("Empty animation")
    for key, width in (("body_pose", 63), ("global_orient", 3), ("transl", 3), ("betas", 10)):
        if params[key].shape != (count, width):
            raise ValueError(f"Unexpected {key} shape: {params[key].shape}")
    model_path = Path(model_path or PROJ_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
    with np.load(model_path, allow_pickle=False) as model:
        left_mean = np.asarray(model["hands_meanl"], dtype=np.float32).reshape(45)
        right_mean = np.asarray(model["hands_meanr"], dtype=np.float32).reshape(45)
    poses = np.zeros((count, 165), dtype=np.float32)
    poses[:, :3] = params["global_orient"]
    poses[:, 3:66] = params["body_pose"]
    poses[:, 75:120] = left_mean
    poses[:, 120:165] = right_mean
    arrays = dict(
        poses=poses, trans=params["transl"], body_pose=params["body_pose"],
        global_orient=params["global_orient"], transl=params["transl"],
        betas=params["betas"].mean(axis=0), betas_per_frame=params["betas"],
        gender=np.array("neutral"), model_type=np.array("smplx"),
        mocap_frame_rate=np.array(float(fps)), mocap_framerate=np.array(float(fps)),
        coordinate_system=np.array("GVHMR world; Y-up; metres"),
        hand_pose_reference=np.array("absolute axis-angle; SMPL-X mean hands"),
        face_hand_source=np.array("not estimated; zero face/expression and mean hands"),
        shape_policy=np.array("betas=temporal mean; exact sequence in betas_per_frame"),
        format_version=np.array(1, dtype=np.int32),
    )
    # Never substitute global parameters for camera parameters.
    for key in params:
        incam = as_array(prediction["smpl_params_incam"][key])
        if incam.shape != params[key].shape:
            raise ValueError(f"Mismatched camera {key} shape")
        arrays[f"incam_{key}"] = incam
    if "K_fullimg" in prediction:
        arrays["K_fullimg"] = as_array(prediction["K_fullimg"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return output


def export_fbx(animation_path, output, blender=None, addon=None):
    executable = find_blender(blender)
    addon = find_smplx_addon(addon)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(executable), "--background", "--factory-startup", "--python-exit-code", "1",
        "--python", str(PROJ_ROOT / "tools/export_smplx_blender.py"), "--",
        "--input", str(Path(animation_path).resolve()), "--output", str(output),
        "--addon", str(addon),
    ], check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Blender did not produce {output}")
    return output

