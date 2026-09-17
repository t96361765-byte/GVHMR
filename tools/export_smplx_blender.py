"""Run inside Blender; use the installed SMPL-X add-on and neutral mesh."""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--addon", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    with np.load(args.input, allow_pickle=False) as data:
        if str(data["gender"]) != "neutral" or data["poses"].shape[1] != 165:
            raise ValueError("Expected neutral SMPL-X animation with 55 joints")
        fps = float(data["mocap_frame_rate"])
        if fps != int(fps) or not 1 <= fps <= 120:
            raise ValueError("The Blender add-on requires an integer frame rate in [1, 120]")
    sys.path.insert(0, str(Path(args.addon).resolve().parent))
    import smplx_blender_addon
    smplx_blender_addon.register()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    result = bpy.ops.object.smplx_add_animation(
        filepath=str(Path(args.input).resolve()), anim_format="SMPL-X",
        rest_position="SMPL-X", hand_reference="FLAT", target_framerate=int(fps),
        keyframe_corrective_pose_weights=True,
    )
    if "FINISHED" not in result:
        raise RuntimeError("SMPL-X animation import failed")
    mesh = bpy.context.object
    if mesh.type != "MESH" or mesh.get("smplx_gender") != "neutral":
        raise RuntimeError("The imported model is not a neutral SMPL-X mesh")
    result = bpy.ops.object.smplx_export_fbx(
        filepath=str(Path(args.output).resolve()),
        export_shape_keys="SHAPE_POSECORRECTIVES", target_format="UNITY", animation_only=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError("SMPL-X FBX export failed")
    print("GVHMR_NEUTRAL_FBX_OK", args.output)


if __name__ == "__main__":
    main()
