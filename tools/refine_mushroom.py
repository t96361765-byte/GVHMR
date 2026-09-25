"""Refine a GVHMR result: base fit, circle-only legs, or circle-only feet."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from hmr4d.utils.mushroom_refine import read_bvh, refine
from hmr4d.utils.mushroom_legs import refine_closed_legs
from hmr4d.utils.mushroom_feet import refine_circle_feet
from hmr4d.utils.export_smplx import export_fbx
from hmr4d.utils.mushroom_io import save_refinement
from hmr4d.utils.mushroom_config import apply_constraints, read_constraints, resolve_constraints


def refine_base(args, cfg):
    source = args.source
    output = Path(args.output_root).resolve() / args.input.name
    if output == args.input or args.input in output.parents:
        raise ValueError("Use a separate output root to preserve the original reconstruction")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output} is nonempty; choose another root or explicitly --overwrite")
    manifest_path = args.input / "input_manifest.json"
    if (
        manifest_path.exists()
        and not json.loads(manifest_path.read_text()).get("static_cam", False)
        and not args.camera_motion
    ):
        raise ValueError("This optimizer requires a fixed-camera source video")
    pred = torch.load(source, map_location="cpu", weights_only=True)
    kp = torch.load(args.input / "preprocess/vitpose.pt", map_location="cpu", weights_only=True).numpy()
    if len(kp) != len(pred["smpl_params_global"]["body_pose"]):
        raise ValueError("2D keypoint length mismatch")
    camera_motion = None
    if args.camera_motion:
        with np.load(args.camera_motion) as track:
            camera_motion = track["rotations"]
    result, arrays, metrics = refine(
        pred,
        kp,
        read_bvh(args.bvh),
        cfg,
        args.iterations,
        args.device,
        lambda row: print(json.dumps(row), flush=True),
        camera_motion,
    )
    cfg.setdefault("keep_range", [0, len(kp)])
    provenance = dict(
        source=str(source),
        bvh=str(Path(args.bvh).resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        camera_motion=str(Path(args.camera_motion).resolve()) if args.camera_motion else None,
        bvh_sha256=hashlib.sha256(Path(args.bvh).read_bytes()).hexdigest(),
        iterations=args.iterations,
        device=args.device,
        source_frame_range_of_trimmed=cfg["keep_range"],
        source_video=str(args.input / "0_input_video.mp4"),
        world_coordinates="Y-up metres; apparatus centred at XZ=(0,0); floor Y=0",
        limitations=[
            "Unpaired BVH is a weak phase/contact prior, not ground truth.",
            "Apparatus is an approximate geometric fit, not a calibrated physical scan.",
            "Contacts are inferred kinematically; no force or pressure measurements.",
            "Collision loss uses a capped-body proxy and sampled body/full hand-foot surfaces; not a collision-free guarantee.",
        ],
    )
    save_refinement(output, result, arrays, cfg, metrics, provenance)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2), flush=True)
    if args.export_fbx:
        export_fbx(output / "smplx_neutral_trimmed.npz", output / "smplx_neutral_trimmed.fbx", args.blender)


def refine_local(args, cfg, parser):
    folder = args.input
    if not args.overwrite:
        parser.error("Pass --overwrite to update the corrected animation")
    prov = json.loads((folder / "provenance.json").read_text(encoding="utf-8-sig"))
    if folder == Path(prov["source"]).resolve().parent:
        parser.error("Cannot overwrite original GVHMR data")
    if not cfg["constraints"][args.stage]["enabled"]:
        print(f"{args.stage} stage disabled; result unchanged (earlier corrections are not undone).")
        return
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8-sig"))
    if args.stage == "legs":
        function, flag, provenance_key = refine_closed_legs, "closed_legs", "closed_leg_refinement"
        joints = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    else:
        function, flag, provenance_key = refine_circle_feet, "feet", "foot_refinement"
        joints = ["left_ankle", "right_ankle"]
    if metrics.get("active_refinement", {}).get(flag):
        parser.error(f"{args.stage} stage already applied; rerun unified pipeline for a fresh base")
    prediction = torch.load(folder / "hmr4d_results.pt", map_location="cpu", weights_only=True)
    with np.load(folder / "diagnostics.npz") as data:
        arrays = {key: data[key] for key in data.files}
    bvh_path = Path(args.bvh or prov["bvh"])
    initial_hash = hashlib.sha256((folder / "hmr4d_results.pt").read_bytes()).hexdigest()
    result, arrays, metrics = function(
        prediction,
        arrays,
        metrics,
        read_bvh(bvh_path),
        cfg,
        args.iterations,
        args.device,
        lambda row: print(json.dumps(row), flush=True),
    )
    prov[provenance_key] = dict(
        base_result_sha256=initial_hash,
        iterations=args.iterations,
        bvh=str(bvh_path.resolve()),
        bvh_sha256=hashlib.sha256(bvh_path.read_bytes()).hexdigest(),
        changed_body_joints=joints,
    )
    save_refinement(folder, result, arrays, cfg, metrics, prov)
    print(json.dumps({key: value for key, value in metrics[flag].items() if key != "history"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["base", "legs", "feet"], default="base")
    parser.add_argument("--input", required=True, help="Result folder or hmr4d_results.pt")
    parser.add_argument("--bvh", help="Required for base; local stages reuse provenance by default")
    parser.add_argument("--config", help="Per-video annotations JSON (base stage only)")
    parser.add_argument("--output-root", help="Separate output root (base stage only)")
    parser.add_argument("--iterations", type=int, help="Override iterations for the selected stage")
    parser.add_argument("--constraints", help="Reusable constraint JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera-motion", help="camera_motion.npz (base stage only)")
    parser.add_argument("--export-fbx", action="store_true", help="Export FBX after base stage")
    parser.add_argument("--blender", help="Blender executable for --export-fbx")
    args = parser.parse_args()
    if args.stage == "base":
        if not all([args.bvh, args.config, args.output_root]):
            parser.error("Base stage requires --bvh, --config and --output-root")
    elif any([args.config, args.output_root, args.camera_motion, args.export_fbx, args.blender]):
        parser.error("Local stages use the existing result config; base-only options do not apply")
    args.input = Path(args.input).resolve()
    args.source = args.input if args.input.is_file() else args.input / "hmr4d_results.pt"
    if args.input.is_file():
        args.input = args.input.parent
    path = Path(args.config) if args.stage == "base" else args.input / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    previous = resolve_constraints(cfg)
    cfg = apply_constraints(cfg, read_constraints(args.constraints) if args.constraints else None)
    if args.stage != "base" and any(
        values != cfg["constraints"][group] for group, values in previous.items() if group != args.stage
    ):
        parser.error("Changing other constraint groups requires a fresh unified run")
    if args.iterations is not None:
        cfg["constraints"][args.stage]["iterations"] = args.iterations
        cfg = apply_constraints(cfg)
    args.iterations = cfg["constraints"][args.stage]["iterations"]
    if args.stage == "base":
        refine_base(args, cfg)
    else:
        refine_local(args, cfg, parser)


if __name__ == "__main__":
    main()
