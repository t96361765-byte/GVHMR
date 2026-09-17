"""Export an existing hmr4d_results.pt without rerunning inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hmr4d.utils.export_smplx import save_smplx_animation, export_fbx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", help="NPZ destination")
    parser.add_argument("--model", help="SMPLX_NEUTRAL.npz")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--fbx", action="store_true")
    parser.add_argument("--blender")
    parser.add_argument("--smplx-addon")
    args = parser.parse_args()
    source = Path(args.input)
    pred = torch.load(source, map_location="cpu", weights_only=True)
    output = Path(args.output) if args.output else source.with_name("smplx_neutral.npz")
    save_smplx_animation(pred, output, args.model, args.fps)
    print(f"Neutral SMPL-X animation: {output}")
    if args.fbx:
        export_fbx(output, output.with_suffix(".fbx"), args.blender, args.smplx_addon)


if __name__ == "__main__":
    main()

