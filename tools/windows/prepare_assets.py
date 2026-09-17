"""Copy local inference assets. Does not download files or modify source assets."""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def copy_verified(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        if target.stat().st_size != source.stat().st_size or digest(target) != digest(source):
            raise FileExistsError(f"Refusing to overwrite a different asset: {target}")
        print(f"Already matches: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".copying")
    shutil.copy2(source, temporary)
    if digest(temporary) != digest(source):
        raise IOError(f"Checksum mismatch while copying {source}")
    temporary.replace(target)
    print(f"Copied: {target}")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", default="E:/GVHMR")
    parser.add_argument("--smplx-dir", required=True, help="User's directory containing SMPLX_NEUTRAL.npz")
    args = parser.parse_args()
    source = Path(args.source_project).resolve()
    smplx_dir = Path(args.smplx_dir).resolve()
    checkpoints = source / "inputs/checkpoints"
    relative = ["gvhmr/gvhmr_siga24_release.ckpt", "hmr2/epoch=10-step=25000.ckpt",
                "vitpose/vitpose-h-multi-coco.pth", "yolo/yolov8x.pt", "body_models/smpl/SMPL_NEUTRAL.pkl"]
    pairs = [(checkpoints / name, ROOT / "inputs/checkpoints" / name) for name in relative]
    pairs.append((smplx_dir / "SMPLX_NEUTRAL.npz",
                  ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"))
    # Check every source before copying large checkpoints.
    for src, dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(src)
    for src, dst in pairs:
        copy_verified(src, dst)
    bins = list(source.glob("ffmpeg*/bin/ffmpeg.exe"))
    if bins:
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            copy_verified(bins[0].parent / name, ROOT / "tools/ffmpeg/bin" / name)


if __name__ == "__main__":
    main()

