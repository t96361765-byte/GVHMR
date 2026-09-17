"""Repackage tested PyTorch3D/cython_bbox binaries and patched chumpy, without downloads.

Run with Python 3.10. The resulting wheels are local migration artifacts, not
official upstream wheels. PyTorch3D requires torch 2.11.0+cu130 on win_amd64.
"""
import argparse
import base64
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


def build_wheel(site, package, output):
    candidates = list(site.glob(f"{package}-*.dist-info"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one {package} installation in {site}")
    metadata = candidates[0]
    wheel_info = (metadata / "WHEEL").read_text(encoding="utf-8")
    tags = [line.split(": ", 1)[1] for line in wheel_info.splitlines() if line.startswith("Tag: ")]
    if len(tags) != 1:
        raise ValueError(f"Unsupported wheel tags: {tags}")
    if package in ("pytorch3d", "cython_bbox") and tags[0] != "cp310-cp310-win_amd64":
        raise ValueError("Expected the tested CPython 3.10 Windows PyTorch3D binary")
    dist = metadata.name.removesuffix(".dist-info")
    target = output / f"{dist}-{tags[0]}.whl"
    entries = []
    files = [p for folder in (site / package, metadata) for p in folder.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
             and p.name not in ("RECORD", "RECORD.jws", "RECORD.p7s")]
    files += list(site.glob(f"{package}.*.pyd"))
    if package in ("pytorch3d", "cython_bbox") and not any(p.suffix == ".pyd" for p in files):
        raise ValueError("The CUDA extension is missing")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        def add(name, data):
            archive.writestr(name, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            entries.append((name, f"sha256={digest}", len(data)))
        for path in files:
            add(path.relative_to(site).as_posix(), path.read_bytes())
        add(f"{metadata.name}/gvhmr_local_origin.json", json.dumps({
            "source": str(site), "package": package, "local_repack": True,
            "torch_requirement": "2.11.0+cu130" if package == "pytorch3d" else None,
            "note": "Preserves the installed files, including local compatibility patches.",
        }, indent=2).encode())
        record = f"{metadata.name}/RECORD"
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerows(entries + [(record, "", "")])
        archive.writestr(record, stream.getvalue())
    print(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="E:/GVHMR/envs", help="Existing tested environment prefix")
    parser.add_argument("--output", default="wheels")
    args = parser.parse_args()
    site = Path(args.source).resolve() / "Lib/site-packages"
    for name in ("pytorch3d", "chumpy", "cython_bbox"):
        build_wheel(site, name, Path(args.output).resolve())


if __name__ == "__main__":
    main()
