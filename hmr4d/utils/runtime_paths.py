"""Resolve optional external programs without depending on the launch directory."""
import os
import shutil
import sys
from pathlib import Path

from hmr4d import PROJ_ROOT


def ffmpeg_program(name):
    explicit = os.environ.get(f"GVHMR_{name.upper()}")
    candidates = [Path(explicit)] if explicit else []
    suffix = ".exe" if os.name == "nt" else ""
    candidates += [PROJ_ROOT / "tools" / "ffmpeg" / "bin" / (name + suffix),
                   Path(sys.prefix) / "Library" / "bin" / (name + suffix)]
    candidates += list(PROJ_ROOT.glob(f"ffmpeg*/bin/{name}{suffix}"))
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found. Install conda-forge ffmpeg or set GVHMR_{name.upper()}.")


def find_blender(explicit=None):
    configured = explicit or os.environ.get("GVHMR_BLENDER")
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    # Prefer the user's installed D: release over any bundled alpha version.
    candidates = sorted(Path("D:/Blender Foundation").glob("Blender */blender.exe"), reverse=True)
    found = shutil.which("blender")
    if found:
        candidates.append(Path(found))
    if not candidates:
        raise FileNotFoundError("Set --blender to blender.exe. NPZ export does not need Blender.")
    return candidates[0].resolve()


def find_smplx_addon(explicit=None):
    configured = explicit or os.environ.get("GVHMR_SMPLX_ADDON")
    candidates = [Path(configured)] if configured else sorted(
        Path("D:/Blender Foundation").glob("smplx_blender_addon*/smplx_blender_addon"), reverse=True
    )
    for path in candidates:
        if (path / "__init__.py").is_file() and (path / "data").is_dir():
            return path.resolve()
    raise FileNotFoundError("Set --smplx-addon to the SMPL-X add-on package directory containing __init__.py and data/.")

