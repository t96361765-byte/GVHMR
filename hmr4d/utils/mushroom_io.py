"""Shared result persistence; dense surface caches are opt-in."""

import json
from pathlib import Path
import numpy as np
import torch
from hmr4d.utils.export_smplx import save_smplx_animation
from hmr4d.utils.mushroom_config import save_constraint_snapshot

DENSE_DIAGNOSTICS = {"original_surface_zup", "corrected_surface_zup"}


def save_diagnostics(path, arrays, full=False):
    kept = {k: v for k, v in arrays.items() if full or k not in DENSE_DIAGNOSTICS}
    np.savez_compressed(path, **kept)


def save_refinement(folder, result, arrays, cfg, metrics, provenance):
    """Use one output contract for base, leg and foot stages."""
    if not np.isfinite(arrays["corrected_joints_zup"]).all():
        raise ValueError("Invalid corrected result")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    start, end = cfg["keep_range"]
    trimmed = {
        key: (
            {name: value[start:end] for name, value in params.items()}
            if isinstance(params, dict)
            else params[start:end]
        )
        for key, params in result.items()
    }
    for suffix, prediction in [("", result), ("_trimmed", trimmed)]:
        torch.save(prediction, folder / f"hmr4d_results{suffix}.pt")
        save_smplx_animation(prediction, folder / f"smplx_neutral{suffix}.npz", fps=cfg.get("fps", 30))
    save_diagnostics(folder / "diagnostics.npz", arrays, full=cfg.get("full_diagnostics", False))
    digest = save_constraint_snapshot(folder, cfg)
    provenance["constraints_sha256"] = metrics["constraints_sha256"] = digest
    for name, value in [("config", cfg), ("metrics", metrics), ("provenance", provenance)]:
        (folder / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
