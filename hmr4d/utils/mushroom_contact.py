"""Soft, contact-gated palm geometry; no constraint on airborne palm direction."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def support_weights(contact, fps, keep_range, minimum=0.2, smoothing_seconds=0.05):
    """Suppress weak contacts and ease touch/release without leaking into flight."""
    if not 0 <= minimum < 0.85:
        raise ValueError("palm_contact_min_confidence must be in [0, .85)")
    x = np.clip((np.asarray(contact) - minimum) / (0.85 - minimum), 0.0, 1.0)
    weight = x * x * (3 - 2 * x)
    start, end = keep_range
    weight[:start] = weight[end:] = 0
    # Multiplying by the unsmoothed gate keeps unsupported frames exactly zero.
    weight *= gaussian_filter1d(weight, max(0.5, smoothing_seconds * fps), axis=0, mode="constant")
    return weight


def palm_geometry(model):
    """Anatomical volar normal and fixed volar samples in the neutral template.

    SMPL-X MCP joint centres define the palm plane. Opposite cross-product signs
    account for the mirrored hands. No frame-dependent lowest-side selection.
    """
    normals = []
    samples = []
    verts = model.v_template[132:]
    for wrist, index, little, sign in [(20, 25, 31, -1.0), (21, 40, 46, 1.0)]:
        rest = model.J_template
        normal = sign * torch.nn.functional.normalize(
            torch.linalg.cross(rest[index] - rest[wrist], rest[little] - rest[wrist]), dim=-1
        )
        offset = verts - rest[wrist]
        distance = offset.norm(dim=-1)
        ids = torch.where(
            (model.lbs_weights[132:, wrist] > 0.55)
            & (distance > 0.025)
            & (distance < 0.095)
            & ((offset * normal).sum(-1) > 0.003)
        )[0]
        if len(ids) < 6:
            raise ValueError("Insufficient anatomical volar palm surface samples")
        normals.append(normal)
        samples.append(ids)
    return torch.stack(normals), samples


def cap_inward_normal(points, radius, dome):
    outward = torch.cat([2 * dome / radius.square() * points[..., :2], torch.ones_like(points[..., :1])], -1)
    return -torch.nn.functional.normalize(outward, dim=-1)


def palm_surface_gaps(vertices, palm_indices, radius, top, dome):
    """Report the six lowest anatomical palm samples against the fitted cap."""
    gaps = []
    for ids in palm_indices:
        points = vertices[:, ids]
        height = top - dome * (np.linalg.norm(points[:, :, :2], axis=-1) / radius) ** 2
        gaps.append(np.sort(points[:, :, 2] - height, axis=1)[:, :6].mean(1))
    return np.stack(gaps, 1)


def palm_orientation_loss(normals, target, weight, tolerance_degrees=30.0):
    """Soft normal cone, free spin about the normal, zero gradient in flight."""
    if not 0 < tolerance_degrees < 90:
        raise ValueError("palm_orientation_tolerance_deg must be in (0, 90)")
    cosine = (normals * target).sum(-1).clamp(-1.0, 1.0)
    excess = torch.relu(np.cos(np.deg2rad(tolerance_degrees)) - cosine)
    return ((excess / 0.5).square() * weight).sum() / weight.sum().clamp_min(1)
