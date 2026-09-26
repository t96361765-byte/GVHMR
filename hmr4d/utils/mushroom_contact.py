"""Hand release detection, shared support gates and arm/body surface constraints."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from pytorch3d.ops import knn_points


def hand_release_weights(joints, keypoints, cfg, options):
    """Airborne evidence overrides unpaired BVH contact timing during circles.

    Relative wrist height removes root drift; forearm length and apparatus width
    normalize performer size and image scale. Complementary cues protect lifts
    at zero image velocity (the apex), where stationarity alone implies contact.
    Low-confidence observations cannot confidently veto support. No frame IDs,
    performer-specific thresholds, circle count, or fixed left/right order.
    """
    count = len(joints)
    release = np.zeros((count, 2))
    if not options["release_enabled"]:
        return release
    ss, se = cfg["circle_range"]
    fps = cfg.get("fps", 30)
    sigma = max(0.5, options["video_smoothing_seconds"] * fps)
    wrists = gaussian_filter1d(joints[:, [20, 21]], sigma, axis=0)
    forearm = np.median(np.linalg.norm(joints[ss:se, [20, 21]] - joints[ss:se, [18, 19]], axis=-1), axis=0)
    height = (wrists[:, :, 2] - wrists[:, ::-1, 2]) / np.maximum(forearm, 1e-4)

    def ramp(value, unit):
        lo, hi = [options[f"release_lift_{unit}_{end}"] for end in ["start", "end"]]
        x = np.clip((value - lo) / (hi - lo), 0, 1)
        return x * x * (3 - 2 * x)

    confidence = gaussian_filter1d(keypoints[:, [9, 10], 2].clip(0, 1), sigma, axis=0)
    reliability = np.clip(
        (confidence - options["hand_confidence"]) / max(0.8 - options["hand_confidence"], 0.05), 0, 1
    )
    # Relative 3D height compares two wrists; both observations must be credible.
    release = ramp(height, "forearm") * reliability.min(1, keepdims=True)
    if cfg.get("apparatus_pixels"):
        ap = np.asarray(cfg["apparatus_pixels"])
        width = max(np.linalg.norm(ap[2] - ap[1]), 1.0)
        uv = gaussian_filter1d(keypoints[:, [9, 10], :2], sigma, axis=0)
        # Compare against the support region, not the other image wrist: two
        # supported hands at different depths can have unequal image heights.
        # Image-up assumes the upright videos accepted by this pipeline.
        upper = ap[0, 1] - options["top_margin_ratio"] * width
        release = np.maximum(release, ramp((upper - uv[:, :, 1]) / width, "width") * reliability)
    release[:ss] = release[se:] = 0
    return release


def arm_body_regions(model):
    """Disjoint forearm/hand versus torso/head/leg surfaces; exclude arm seams.

    Deterministic sparse triangles/points limit optimization cost. These are
    anatomical regions of the actual SMPL-X mesh, so shape follows source betas.
    """
    weights = model.lbs_weights.detach().cpu().numpy()
    faces = np.asarray(model.faces, dtype=np.int64)
    body = weights[:, list(range(13)) + [15, 22, 23, 24]].sum(-1) > 0.7
    arm = weights[:, [18, 19, 20, 21] + list(range(25, 55))].sum(-1) > 0.55
    return faces[body[faces].all(1)][::3].copy(), np.flatnonzero(arm)[::4].copy()


class ArmBodyCollision:
    """Local signed distance to nearby body triangles, differentiable in pose.

    This sampled surface penalty is not a global watertight SDF or a collision-
    free guarantee. It excludes adjacent arm surfaces and allows soft contact.
    """

    def __init__(self, faces, arm_ids, vertex_ids, device, options):
        size = int(max(np.max(vertex_ids), np.max(faces), np.max(arm_ids))) + 1
        remap = np.full(size, -1, dtype=int)
        remap[np.asarray(vertex_ids)] = np.arange(len(vertex_ids))
        if (remap[faces] < 0).any() or (remap[arm_ids] < 0).any():
            raise ValueError("Arm/body surface vertices missing; rerun the base stage with the current constraints")
        self.faces = torch.as_tensor(remap[faces], device=device)
        self.arms = torch.as_tensor(remap[arm_ids], device=device)
        self.tolerance = options["self_collision_tolerance_m"]
        self.sigma = options["self_collision_sigma_m"]

    def __call__(self, vertices):
        points = vertices[:, self.arms]
        triangles = vertices[:, self.faces]
        with torch.no_grad():
            ids = knn_points(points.detach(), triangles.detach().mean(2), K=min(8, len(self.faces))).idx
        batch = torch.arange(len(vertices), device=vertices.device)[:, None, None]
        tri = triangles[batch, ids]
        p = points[:, :, None]
        a, b, c = tri.unbind(-2)
        u, v = b - a, c - a
        normal = torch.nn.functional.normalize(torch.linalg.cross(u, v), dim=-1)
        plane = ((p - a) * normal).sum(-1)
        projection = p - plane[..., None] * normal
        rel = projection - a
        uu, vv, uv = (u * u).sum(-1), (v * v).sum(-1), (u * v).sum(-1)
        ru, rv = (rel * u).sum(-1), (rel * v).sum(-1)
        den = (uu * vv - uv.square()).clamp_min(1e-12)
        s, t = (ru * vv - rv * uv) / den, (rv * uu - ru * uv) / den
        inside = (s >= 0) & (t >= 0) & (s + t <= 1)
        candidates = [projection]
        distances = [torch.where(inside, plane.square(), torch.full_like(plane, float("inf")))]
        for start, end in [(a, b), (b, c), (c, a)]:
            edge = end - start
            alpha = (((p - start) * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
            closest = start + alpha[..., None] * edge
            candidates.append(closest)
            distances.append((p - closest).square().sum(-1))
        distance, choice = torch.stack(distances, -1).min(-1)
        closest = torch.stack(candidates, -2).gather(-2, choice[..., None, None].expand(*choice.shape, 1, 3)).squeeze(-2)
        nearest = distance.argmin(-1, keepdim=True)
        signed = ((p - closest) * normal).sum(-1).gather(-1, nearest).squeeze(-1)
        excess = torch.relu(-signed - self.tolerance) / self.sigma
        return excess.square().topk(min(8, excess.shape[1]), dim=1).values.mean()


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
