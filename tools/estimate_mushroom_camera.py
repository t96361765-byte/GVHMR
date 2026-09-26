"""Select static background and measure camera rotation independently of HMR."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

MIN_FEATURES = 20
MIN_INLIERS = 10
TRACKING_MODES = {
    'strict': dict(min_inliers=15, ransac_px=1.5, rotation_px=2.0),
    'balanced': dict(min_inliers=10, ransac_px=2.5, rotation_px=3.5,
                     min_inlier_ratio=.4, min_spread_px=2., rotation_p90_px=8.),
}
PREVIEW_HEADER = 96


class BackgroundTrackingError(ValueError):
    """A different background selection may resolve this failure."""


def load_person_tracks(folder):
    folder = Path(folder)
    boxes = torch.load(folder / 'preprocess/bbx.pt', map_location='cpu', weights_only=True)['bbx_xyxy'].numpy()
    pose_file = folder / 'preprocess/vitpose.pt'
    poses = torch.load(pose_file, map_location='cpu', weights_only=True).numpy() if pose_file.exists() else None
    if poses is not None and poses.shape != (len(boxes), 17, 3):
        raise ValueError('Expected video-aligned COCO keypoints with shape (frames, 17, 3)')
    return boxes, poses


def background_mask(gray, box, keypoints=None):
    """Exclude current person envelope; use a box when pose confidence is poor."""
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    # Remove contiguous black padding at image edges, not dark interior objects.
    for axis in (0, 1):
        content = np.flatnonzero(np.any(gray > 5, axis=axis))
        if not len(content):
            return np.zeros_like(gray)
        start, end = content[0], content[-1] + 1
        if axis == 0:
            mask[:, :start] = 0
            mask[:, end:] = 0
        else:
            mask[:start] = 0
            mask[end:] = 0
    if box is not None:
        if np.shape(box) != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid person bounding box')
        height, width = gray.shape
        padding = max(2, round(20 * min(height, width) / 1080))
        # A sweeping leg makes its rectangular detector box cover empty wall.
        # This padded convex envelope is a conservative proxy, not segmentation.
        confident = (keypoints is not None and np.isfinite(keypoints).all()
                     and np.all(keypoints[[5, 6, 11, 12], 2] >= .35)
                     and np.count_nonzero(keypoints[5:17, 2] >= .35) >= 8
                     and np.count_nonzero(keypoints[:5, 2] >= .35) >= 3)
        if confident:
            body = keypoints[5:17, :2]
            face = keypoints[:5, :2][keypoints[:5, 2] >= .35]
            head = face.mean(axis=0)
            neck = body[:2].mean(axis=0)
            radius = max(padding, np.linalg.norm(head - neck) * .9, np.linalg.norm(np.ptp(face, axis=0)) * .7)
            angles = np.arange(8) * np.pi / 4
            head_outline = head + radius * np.c_[np.cos(angles), np.sin(angles)]
            hull = cv2.convexHull(np.rint(np.concatenate([body, head_outline])).astype(np.int32))
            person = np.zeros_like(gray)
            cv2.fillConvexPoly(person, hull, 255)
            margin = max(padding, round(np.linalg.norm(body[0] - body[1]) * .2))
            person = cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
            mask[person != 0] = 0
        else:
            x0, y0, x1, y1 = np.rint(box).astype(int)
            x0, x1 = np.clip([x0 - padding, x1 + padding], 0, width)
            y0, y1 = np.clip([y0 - padding, y1 + padding], 0, height)
            mask[y0:y1, x0:x1] = 0
    return mask


def background_features(gray, available, roi):
    """Use exactly the same original-resolution feature test in UI and estimator."""
    mask = np.zeros_like(gray)
    height, width = gray.shape
    if roi is None:
        return np.empty((0, 1, 2), np.float32), mask
    if len(roi) != 4 or not np.isfinite(roi).all():
        raise BackgroundTrackingError('Background ROI must contain four finite pixel coordinates')
    x0, y0, x1, y1 = np.rint(roi).astype(int)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise BackgroundTrackingError(f'Background ROI must lie inside the {width} x {height} image')
    mask[y0:y1, x0:x1] = available[y0:y1, x0:x1]
    spacing = max(3, round(8 * min(height, width) / 1080))
    points = cv2.goodFeaturesToTrack(gray, 500, .01, spacing, mask=mask, blockSize=7)
    if points is None:
        points = np.empty((0, 1, 2), np.float32)
    return points, mask


def background_preview(frame, available, roi, points, message=''):
    """Render a compact preview; red exclusions are independent of ROI size."""
    height, width = frame.shape[:2]
    scale = min(1., 1200 / width, 700 / height)
    size = (round(width * scale), round(height * scale))
    view = cv2.resize(frame, size)
    excluded = cv2.resize(available, size, interpolation=cv2.INTER_NEAREST) == 0
    tint = np.full_like(view, (30, 30, 220))
    view[excluded] = cv2.addWeighted(view, .6, tint, .4, 0)[excluded]
    xy_scale = np.array([size[0] / width, size[1] / height])
    if roi is not None:
        x0, y0, x1, y1 = np.rint(np.array(roi) * np.tile(xy_scale, 2)).astype(int)
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 220, 255), 2)
    for point in points[:, 0]:
        cv2.circle(view, tuple(np.rint(point * xy_scale).astype(int)), 2, (0, 255, 0), -1)
    canvas = np.zeros((size[1] + PREVIEW_HEADER, max(800, size[0]), 3), np.uint8)
    canvas[PREVIEW_HEADER:, :size[0]] = view
    count = len(points)
    color = (0, 220, 0) if count >= 50 else (0, 200, 255) if count >= MIN_FEATURES else (70, 70, 255)
    status = 'Ready' if count >= MIN_FEATURES else 'Too few points: draw another rectangle'
    lines = [
        'Drag ROI | Green: usable points | Red: person / black padding excluded',
        f'{count} points (minimum {MIN_FEATURES}; 50+ preferred) | {status}',
        message or 'ENTER accepts | R clears | ESC cancels | Point count does not guarantee full-video tracking',
    ]
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (10, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, color if i == 1 else (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def select_background(frame, box, initial_roi=None, keypoints=None):
    """Drag/retry until the actual estimator can seed enough background points."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    available = background_mask(gray, box, keypoints)
    height, width = gray.shape
    scale = min(1., 1200 / width, 700 / height)
    display_width, display_height = round(width * scale), round(height * scale)
    state = dict(roi=initial_roi, anchor=None, dirty=True, message='')
    name = 'Static background - drag ROI; ENTER accepts; R clears; ESC cancels'

    def drag(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not (0 <= x < display_width and PREVIEW_HEADER <= y < PREVIEW_HEADER + display_height):
                return
            state['anchor'] = (x, y - PREVIEW_HEADER)
        if state['anchor'] is None:
            return
        if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP):
            ax, ay = state['anchor']
            bx, by = np.clip(x, 0, display_width), np.clip(y - PREVIEW_HEADER, 0, display_height)
            roi = [round(min(ax, bx) * width / display_width), round(min(ay, by) * height / display_height),
                   round(max(ax, bx) * width / display_width), round(max(ay, by) * height / display_height)]
            state.update(roi=roi if roi[2] > roi[0] and roi[3] > roi[1] else None, dirty=True, message='')
        if event == cv2.EVENT_LBUTTONUP:
            state['anchor'] = None

    cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(name, drag)
    points = np.empty((0, 1, 2), np.float32)
    try:
        while True:
            if state['dirty']:
                try:
                    points, _ = background_features(gray, available, state['roi'])
                except BackgroundTrackingError as exc:
                    state.update(roi=None, message=str(exc))
                    points = np.empty((0, 1, 2), np.float32)
                cv2.imshow(name, background_preview(frame, available, state['roi'], points, state['message']))
                state['dirty'] = False
            key = cv2.waitKey(30) & 255
            try:
                visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                visible = False
            if key == 27 or not visible:
                raise BackgroundTrackingError('Background selection cancelled')
            if key in (ord('r'), ord('R')):
                state.update(roi=None, anchor=None, dirty=True, message='')
            if key in (10, 13) and state['anchor'] is None:
                # Mouse events can arrive during waitKey; do not accept a new
                # rectangle using the previous rectangle's cached point count.
                points, _ = background_features(gray, available, state['roi'])
                if len(points) >= MIN_FEATURES:
                    return state['roi']
                state.update(message='Cannot accept: select textured stationary scenery outside the red area.', dirty=True)
                print(f'Only {len(points)} usable background points; need {MIN_FEATURES}. Please draw another ROI.', flush=True)
    finally:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass  # The window may already have been closed with its title-bar X.


def track_points(source, target, points, available):
    """Forward/backward LK check, including current-frame person exclusion."""
    if len(points) < MIN_INLIERS:
        return np.empty((0, 1, 2), np.float32), np.zeros(len(points), bool)
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        source, target, points, None, winSize=(31, 31), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    if tracked is None:
        return np.empty((0, 1, 2), np.float32), np.zeros(len(points), bool)
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(target, source, tracked, None, winSize=(31, 31), maxLevel=4)
    if back is None:
        return np.empty((0, 1, 2), np.float32), np.zeros(len(points), bool)
    valid = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
    valid &= np.isfinite(tracked).all(axis=(1, 2)) & np.isfinite(back).all(axis=(1, 2))
    valid &= np.linalg.norm(back[:, 0] - points[:, 0], axis=-1) < 1
    height, width = target.shape
    xy = np.rint(np.nan_to_num(tracked[:, 0], nan=-1, posinf=-1, neginf=-1)).astype(np.int64)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    valid &= available[np.clip(xy[:, 1], 0, height - 1), np.clip(xy[:, 0], 0, width - 1)] != 0
    return tracked[valid], valid


def fit_rotation(reference_points, tracked, K, kinv, mode='strict', diagnostics=None):
    """Always fit against observed reference coordinates, never chained rotations."""
    limits = TRACKING_MODES[mode]
    detail = diagnostics if diagnostics is not None else {}
    detail.update(tracks=len(tracked), inliers=0, reason='too few tracked points')
    if len(tracked) < limits['min_inliers']:
        return None
    p, q = reference_points[:, 0], tracked[:, 0]
    affine, inliers = cv2.estimateAffinePartial2D(
        p, q, method=cv2.RANSAC, ransacReprojThreshold=limits['ransac_px'], maxIters=2000)
    detail.update(inliers=0 if inliers is None else int(inliers.sum()), reason='too few geometric inliers')
    if affine is None or inliers is None or inliers.sum() < limits['min_inliers']:
        return None
    good = inliers[:, 0].astype(bool)
    p, q = p[good], q[good]
    # A lower count is useful only when the matches agree and cover an area.
    # Reject small accidental clusters and nearly collinear point sets.
    if mode == 'balanced':
        ratio = float(good.mean())
        spread = min(float(np.linalg.eigvalsh(np.cov(x.T)).min()) for x in (p, q))
        detail.update(inlier_ratio=ratio, minimum_spread_px=float(np.sqrt(max(0, spread))))
        if ratio < limits['min_inlier_ratio'] or spread < limits['min_spread_px'] ** 2:
            detail['reason'] = 'inliers too sparse or too narrowly distributed'
            return None
    a = np.c_[p, np.ones(len(p))] @ kinv.T
    b = np.c_[q, np.ones(len(q))] @ kinv.T
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    u, _, vh = np.linalg.svd(b.T @ a)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
    projected = (np.c_[p, np.ones(len(p))] @ kinv.T @ rotation.T) @ K.T
    projected = projected[:, :2] / projected[:, 2:]
    residuals = np.linalg.norm(projected - q, axis=-1)
    error = float(np.median(residuals))
    p90 = float(np.percentile(residuals, 90))
    detail.update(rotation_median_px=error, rotation_p90_px=p90, reason='rotation fit exceeds pixel tolerance')
    if not np.isfinite(error) or error > limits['rotation_px'] or (mode == 'balanced' and p90 > limits['rotation_p90_px']):
        return None
    detail['reason'] = 'accepted'
    return rotation, affine, error, reference_points[good], tracked[good]


def describe_fit(detail):
    text = f"{detail['tracks']} tracks -> {detail['inliers']} inliers; {detail['reason']}"
    if 'rotation_median_px' in detail:
        text += f" (median {detail['rotation_median_px']:.2f} px, p90 {detail['rotation_p90_px']:.2f} px)"
    return text


def estimate(folder, roi, reference_frame, output, mode='balanced'):
    if mode not in TRACKING_MODES:
        raise ValueError(f'Unknown camera tracking mode: {mode}')
    limits = TRACKING_MODES[mode]
    folder, output = Path(folder), Path(output)
    pred = torch.load(folder / 'hmr4d_results.pt', map_location='cpu', weights_only=True)
    K = pred['K_fullimg'][0].numpy()
    kinv = np.linalg.inv(K)
    boxes, poses = load_person_tracks(folder)
    cap = cv2.VideoCapture(str(folder / '0_input_video.mp4'))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames or len(frames) != len(boxes) or not 0 <= reference_frame < len(frames):
        raise ValueError('Video, person boxes and reference frame are inconsistent')
    ref = frames[reference_frame]
    available = background_mask(ref, boxes[reference_frame], None if poses is None else poses[reference_frame])
    points, mask = background_features(ref, available, roi)
    if len(points) < MIN_FEATURES:
        raise BackgroundTrackingError(
            f'Background ROI has {len(points)} usable features (need {MIN_FEATURES}); '
            f'{np.count_nonzero(mask)} pixels remain after this frame\'s person/padding exclusion. Reselect a textured background.')
    count = len(frames)
    rotations = np.repeat(np.eye(3)[None], count, axis=0)
    matrices = np.repeat(np.eye(3)[None, :2], count, axis=0)
    counts = np.full(count, len(points), dtype=int)
    errors = np.zeros(count)
    fallback_frames = []
    fit_details = [None] * count
    # Process outward from the reference, so both earlier and later frames can recover.
    for indices in (range(reference_frame + 1, count), range(reference_frame - 1, -1, -1)):
        previous_frame, anchors, previous_points = reference_frame, points, points
        for i in indices:
            current_mask = background_mask(frames[i], boxes[i], None if poses is None else poses[i])
            tracked, valid = track_points(ref, frames[i], points, current_mask)
            direct_detail = {}
            result = fit_rotation(points[valid], tracked, K, kinv, mode, direct_detail)
            adjacent = None
            adjacent_detail = None
            if previous_frame != reference_frame:
                tracked, valid = track_points(frames[previous_frame], frames[i], previous_points, current_mask)
                adjacent_detail = {}
                adjacent = fit_rotation(anchors[valid], tracked, K, kinv, mode, adjacent_detail)
            # Keep the better-supported correspondence set. A marginal direct
            # match must not discard a healthy continuous track and strand it.
            if adjacent is not None and (result is None or len(adjacent[4]) > len(result[4])):
                result = adjacent
                fit_details[i] = adjacent_detail
                fallback_frames.append(i)
            else:
                fit_details[i] = direct_detail
            if result is None:
                evidence = f'direct: {describe_fit(direct_detail)}'
                if adjacent_detail is not None:
                    evidence += f'; adjacent: {describe_fit(adjacent_detail)}'
                raise BackgroundTrackingError(
                    f'Unreliable background at frame {i} [{mode}]: {evidence}. '
                    f'Need {limits["min_inliers"]} inliers and median rotation fit <= {limits["rotation_px"]} px. '
                    'Choose a clearer static background; camera translation/zoom may also exceed this model.')
            rotations[i], matrices[i], errors[i], anchors, previous_points = result
            counts[i] = len(previous_points)
            previous_frame = i
    smooth = gaussian_filter1d(Rotation.from_matrix(rotations).as_rotvec(), .65, axis=0)
    rotation = Rotation.from_rotvec(smooth).as_matrix()
    # Smoothing must not shift the annotated frame's coordinate system.
    rotation = rotation @ rotation[reference_frame].T
    report = dict(
        method='Reference-anchored LK tracks, similarity RANSAC, unit-ray SO(3) alignment',
        tracking_mode=mode, thresholds=limits,
        reference_frame=reference_frame, roi=list(roi), frames=count, features=len(points),
        mask_policy='Current frame padded pose envelope (box fallback for low confidence) and outer black padding; no all-frame box union',
        adjacent_recovery_frames=sorted(fallback_frames),
        recovery_note='Adjacent tracking retains original reference feature identities. All fit errors are measured against the reference, not chained local rotations. Optical-flow drift is still possible.',
        minimum_inlier_count=int(counts.min()), median_rotation_fit_error_px=float(np.median(errors)),
        p95_rotation_fit_error_px=float(np.percentile(errors, 95)),
        maximum_p90_rotation_fit_error_px=float(max((d['rotation_p90_px'] for d in fit_details if d), default=0)),
        maximum_rotation_from_reference_deg=float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec(), axis=-1).max())),
        maximum_affine_translation_px=float(np.linalg.norm(matrices[:, :, 2], axis=-1).max()),
        limitations='Rotational camera model only. Translational parallax, rolling shutter, zoom and arbitrary moving cameras need a different estimator. Inlier counts do not prove physical stationarity.')
    # No partial camera files are written if any frame fails validation.
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'camera_motion.npz', rotations=rotation, raw_rotations=rotations, affine=matrices,
                        inliers=counts, median_track_error_px=errors, reference_frame=reference_frame, roi=roi,
                        adjacent_recovery_frames=np.array(sorted(fallback_frames), dtype=int))
    (output / 'camera_motion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    preview = background_preview(cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR), available, roi, points)
    cv2.imwrite(str(output / 'background_tracks.jpg'), preview)
    print(f'Background [{mode}]: {len(points)} features, {count} frames, min {counts.min()} inliers, '
          f'p95 rotation fit {report["p95_rotation_fit_error_px"]:.3f} px, {len(fallback_frames)} adjacent recoveries.', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roi', nargs=4, type=int, required=True)
    parser.add_argument('--reference-frame', type=int, default=60)
    parser.add_argument('--tracking-mode', choices=TRACKING_MODES, default='balanced')
    args = parser.parse_args()
    try:
        estimate(args.input, args.roi, args.reference_frame, args.output, mode=args.tracking_mode)
    except BackgroundTrackingError as exc:
        parser.exit(1, f'Background tracking failed: {exc}\n')
