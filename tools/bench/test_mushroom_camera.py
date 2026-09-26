"""Background selection and reference-coordinate camera tracking regressions."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools import estimate_mushroom_camera as camera
from tools import optimize_mushroom as pipeline


class BackgroundTests(unittest.TestCase):
    def test_balanced_mode_accepts_ten_well_distributed_matches(self):
        p = np.array([[x, y] for y in [80, 240, 400] for x in [80, 220, 380, 550]], np.float32)[:, None]
        p = p[:10]
        K = np.array([[700., 0, 320.], [0, 700., 240.], [0, 0, 1.]])
        angle = Rotation.from_euler('z', .8, degrees=True).as_matrix()
        H = K @ angle @ np.linalg.inv(K)
        q = cv2.perspectiveTransform(p, H)
        self.assertIsNone(camera.fit_rotation(p, q, K, np.linalg.inv(K), mode='strict'))
        detail = {}
        result = camera.fit_rotation(p, q, K, np.linalg.inv(K), mode='balanced', diagnostics=detail)
        self.assertIsNotNone(result)
        self.assertEqual(detail['reason'], 'accepted')
        self.assertEqual(detail['inliers'], 10)
        self.assertLess(Rotation.from_matrix(result[0] @ angle.T).magnitude(), 1e-5)

    def test_balanced_mode_rejects_clustered_matches(self):
        rng = np.random.default_rng(6)
        points = (rng.normal(0, .1, (25, 1, 2)) + [200, 150]).astype(np.float32)
        K = np.diag([700., 700., 1.])
        detail = {}
        result = camera.fit_rotation(points, points + [3., 1.], K, np.linalg.inv(K),
                                     mode='balanced', diagnostics=detail)
        self.assertIsNone(result)
        self.assertIn('narrowly distributed', detail['reason'])

    def test_balanced_mode_still_rejects_zoom_and_reports_actual_error(self):
        p = np.array([[x, y] for x in np.linspace(30, 610, 8) for y in np.linspace(30, 450, 8)], np.float32)[:, None]
        q = (p - [320., 240.]) * 1.2 + [320., 240.]
        K = np.array([[700., 0, 320.], [0, 700., 240.], [0, 0, 1.]])
        detail = {}
        self.assertIsNone(camera.fit_rotation(p, q, K, np.linalg.inv(K), mode='balanced', diagnostics=detail))
        self.assertEqual(detail['inliers'], 64)
        self.assertGreater(detail['rotation_median_px'], 3.5)
        self.assertIn('rotation fit exceeds', camera.describe_fit(detail))

    def test_balanced_mode_rejects_small_consensus_among_outliers(self):
        rng = np.random.default_rng(13)
        p = rng.uniform([30, 30], [610, 450], (100, 1, 2)).astype(np.float32)
        q = rng.uniform([30, 30], [610, 450], (100, 1, 2)).astype(np.float32)
        q[:15] = p[:15] + [3., 1.]
        K = np.array([[700., 0, 320.], [0, 700., 240.], [0, 0, 1.]])
        detail = {}
        self.assertIsNone(camera.fit_rotation(p, q, K, np.linalg.inv(K), mode='balanced', diagnostics=detail))
        self.assertIn('sparse', detail['reason'])

    def test_person_and_padding_exclusion_clips_at_image_edges(self):
        image = np.full((120, 200), 120, np.uint8)
        image[:, :20] = 0
        image[:, 180:] = 0
        mask = camera.background_mask(image, [-100, 20, 50, 80])
        self.assertFalse(mask[:, :20].any())
        self.assertFalse(mask[:, 180:].any())
        self.assertEqual(mask[50, 40], 0)
        self.assertEqual(mask[50, 150], 255)  # Negative box coordinates must not wrap.

    def test_pose_envelope_keeps_empty_box_corners_and_masks_limbs(self):
        image = np.full((300, 400), 120, np.uint8)
        pose = np.array([[110, 65], [103, 62], [117, 62], [96, 65], [124, 65],
                         [90, 90], [130, 90], [85, 130], [140, 130], [80, 160], [150, 160],
                         [125, 160], [155, 160], [190, 205], [210, 205], [280, 260], [310, 260]], float)
        pose = np.c_[pose, np.ones(17)]
        box = [70, 30, 320, 270]
        envelope = camera.background_mask(image, box, pose)
        self.assertEqual(envelope[70, 290], 255)  # Wall inside the loose detector box.
        self.assertEqual(envelope[205, 190], 0)
        self.assertEqual(envelope[65, 110], 0)
        pose[:, 2] = 0
        self.assertEqual(camera.background_mask(image, box, pose)[70, 290], 0)

    def test_feature_count_uses_full_resolution_and_rejects_blank_roi(self):
        rng = np.random.default_rng(17)
        gray = np.full((240, 400), 140, np.uint8)
        gray[30:210, 260:390] = rng.integers(30, 230, (180, 130), dtype=np.uint8)
        available = camera.background_mask(gray, [100, 40, 200, 220])
        points, _ = camera.background_features(gray, available, [260, 30, 390, 210])
        self.assertGreaterEqual(len(points), camera.MIN_FEATURES)
        empty, _ = camera.background_features(gray, available, [20, 30, 70, 180])
        self.assertEqual(len(empty), 0)
        with self.assertRaises(camera.BackgroundTrackingError):
            camera.background_features(gray, available, [-1, 0, 10, 10])

    def test_selector_refuses_enter_on_blank_roi_then_accepts_redraw(self):
        rng = np.random.default_rng(9)
        frame = np.full((240, 400, 3), 140, np.uint8)
        frame[30:210, 260:390] = rng.integers(30, 230, (180, 130, 1), dtype=np.uint8)
        events, previews, mouse = [], [], {}

        def drag(x0, y0, x1, y1):
            callback = mouse['callback']
            callback(cv2.EVENT_LBUTTONDOWN, x0, y0 + camera.PREVIEW_HEADER, 0, None)
            callback(cv2.EVENT_MOUSEMOVE, x1, y1 + camera.PREVIEW_HEADER, 0, None)
            callback(cv2.EVENT_LBUTTONUP, x1, y1 + camera.PREVIEW_HEADER, 0, None)

        def key(delay):
            action = len(events)
            events.append(action)
            if action == 0:
                drag(20, 30, 70, 180)
                return 13  # Mouse redraw and ENTER may arrive in the same waitKey.
            elif action in (1, 3):
                return 13
            elif action == 2:
                drag(260, 30, 390, 210)
            else:
                self.fail('Valid redraw was not accepted')
            return -1

        with patch.multiple(camera.cv2, namedWindow=lambda *a: None,
                            setMouseCallback=lambda name, cb: mouse.update(callback=cb),
                            imshow=lambda name, im: previews.append(im.copy()), waitKey=key,
                            getWindowProperty=lambda *a: 1, destroyWindow=lambda *a: None), contextlib.redirect_stdout(io.StringIO()):
            roi = camera.select_background(frame, [100, 40, 200, 220], initial_roi=[260, 30, 390, 210])
        self.assertEqual(roi, [260, 30, 390, 210])
        self.assertEqual(len(events), 4)
        self.assertGreater(len(previews), 2)
        red_pixel = previews[-1][camera.PREVIEW_HEADER + 100, 150]
        self.assertGreater(int(red_pixel[2]), int(red_pixel[0]))

    def test_selector_cancel_destroys_window(self):
        with patch.multiple(camera.cv2, namedWindow=lambda *a: None, setMouseCallback=lambda *a: None,
                            imshow=lambda *a: None, waitKey=lambda *a: 27,
                            getWindowProperty=lambda *a: 1), patch.object(camera.cv2, 'destroyWindow') as close:
            with self.assertRaisesRegex(camera.BackgroundTrackingError, 'cancelled'):
                camera.select_background(np.full((80, 100, 3), 120, np.uint8), None)
            close.assert_called_once()

    def test_scaled_selector_returns_original_pixels(self):
        rng = np.random.default_rng(5)
        frame = rng.integers(30, 220, (1080, 1920, 3), dtype=np.uint8)
        mouse = {}
        calls = 0

        def key(delay):
            nonlocal calls
            calls += 1
            # Preview is 1200 x 675, with a separate header above the image.
            if calls == 1:
                cb = mouse['callback']
                cb(cv2.EVENT_LBUTTONDOWN, 100, camera.PREVIEW_HEADER + 50, 0, None)
                cb(cv2.EVENT_LBUTTONUP, 350, camera.PREVIEW_HEADER + 200, 0, None)
                return -1
            return 13

        with patch.multiple(camera.cv2, namedWindow=lambda *a: None,
                            setMouseCallback=lambda name, cb: mouse.update(callback=cb), imshow=lambda *a: None,
                            waitKey=key, getWindowProperty=lambda *a: 1, destroyWindow=lambda *a: None):
            roi = camera.select_background(frame, None)
        self.assertEqual(roi, [160, 80, 560, 320])


class CameraSequenceTests(unittest.TestCase):
    def make_sequence(self, directory, blank=False):
        folder = Path(directory)
        (folder / 'preprocess').mkdir()
        width, height, count = 640, 480, 13
        K = np.array([[700., 0, width / 2], [0, 700., height / 2], [0, 0, 1]])
        rng = np.random.default_rng(7)
        base = rng.integers(40, 220, (height, width), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (5, 5), 0)
        if blank:
            base[:] = 120
        rotations = Rotation.from_euler('xyz', np.c_[np.arange(count) * .08, np.arange(count) * .11,
                                                     np.arange(count) * .03], degrees=True).as_matrix()
        writer = cv2.VideoWriter(str(folder / '0_input_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        self.assertTrue(writer.isOpened())
        for rotation in rotations:
            gray = cv2.warpPerspective(base, K @ rotation @ np.linalg.inv(K), (width, height), borderValue=120)
            writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        writer.release()
        torch.save({'K_fullimg': torch.from_numpy(np.repeat(K[None], count, axis=0))}, folder / 'hmr4d_results.pt')
        boxes = torch.tensor([[300., 260., 400., 450.]]).repeat(count, 1)
        torch.save({'bbx_xyxy': boxes}, folder / 'preprocess/bbx.pt')
        return folder, rotations

    def test_middle_reference_tracks_both_directions_in_reference_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder, expected = self.make_sequence(tmp)
            output = folder / 'camera'
            report = camera.estimate(folder, [30, 30, 600, 230], 6, output)
            with np.load(output / 'camera_motion.npz') as data:
                observed = data['raw_rotations']
                angle = Rotation.from_matrix(observed @ (expected @ expected[6].T).transpose(0, 2, 1)).magnitude()
                self.assertLess(np.rad2deg(angle).max(), .08)
                np.testing.assert_allclose(data['rotations'][6], np.eye(3), atol=1e-10)
                self.assertGreaterEqual(data['inliers'].min(), camera.MIN_INLIERS)
            self.assertEqual(report['frames'], 13)

    def test_adjacent_recovery_keeps_reference_anchor_beyond_direct_matching(self):
        track = camera.track_points
        reference = None
        direct_calls = 0

        def reject_long_baseline(source, target, points, available):
            nonlocal reference, direct_calls
            if reference is None:
                reference = source
            if source is reference:
                direct_calls += 1
                if direct_calls > 1:
                    return np.empty((0, 1, 2), np.float32), np.zeros(len(points), bool)
            return track(source, target, points, available)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder, expected = self.make_sequence(tmp)
            with patch.object(camera, 'track_points', side_effect=reject_long_baseline):
                report = camera.estimate(folder, [30, 30, 600, 230], 0, folder / 'camera')
            with np.load(folder / 'camera/camera_motion.npz') as data:
                error = Rotation.from_matrix(data['raw_rotations'] @ expected.transpose(0, 2, 1)).magnitude()
                self.assertLess(np.rad2deg(error).max(), .1)
            self.assertEqual(report['adjacent_recovery_frames'], list(range(2, 13)))

    def test_failure_does_not_write_partial_camera_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.make_sequence(tmp, blank=True)
            with self.assertRaises(camera.BackgroundTrackingError):
                camera.estimate(folder, [30, 30, 600, 230], 0, folder / 'camera')
            self.assertFalse((folder / 'camera').exists())

    def test_background_reselection_and_failed_tracking_preserve_apparatus_cache(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            folder, _ = self.make_sequence(tmp)
            torch.save(torch.ones(13, 17, 3), folder / 'preprocess/vitpose.pt')
            bvh, blender, addon = folder / 'prior.bvh', folder / 'blender.exe', folder / 'addon'
            bvh.touch(); blender.touch(); addon.mkdir(); (addon / '__init__.py').touch()
            # Keep every file inside the temporary directory for automatic cleanup.
            source = folder / 'input'
            source.mkdir()
            for path in [folder / 'preprocess', folder / 'hmr4d_results.pt', folder / '0_input_video.mp4']:
                path.rename(source / path.name)
            output = folder / 'output'
            config_path = output / '_setup/input/annotations.json'
            config_path.parent.mkdir(parents=True)
            apparatus = [[90, 100], [60, 120], [120, 120], [100, 190]]
            config_path.write_text(json.dumps(dict(source_folder=str(source.resolve()), reference_frame=6,
                                                   camera_roi=[30, 30, 600, 230], apparatus_pixels=apparatus,
                                                   keep_range=[0, 13], circle_range=[3, 10], custom_note='preserve')))
            args = ['optimize_mushroom.py', '--input', str(source), '--output-root', str(output),
                    '--start', '0', '--end', '13', '--circle-start', '3', '--circle-end', '10',
                    '--bvh', str(bvh), '--blender', str(blender), '--addon', str(addon), '--reselect-background',
                    '--camera-tracking', 'strict']
            selected = [[40, 30, 580, 230], [45, 35, 570, 225]]
            with patch.object(sys, 'argv', args), patch.object(camera, 'select_background', side_effect=selected) as select, \
                    patch.object(camera, 'estimate', side_effect=[camera.BackgroundTrackingError('occluded'), {}]) as estimate, \
                    patch.object(pipeline, 'annotate_apparatus') as annotate, patch.object(pipeline.subprocess, 'run') as run:
                pipeline.main()
            saved = json.loads(config_path.read_text())
            self.assertEqual(saved['apparatus_pixels'], apparatus)
            self.assertEqual(saved['camera_roi'], selected[-1])
            self.assertEqual(saved['custom_note'], 'preserve')
            self.assertEqual(select.call_count, 2)
            self.assertEqual(estimate.call_count, 2)
            self.assertEqual(estimate.call_args.kwargs['mode'], 'strict')
            annotate.assert_not_called()
            self.assertGreater(run.call_count, 0)


if __name__ == '__main__':
    cv2.setNumThreads(2)
    unittest.main()
