"""Configuration, ablation isolation and video-independent stage checks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hmr4d.utils.mushroom_config import (
    DEFAULT_CONSTRAINTS,
    PRESETS,
    apply_constraints,
    constraint_hash,
    pixel_scale,
    resolve_constraints,
    save_constraint_snapshot,
    weighted_loss,
)
from hmr4d.utils.mushroom_priors import circle_envelope
from hmr4d.utils.mushroom_feet import refine_circle_feet
from hmr4d.utils.mushroom_legs import refine_closed_legs
from hmr4d.utils.mushroom_priors import infer_video_support


class MushroomConfigTest(unittest.TestCase):
    def test_legacy_and_explicit_override_precedence(self):
        annotations = dict(circle_range=[60, 180], foot_refinement=False, constraints={"feet": {"enabled": True}})
        cfg = apply_constraints(annotations, {"feet": {"enabled": False}})
        self.assertFalse(cfg["foot_refinement"])
        self.assertFalse(cfg["constraints"]["feet"]["enabled"])
        self.assertEqual(cfg["circle_range"], [60, 180])
        self.assertTrue(annotations["constraints"]["feet"]["enabled"])
        self.assertEqual(resolve_constraints()["legs"]["iterations"], 800)

    def test_invalid_config_is_rejected(self):
        invalid = [
            {"feet": {"weigths": {}}},
            {"circle_range": [165, 311]},
            {"legs": {"enabled": "false"}},
            {"legs": {"iterations": 0}},
            {"legs": {"fade_seconds": 0}},
            {"base": {"weights": {"image": -1}}},
            {"feet": {"weights": {"parallel": float("nan")}}},
            {"shared": {"image_sigma_px": 0}},
            {"hands": {"self_collision_sigma_m": 0}},
            {"hands": {"detection": {"release_lift_forearm_start": .5, "release_lift_forearm_end": .4}}},
            {"schema_version": 2},
        ]
        for patch in invalid:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                resolve_constraints(overrides=patch)

    def test_snapshot_roundtrip_and_hash(self):
        cfg = apply_constraints({}, PRESETS["no-feet"])
        with tempfile.TemporaryDirectory() as tmp:
            digest = save_constraint_snapshot(tmp, cfg)
            saved = json.loads(Path(tmp, "constraints.json").read_text())
        self.assertEqual(saved, cfg["constraints"])
        self.assertEqual(digest, constraint_hash(saved))
        self.assertNotEqual(digest, constraint_hash(DEFAULT_CONSTRAINTS))

    def test_default_template_matches_authoritative_defaults(self):
        saved = json.loads((ROOT / "tools/configs/mushroom_constraints.json").read_text())
        self.assertEqual(saved, DEFAULT_CONSTRAINTS)

    def test_disabled_local_stages_do_not_require_models_or_motion_metadata(self):
        prediction = {"arbitrary_tensor": torch.randn(8, 3)}
        arrays = {"preserved": np.arange(8)}
        for fn, group, flag in [(refine_closed_legs, "legs", "closed_legs"), (refine_circle_feet, "feet", "feet")]:
            out, data, metrics = fn(prediction, arrays, {}, None, {"constraints": {group: {"enabled": False}}})
            self.assertTrue(torch.equal(out["arbitrary_tensor"], prediction["arbitrary_tensor"]))
            np.testing.assert_array_equal(data["preserved"], arrays["preserved"])
            self.assertFalse(metrics["active_refinement"][flag])
            prior_metrics = {"active_refinement": {flag: True}}
            _, _, unchanged = fn(prediction, arrays, prior_metrics, None, {"constraints": {group: {"enabled": False}}})
            self.assertEqual(unchanged, prior_metrics)

    def test_zero_loss_weight_removes_its_gradient(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        terms = dict(a=x[0] ** 2, b=x[1] ** 2)
        weighted_loss(terms, dict(a=1.0, b=0.0)).backward()
        torch.testing.assert_close(x.grad, torch.tensor([4.0, 0.0]))
        x.grad = None
        weighted_loss(dict(a=x[0] ** 2, b=x[1] ** 2), dict(a=1.0, b=2.0), enabled=False).backward()
        torch.testing.assert_close(x.grad, torch.zeros(2))

    def test_resizing_video_preserves_contact_detection_and_pixel_residual(self):
        kp = np.zeros((140, 17, 3))
        kp[:, :, 2] = 0.9
        kp[:, 9, :2] = [110, 85]
        kp[:, 10, :2] = [90, 85]
        kp[45:70, 10, :2] = [20, 25]
        cfg = dict(
            keep_range=[0, 140],
            circle_range=[90, 120],
            fps=30,
            image_size=[1920, 1080],
            apparatus_pixels=[[100, 100], [60, 120], [140, 120], [100, 180]],
        )
        expected = infer_video_support(kp, cfg)
        resized = copy.deepcopy(cfg)
        resized["apparatus_pixels"] = (np.array(cfg["apparatus_pixels"]) * 0.5).tolist()
        resized["image_size"] = [960, 540]
        small = kp.copy()
        small[:, :, :2] *= 0.5
        actual = infer_video_support(small, resized)
        for a, b in zip(expected[:2], actual[:2]):
            np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertEqual(20 / pixel_scale(cfg), 10 / pixel_scale(resized))

    def test_stage_gate_is_not_tied_to_a_video_or_frame_rate(self):
        for length, start, end, fps in [(300, 55, 195, 25), (700, 200, 600, 60), (200, 25, 100, 24)]:
            gate = circle_envelope(length, start, end, fps)
            self.assertTrue((gate[: start + 2] == 0).all())
            self.assertTrue((gate[end - 2 :] == 0).all())
            self.assertEqual(gate.max(), 1.0)
            self.assertLess(np.abs(np.diff(gate)).max(), 0.3)

    def test_config_cli_rejects_typos_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, "invalid.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/configure_mushroom_constraints.py"),
                    "--set",
                    "feet.weights.typographical_error=1",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Unknown scalar", proc.stderr)
            self.assertFalse(output.exists())

    def test_shared_stage_cli_preserves_skip_and_repeat_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, "corrected")
            folder.mkdir()
            config = folder / "config.json"
            config.write_text(json.dumps({"foot_refinement": False}))
            (folder / "provenance.json").write_text(json.dumps({"source": str(Path(tmp, "original/hmr4d_results.pt"))}))
            command = [
                sys.executable,
                str(ROOT / "tools/refine_mushroom.py"),
                "--stage",
                "feet",
                "--input",
                str(folder),
                "--overwrite",
            ]
            skipped = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(skipped.returncode, 0, skipped.stderr)
            self.assertIn("disabled", skipped.stdout)
            # No motion file is present: a disabled stage must not attempt to load it.
            config.write_text("{}")
            (folder / "metrics.json").write_text(json.dumps({"active_refinement": {"feet": True}}))
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already applied", repeated.stderr)
            wrong_stage_option = subprocess.run(
                command + ["--camera-motion", "unused.npz"], capture_output=True, text=True
            )
            self.assertNotEqual(wrong_stage_option.returncode, 0)
            self.assertIn("base-only options", wrong_stage_option.stderr)


if __name__ == "__main__":
    unittest.main()
