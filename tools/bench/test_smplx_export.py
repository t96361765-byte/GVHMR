"""Regression tests for the neutral animation interchange contract."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from hmr4d.utils.export_smplx import save_smplx_animation


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "SMPLX_NEUTRAL.npz"
        np.savez(self.model, hands_meanl=np.ones(45) * .2, hands_meanr=np.ones(45) * -.3)
        self.pred = {frame: {"body_pose": torch.zeros(2, 63), "betas": torch.zeros(2, 10),
                             "global_orient": torch.zeros(2, 3), "transl": torch.zeros(2, 3)}
                     for frame in ("smpl_params_global", "smpl_params_incam")}

    def test_full_pose_camera_separation_and_shape(self):
        self.pred["smpl_params_global"]["transl"][:, 0] = 5
        self.pred["smpl_params_incam"]["transl"][:, 2] = 2
        self.pred["smpl_params_global"]["betas"][1] = 2
        result = save_smplx_animation(self.pred, self.root / "motion.npz", self.model, fps=30)
        with np.load(result, allow_pickle=False) as data:
            self.assertEqual(str(data["gender"]), "neutral")
            self.assertEqual(data["poses"].shape, (2, 165))
            np.testing.assert_array_equal(data["trans"], [[5, 0, 0], [5, 0, 0]])
            np.testing.assert_array_equal(data["incam_transl"], [[0, 0, 2], [0, 0, 2]])
            np.testing.assert_allclose(data["poses"][:, 75:120], .2)
            np.testing.assert_allclose(data["poses"][:, 120:], -.3)
            np.testing.assert_array_equal(data["poses"][:, 66:75], 0)
            np.testing.assert_array_equal(data["betas"], 1)
            np.testing.assert_array_equal(data["betas_per_frame"][1], 2)
            self.assertEqual(float(data["mocap_frame_rate"]), 30)

    def test_reject_nonfinite_or_wrong_shape(self):
        self.pred["smpl_params_global"]["transl"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            save_smplx_animation(self.pred, self.root / "bad.npz", self.model)
        self.pred["smpl_params_global"]["transl"] = torch.zeros(1, 3)
        with self.assertRaises(ValueError):
            save_smplx_animation(self.pred, self.root / "bad.npz", self.model)


if __name__ == "__main__":
    unittest.main()
