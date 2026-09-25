"""Behavior checks for stage detection and shape-preserving cycle statistics."""
import sys
import io
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from hmr4d.utils.mushroom_priors import infer_bvh_support, infer_video_support, trajectory_statistics
from hmr4d.utils.mushroom_contact import support_weights, palm_orientation_loss
from hmr4d.utils.mushroom_legs import circle_envelope
from hmr4d.utils.mushroom_feet import bvh_foot_prior
from hmr4d.utils.mushroom_io import save_diagnostics


class MushroomPriorsTest(unittest.TestCase):
    def test_foot_prior_is_relative_to_shin_not_world_heading(self):
        from scipy.spatial.transform import Rotation
        names=['LeftHip','RightHip','LeftKnee','RightKnee','LeftAnkle','RightAnkle','LeftToe','RightToe']
        j=np.array([[.1,0,1],[-.1,0,1],[.1,0,.6],[-.1,0,.6],[.1,0,.2],[-.1,0,.2],[.1,-.1,.15],[-.1,-.1,.15]])
        bvh=dict(names=names,joints=np.tile(j,(20,1,1)),rotations=np.tile(np.eye(3),(20,8,1,1)),
                 ends=dict(LeftToe=[0,-.08,0],RightToe=[0,-.08,0]))
        prior,_=bvh_foot_prior(bvh,[0,20])
        rot=Rotation.from_euler('xyz',[.8,-.5,1.4]).as_matrix()
        bvh['joints']=bvh['joints']@rot.T+np.array([2,3,4])
        bvh['rotations']=rot@bvh['rotations']
        transformed,_=bvh_foot_prior(bvh,[0,20])
        np.testing.assert_allclose(prior,transformed,atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(prior),1.)

    def test_compact_diagnostics_keep_preview_data_without_surface_cache(self):
        arrays=dict(original_surface_zup=np.zeros((2,100,3)),corrected_surface_zup=np.ones((2,100,3)),
                    corrected_joints_zup=np.ones((2,22,3)),camera_t=np.ones((2,3)))
        stream=io.BytesIO();save_diagnostics(stream,arrays);stream.seek(0)
        with np.load(stream) as data:
            self.assertNotIn('corrected_surface_zup',data.files)
            self.assertNotIn('original_surface_zup',data.files)
            np.testing.assert_array_equal(data['corrected_joints_zup'],arrays['corrected_joints_zup'])
            np.testing.assert_array_equal(data['camera_t'],arrays['camera_t'])
        stream=io.BytesIO();save_diagnostics(stream,arrays,full=True);stream.seek(0)
        with np.load(stream) as data:self.assertIn('corrected_surface_zup',data.files)

    def test_closed_leg_ramp_protects_both_stages_and_boundary_velocity(self):
        gate=circle_envelope(373,165,311,30.)
        self.assertTrue(np.all(gate[:167]==0))
        self.assertTrue(np.all(gate[309:]==0))
        self.assertTrue(np.all(gate[178:298]==1))
        self.assertTrue(np.all(np.diff(gate[166:179])>=0))
        np.testing.assert_allclose(gate[165:311],gate[165:311][::-1])
        # Even the maximum allowed correction is less than half a degree at
        # the first active frame; a large raw optimizer variable cannot undo it.
        self.assertLess(np.degrees(1.5*gate[167]),.5)

    def test_closed_leg_ramp_rejects_invalid_stage(self):
        for start,end,fps in [(-1,90,30),(20,101,30),(20,20,30),(20,90,0)]:
            with self.assertRaises(ValueError):circle_envelope(100,start,end,fps)

    def test_palm_gate_does_not_spread_into_flight(self):
        contact=np.zeros((60,2));contact[10:40,0]=1.;contact[10:40,1]=.1
        weights=support_weights(contact,30,[5,55])
        self.assertEqual(weights[:10].max(),0.)
        self.assertEqual(weights[40:].max(),0.)
        self.assertEqual(weights[:,1].max(),0.)
        self.assertLess(weights[10,0],weights[14,0])
        self.assertGreater(weights[20,0],.99)

    def test_airborne_palm_gets_no_orientation_gradient(self):
        theta=torch.tensor([1.2,1.2],requires_grad=True)
        normals=torch.stack([torch.sin(theta),torch.zeros_like(theta),-torch.cos(theta)],-1)
        loss=palm_orientation_loss(normals,torch.tensor([[0.,0.,-1.]]).expand_as(normals),torch.tensor([1.,0.]))
        loss.backward()
        self.assertGreater(theta.grad[0].item(),0.)
        self.assertEqual(theta.grad[1].item(),0.)

    def test_support_tilt_inside_cone_is_free(self):
        theta=torch.tensor([np.deg2rad(20.)],requires_grad=True)
        normals=torch.stack([torch.sin(theta),torch.zeros_like(theta),-torch.cos(theta)],-1)
        loss=palm_orientation_loss(normals,torch.tensor([[0.,0.,-1.]]),torch.ones(1),30.)
        loss.backward()
        self.assertEqual(loss.item(),0.)
        self.assertEqual(theta.grad.item(),0.)

    def test_stationary_hands_at_body_are_not_apparatus_support(self):
        joints = np.zeros((160, 2, 3))
        joints[:, :, 0] = [-.1, .1]
        joints[:, :, 2] = .45
        joints[:30, :, 1] = .7
        joints[:30, :, 2] = .9
        bvh = dict(joints=joints, names=['LeftWrist', 'RightWrist'], fps=30.)
        mask, info = infer_bvh_support(bvh, [100, 155])
        self.assertFalse(mask[:25].any())
        self.assertTrue(mask[50:90].all())
        self.assertIsNotNone(info['prep_double_support'])

    def test_video_release_is_not_forced_to_double_support(self):
        kp = np.zeros((140, 17, 3)); kp[:, :, 2] = .9
        kp[:, 9, :2] = [110, 85]; kp[:, 10, :2] = [90, 85]
        kp[45:70, 10, :2] = [20, 25]
        cfg = dict(keep_range=[0, 140], circle_range=[90, 120], fps=30,
                   apparatus_pixels=[[100, 100], [60, 120], [140, 120], [100, 180]])
        contact, _, _ = infer_video_support(kp, cfg)
        self.assertTrue((contact[20:35] > .9).all())
        self.assertLess(contact[55:60, 1].max(), .01)
        self.assertTrue((contact[55:60, 0] > .9).all())
        self.assertEqual(contact[95:115].max(), 0.)

    def test_ellipse_is_accepted_without_becoming_a_circle(self):
        phase = np.linspace(0, 6*np.pi, 181)
        j = np.zeros((181, 22, 3))
        path = np.stack([1.3*np.cos(phase), .7*np.sin(phase), .2+.05*np.sin(2*phase)], -1)
        j[:, 7] = j[:, 8] = path
        stats = trajectory_statistics(j, phase, [0, 60, 120, 180])
        self.assertLess(stats['cycle_variation_rms_cm'], 1e-8)
        self.assertLess(stats['mean_path_high_frequency_rms_cm'], 1e-8)
        j[80:85, [7, 8], 0] += .12
        changed = trajectory_statistics(j, phase, [0, 60, 120, 180])
        self.assertGreater(changed['cycle_variation_rms_cm'], 1.)


if __name__ == '__main__':
    unittest.main()
