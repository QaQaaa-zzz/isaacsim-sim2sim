import unittest
import numpy as np
from drive_calibration import scaled_gains, configured_gains


class TargetGainsTest(unittest.TestCase):
    def test_absolute_si_gains_follow_joint_names(self):
        p,d=configured_gains(['knee','rear','steering'],np.array([[100.,0.,10.]]),np.array([[6.,5.,.1]]),{'steering':{'kp':2.5,'kd':.4},'rear':{'kp':0.,'kd':.005}})
        np.testing.assert_allclose(p,[[100.,0.,2.5]])
        np.testing.assert_allclose(d,[[6.,.005,.4]])

    def test_joint_name_mapping_and_input_preservation(self):
        names = ['rear', 'front', 'steering']
        kp = np.array([[0., 0., 10.]])
        kd = np.array([[5., 0., .1]])
        p, d = scaled_gains(names, kp, kd, {'rear': {'kp': 1., 'kd': .001}, 'steering': {'kp': .25, 'kd': 4.}})
        np.testing.assert_allclose(p, [[0., 0., 2.5]])
        np.testing.assert_allclose(d, [[.005, 0., .4]])
        np.testing.assert_array_equal(kp, [[0., 0., 10.]])
        np.testing.assert_array_equal(kd, [[5., 0., .1]])

    def test_unknown_joint_or_invalid_gain_is_rejected(self):
        for values in [{'absent': {'kp': 1., 'kd': 1.}}, {'rear': {'kp': -1., 'kd': 1.}}, {'rear': {'kp': 1., 'kd': float('nan')}}]:
            with self.assertRaises(ValueError):
                scaled_gains(['rear'], np.zeros((1, 1)), np.ones((1, 1)), values)


if __name__ == '__main__': unittest.main()
