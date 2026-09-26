import unittest
import numpy as np
from actuator_reference import velocity_step


class AnalyticVelocityTest(unittest.TestCase):
    def test_unsaturated_exponential_and_position(self):
        t = np.array([0., .1, .5])
        q, v = velocity_step(t, inertia=2., damping=4., limit=100., target=3.)
        np.testing.assert_allclose(v, 3*(1-np.exp(-2*t)), atol=1e-12)
        np.testing.assert_allclose(q, 3*t-1.5*(1-np.exp(-2*t)), atol=1e-12)

    def test_saturation_release_and_sign(self):
        # I=2, cap=4 => acceleration=2 until v=3-4/2=1 at t=.5.
        t = np.array([0., .25, .5, 1.])
        q, v = velocity_step(t, inertia=2., damping=2., limit=4., target=3.)
        np.testing.assert_allclose(v, [0., .5, 1., 3-2*np.exp(-.5)])
        np.testing.assert_allclose(q, [0., .0625, .25, .25+1.5-2*(1-np.exp(-.5))])
        nq, nv = velocity_step(t, inertia=2., damping=2., limit=4., target=-3.)
        np.testing.assert_allclose(nq, -q); np.testing.assert_allclose(nv, -v)

    def test_zero_target_is_stationary(self):
        q, v = velocity_step(np.array([0., 1.]), inertia=2., damping=2., limit=4., target=0.)
        np.testing.assert_array_equal(q, 0.); np.testing.assert_array_equal(v, 0.)

if __name__ == '__main__':
    unittest.main()
