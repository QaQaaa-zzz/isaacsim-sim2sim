import unittest
import numpy as np
from physx_training_math import pd_effort,gae_returns
class TrainingMath(unittest.TestCase):
 def test_unwrap_physx_four_pi_wheel_reset(self):
  from physx_training_math import continuous_joint_position
  result=continuous_joint_position(np.array([6.2]),np.array([-6.2]),np.array([18.7]))
  np.testing.assert_allclose(result,[18.7+4*np.pi-12.4])
 def test_saturated_pd_is_known_applied_force(self):
  np.testing.assert_allclose(pd_effort(np.array([[.2,-.2]]),np.array([[1.,-1.]]),np.zeros((1,2)),[100,100],[6,6],[30,30]),[[-26,26]])
  np.testing.assert_allclose(pd_effort(np.array([[1.,-1.]]),np.zeros((1,2)),np.zeros((1,2)),[100,100],[6,6],[30,30]),[[-30,30]])
 def test_gae_never_leaks_across_reset(self):
  adv,ret=gae_returns(np.array([[1.],[2.]]),np.zeros((2,1)),np.array([[1.],[0.]]),np.array([10.]),gamma=.9,lam=1.)
  np.testing.assert_allclose(adv,[[1],[11]]);np.testing.assert_allclose(ret,adv)
if __name__=='__main__':unittest.main()
