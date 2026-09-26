import unittest,json
from pathlib import Path
import numpy as np,mujoco
from domains import sample_domain,apply_domain,identity_domain,ActionDelay
ROOT=Path(__file__).resolve().parents[1]
class Domains(unittest.TestCase):
 def setUp(self):self.spec=json.loads((ROOT/'configs/domain_randomization.json').read_text())
 def test_seed_and_roles(self):
  a=sample_domain(self.spec,101,'train');b=sample_domain(self.spec,101,'train');self.assertEqual(a,b);self.assertNotEqual(a,sample_domain(self.spec,101,'test'))
 def test_identity_preserves_model(self):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));before=[x.copy() for x in [m.body_mass,m.body_inertia,m.pair_friction,m.actuator_gainprm,m.actuator_biasprm]];apply_domain(m,identity_domain());after=[m.body_mass,m.body_inertia,m.pair_friction,m.actuator_gainprm,m.actuator_biasprm]
  for a,b in zip(before,after):np.testing.assert_array_equal(a,b)
 def test_density_mass_inertia_consistency(self):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));mass=m.body_mass.copy();I=m.body_inertia.copy();d=identity_domain();d['density_scale']=1.1;apply_domain(m,d);np.testing.assert_allclose(m.body_mass,mass*1.1);np.testing.assert_allclose(m.body_inertia,I*1.1)
 def test_pair_friction_explicit(self):
  m=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));d=identity_domain();d.update(contact_mode='isotropic',friction=.7);apply_domain(m,d);np.testing.assert_allclose(m.pair_friction[:,:2],.7)
 def test_delay_reset(self):
  q=ActionDelay(1,np.zeros(4));np.testing.assert_array_equal(q(np.ones(4)),np.zeros(4));np.testing.assert_array_equal(q(np.ones(4)*2),np.ones(4));q.reset();np.testing.assert_array_equal(q(np.ones(4)*3),np.zeros(4))
if __name__=='__main__':unittest.main()
