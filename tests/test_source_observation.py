import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from policy_runtime import Observation

def test_privileged_state_matches_source_definition():
    m=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/'model/source.xml'))
    ob=Observation(m);names=['rearwheel_joint','steering_joint','hip_joint','frontwheel_joint','knee_joint']
    pos=np.array([2.8,.1,.65]);rot=Rotation.from_euler('xyz',[.1,-.2,.3]);qxy=rot.as_quat();quat=qxy[[3,0,1,2]]
    q=np.array([.4,.1,-1.1,.5,2.2]);qd=np.arange(5)*.1;v=np.array([2,.1,1]);omega=np.array([.2,-.3,.4])
    ob.initial(pos[0]);actor=ob.advance(pos,quat,q,qd,names,v,omega,np.array([.1,.2,.3]),np.zeros(4))
    privileged=ob.privileged(actor,v,omega,qd,names)
    expected_qpos=np.r_[pos,quat,[q[names.index(n)] for n in ['rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint']]]
    expected_qvel=np.r_[v,rot.inv().apply(omega),[qd[names.index(n)] for n in ['rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint']]]
    assert privileged.shape==(106,)
    np.testing.assert_allclose(privileged[:76],actor)
    np.testing.assert_allclose(privileged[76:88],expected_qpos,atol=1e-6)
    np.testing.assert_allclose(privileged[88:99],expected_qvel,atol=1e-6)
    np.testing.assert_allclose(privileged[99:102],[.1,-.2,.3],atol=1e-6)
    np.testing.assert_allclose(privileged[102:105],v,atol=1e-6)
    assert privileged[105]==0
