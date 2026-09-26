import sys,os,json
from pathlib import Path
root=Path('/home/qy/ISAAC——SIM');sys.path.insert(0,str(root));os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4})
import omni.usd,numpy as np
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from usd_model import build
rows=[]
for mode in ['drive','external','drive_limit']:
 omni.usd.get_context().new_stage();World.clear_instance();w=World(physics_dt=.001,rendering_dt=.02);w.get_physics_context().enable_fabric(False);w.get_physics_context().set_gravity(0);stage=omni.usd.get_context().get_stage();m,_=build(stage,root/'results/actuator_audit/rotor.xml',fixed=True)
 if mode=='drive_limit':
  j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Bike/joints/rearwheel_joint'));j.CreateLowerLimitAttr(-.001);j.CreateUpperLimitAttr(.001)
 r=w.scene.add(SingleArticulation('/World/Bike',name='rotor'));w.reset();v=r._articulation_view;p=v._physics_view
 if mode=='external':v.set_gains(kps=np.zeros((1,1)),kds=np.zeros((1,1)));r.set_joint_efforts(np.array([.2]))
 else:r.apply_action(ArticulationAction(joint_velocities=np.array([12.])))
 for k in range(5):
  v0=r.get_joint_velocities().copy();w.step(render=False)
  rows.append(dict(mode=mode,k=k,q=r.get_joint_positions().tolist(),qd=r.get_joint_velocities().tolist(),I_dv_dt=(.0004165*(r.get_joint_velocities()-v0)/.001).tolist(),actuation=p.get_dof_actuation_forces().tolist(),projected=p.get_dof_projected_joint_forces().tolist()))
 w.stop()
(root/'results/isaac_training_engineering/effort_probe.json').write_text(json.dumps(rows,indent=2));print('RESULT',json.dumps(rows),flush=True);app.close()
