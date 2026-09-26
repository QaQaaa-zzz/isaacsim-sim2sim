"""Read actual PhysX tensors and paired finite-difference velocity diagnostics."""
import os,json
from pathlib import Path
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,mujoco,omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation
from usd_model import build
root=Path(__file__).resolve().parent;out=root/'results/domain_audit';out.mkdir(exist_ok=True)
w=World(physics_dt=.005,rendering_dt=.02);w.get_physics_context().enable_fabric(False)
m,conv=build(omni.usd.get_context().get_stage(),root/'model/source.xml')
r=w.scene.add(SingleArticulation('/World/Bike',name='bike'));w.reset();v=r._articulation_view;p=v._physics_view;names=r.dof_names
report={'joints':names,'bodies':v.body_names,'gain_stiffness_damping':[x.tolist() for x in v.get_gains()]}
for method in ['get_dof_limits','get_dof_max_forces','get_dof_friction_coefficients','get_dof_armatures','get_dof_max_velocities','get_masses','get_inertias','get_coms']:
 try:report[method]=getattr(p,method)().tolist()
 except Exception as e:report[method]=str(e)
q=np.array([m.key_qpos[0,m.jnt_qposadr[m.joint(n).id]] for n in names],np.float32)
r.set_joint_positions(q);r.set_joint_velocities(np.zeros(5));r.set_world_pose(m.key_qpos[0,:3],m.key_qpos[0,3:7]);r.set_linear_velocity(np.array([2.,0,0]));r.set_angular_velocity(np.array([.3,.4,.5]));p0=r.get_world_pose()[0].copy()
veltar=np.zeros(5);veltar[names.index('rearwheel_joint')]=12
r.apply_action(ArticulationAction(joint_positions=q,joint_velocities=veltar));rows=[]
for i in range(60):
 w.step(render=False);pos,quat=r.get_world_pose();vel=r.get_linear_velocity();omega=r.get_angular_velocity();R=Rotation.from_quat(np.r_[quat[1:],quat[0]]);corrected=vel-np.cross(omega,R.apply(m.body_ipos[1]));rows.append(np.r_[(i+1)*.005,pos,(pos-p0)/.005,vel,corrected,omega]);p0=pos.copy()
np.savetxt(out/'velocity_semantics.csv',rows,delimiter=',',header='t,x,y,z,fd_vx,fd_vy,fd_vz,api_vx,api_vy,api_vz,corrected_vx,corrected_vy,corrected_vz,wx,wy,wz')
report['velocity_rmse_api_vs_position_fd']=np.sqrt(np.mean((np.array(rows)[:,4:7]-np.array(rows)[:,7:10])**2,axis=0)).tolist();report['velocity_rmse_corrected_vs_position_fd']=np.sqrt(np.mean((np.array(rows)[:,4:7]-np.array(rows)[:,10:13])**2,axis=0)).tolist()
(out/'physx_runtime.json').write_text(json.dumps(report,indent=2));print('AUDIT',json.dumps(report),flush=True);app.close()
