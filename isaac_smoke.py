"""Actual PhysX articulated smoke and RTX video, no learned policy claim."""
import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fixed','free','policy'],required=True);p.add_argument('--duration',type=float,default=4.0);p.add_argument('--output',type=Path,required=True);p.add_argument('--policy-dir',type=Path,default=Path(__file__).resolve().parent/'policy')
a=p.parse_args();assert 0<a.duration<=10
root=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True)
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':640,'height':480,'renderer':'RaytracedLighting','anti_aliasing':0,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np
import imageio.v2 as imageio
import mujoco
import omni.usd,omni.kit.commands
from pxr import UsdGeom,UsdPhysics,UsdLux,PhysxSchema,Gf,Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
enable_extension('isaacsim.asset.importer.mjcf')
stage=omni.usd.get_context().get_stage();UsdGeom.SetStageMetersPerUnit(stage,1.0);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
world=World(physics_dt=.005,rendering_dt=.04,stage_units_in_meters=1.0)
world.get_physics_context().enable_fabric(False)
from omni.physx import get_physx_interface
from usd_model import build
m,conversion=build(stage,root/'model/source.xml',fixed=a.mode=='fixed')
from scene_visuals import enhance
enhance(stage,m)
(a.output/'conversion_audit.json').write_text(json.dumps(conversion,indent=2))
source=json.loads((root/'model/source_audit.json').read_text());m=mujoco.MjModel.from_xml_path(str(root/'model/source.xml'))
# Inspect actual imported body identities; fail instead of guessing missing links.
bodies={prim.GetName():prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
mass_rows=[]
for row in source['bodies']:
 prim=bodies[row['name']];api=UsdPhysics.MassAPI(prim);mass=api.GetMassAttr().Get();inertia=api.GetDiagonalInertiaAttr().Get()
 mass_rows.append({'body':row['name'],'source_mass':row['mass'],'imported_mass':mass,'source_inertia':row['principal_inertia'],'imported_inertia':list(inertia)})
 np.testing.assert_allclose(mass,row['mass'],rtol=1e-5,atol=1e-7)
 np.testing.assert_allclose(inertia,row['principal_inertia'],rtol=1e-4,atol=1e-7)
(a.output/'mass_audit.json').write_text(json.dumps(mass_rows,indent=2))
# White area lighting, no external scene assets.
light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(650.0);light.CreateColorAttr(Gf.Vec3f(.85,.9,1))
# Importer retains original floor/step; camera is close to the original root.
robot=world.scene.add(SingleArticulation(prim_path='/World/Bike',name='bike'))
camera=Camera('/World/Camera',frequency=-1,resolution=(640,480));camera.initialize()
set_camera_view(eye=np.array([1.0,-1.3,.8]),target=np.array([0,0,.32]),camera_prim_path='/World/Camera')
camera.set_focal_length(3.0)
world.reset();names=robot.dof_names;print('JOINT_NAMES',names,flush=True)
assert set(names)=={r['name'] for r in source['joints'] if r['type']!=0}
q0=np.array([m.key_qpos[0,m.jnt_qposadr[m.joint(n).id]] for n in names],dtype=np.float32)
robot.set_joint_positions(q0);robot.set_joint_velocities(np.zeros(5,dtype=np.float32))
ctrl0=np.array([0.0,0.0,-1.2,2.5])
view=robot._articulation_view
physx_mass=view.get_body_masses()[0]
for name,value in zip(view.body_names,physx_mass):np.testing.assert_allclose(value,m.body(name).mass[0],rtol=1e-5,atol=1e-7)
(a.output/'physx_mass_audit.json').write_text(json.dumps(dict(zip(view.body_names,map(float,physx_mass))),indent=2))
act_indices=np.array([names.index(r['joint']) for r in source['actuators']])
# Explicit keyframe zero velocity. Floating origin is imported at XML position.
if a.mode in ('free','policy'):
 robot.set_world_pose(position=m.body_pos[1].copy(),orientation=m.body_quat[1].copy())
 robot.set_linear_velocity(np.zeros(3));robot.set_angular_velocity(np.zeros(3))
if a.mode=='policy':
 from policy_runtime import Actor,Observation,control
 from endpoint import Endpoint
 terminal_monitor=Endpoint();terminal_status={};terminated=False
 import hashlib
 identity=json.loads((a.policy_dir/'identity.json').read_text());assert identity['xml_sha256']==source['sha256']
 cfg=json.loads((a.policy_dir/'resolved_config.json').read_text());actor=Actor(a.policy_dir/'actor.npz');observer=Observation(m)
 robot.set_world_pose(position=m.key_qpos[0,:3].copy(),orientation=m.key_qpos[0,3:7].copy());robot.set_linear_velocity(np.array([cfg['reset']['initial_forward_velocity'],0.,0.]))
 obs=observer.initial(float(m.key_qpos[0,0]));policy_rows=[];last_action=np.zeros(4);prev_origin_velocity=np.array([2.,0.,0.]);acceleration=np.zeros(3)
world.physics_sim_view.update_articulations_kinematic()
world.render()
for _ in range(12):world.render()
stage.GetRootLayer().Export(str(a.output/'scene.usda'))
writer=imageio.get_writer(a.output/'simulation.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
from PIL import Image,ImageDraw
rows=[];frame_count=0;start=time.time();endpoint='horizon';video_error=None;fk_audit=[];render_pose_errors=[];md=mujoco.MjData(m)
for i in range(round(a.duration/.005)):
 t=i*.005;q=robot.get_joint_positions();qd=robot.get_joint_velocities();ctrl=ctrl0.copy()
 if a.mode=='policy' and i%4==0:
  last_action=actor(obs);ctrl=control(last_action,m,cfg);policy_rows.append((t,obs.copy(),last_action.copy(),ctrl.copy()))
 elif a.mode=='policy':ctrl=held_ctrl.copy()
 if a.mode=='fixed':
  ctrl[0]=.15*np.sin(2*np.pi*.5*t);ctrl[1]=5.0*(1-np.exp(-t));ctrl[2]=q0[names.index('hip_joint')]+.2*(1-np.cos(2*np.pi*.4*t));ctrl[3]=q0[names.index('knee_joint')]-.3*(1-np.cos(2*np.pi*.4*t))
 # Same target/force/gain contract, integrated implicitly by PhysX.
 ctrl=np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
 if i%4==0:
  pos_target=q0.copy();vel_target=np.zeros(5,dtype=np.float32)
  for act,jidx in enumerate(act_indices):
   kp=-m.actuator_biasprm[act,1];kd=-m.actuator_biasprm[act,2];gain=m.actuator_gainprm[act,0]
   if kp:pos_target[jidx]=ctrl[act]*gain/kp
   else:vel_target[jidx]=ctrl[act]*gain/kd
  robot.apply_action(ArticulationAction(joint_positions=pos_target,joint_velocities=vel_target))
  held_ctrl=ctrl.copy()
 world.step(render=False)
 eff=robot.get_measured_joint_efforts()
 pos,quat=robot.get_world_pose();qn=robot.get_joint_positions();qdn=robot.get_joint_velocities()
 if a.mode=='policy':
  # PhysX linear velocity is COM velocity; actor expects root-origin world velocity.
  from scipy.spatial.transform import Rotation
  rr=Rotation.from_quat(np.r_[quat[1:],quat[0]]);omega=robot.get_angular_velocity();velocity=robot.get_linear_velocity()-np.cross(omega,rr.apply(m.body_ipos[1]))
  acceleration=(velocity-prev_origin_velocity)/.005;prev_origin_velocity=velocity.copy()
  if i%4==3:
   obs=observer.advance(pos,quat,qn,qdn,names,velocity,omega,acceleration,last_action);terminal_status=terminal_monitor.advance(pos,quat,velocity);terminated=terminal_status['terminated'] or terminal_status['truncated']
   if terminated:endpoint='phase_u_end_'+str(terminal_status['end_code'])
 if not np.isfinite(np.r_[pos,quat,qn,qdn]).all():endpoint='nonfinite';break
 if i in (0,round(a.duration/.005)-1) or (a.mode=='policy' and terminated):
  md.qpos[:3]=pos;md.qpos[3:7]=quat
  for idx,n in enumerate(names):md.qpos[m.jnt_qposadr[m.joint(n).id]]=qn[idx]
  mujoco.mj_forward(m,md)
  lt=view._physics_view.get_link_transforms()[0]
  expected=np.array([md.xpos[m.body(n).id] for n in view.body_names]);err=np.max(np.abs(lt[:,:3]-expected))
  from scipy.spatial.transform import Rotation
  expected_q=md.xquat[[m.body(n).id for n in view.body_names]][:,[1,2,3,0]];orientation_error=(Rotation.from_quat(lt[:,3:7]).inv()*Rotation.from_quat(expected_q)).magnitude()
  fk_audit.append({'max_link_orientation_error_rad':float(np.max(orientation_error)),'step':i+1,'max_link_position_error_m':float(err),'physics_link_positions':lt[:,:3].tolist(),'mujoco_forward_kinematics_positions':expected.tolist(),'body_names':view.body_names})
 rows.append(((i+1)*.005,qn.copy(),qdn.copy(),pos.copy(),quat.copy(),eff.copy(),held_ctrl.copy()))
 if i%8==7 or (a.mode=='policy' and terminated):
  if a.mode=='policy':set_camera_view(eye=pos+np.array([.9,-1.5,.7]),target=pos+np.array([.15,0,.22]),camera_prim_path='/World/Camera')
  get_physx_interface().update_transformations(False,True,True)
  world.render();world.render();world.render();rgba=camera.get_rgba()
  lt_render=view._physics_view.get_link_transforms()[0]
  cache=UsdGeom.XformCache();usd_positions=np.array([list(cache.GetLocalToWorldTransform(bodies[n]).ExtractTranslation()) for n in view.body_names])
  render_pose_errors.append(float(np.max(np.abs(usd_positions-lt_render[:,:3]))))
  if rgba is None or rgba.size==0:video_error='empty camera frame';continue
  im=Image.fromarray(rgba[:,:,:3].astype(np.uint8));draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,43),fill=(15,20,30));draw.text((10,5),f'DVGC bike | Isaac Sim / PhysX | {a.mode} base | t={(i+1)*.005:.2f}s',fill='white');draw.text((10,23),(('Frozen original actor' if a.policy_dir.resolve()==(root/'policy').resolve() else 'DR-derived actor; not adopted') if a.mode=='policy' else 'Joint/asset smoke; no learned controller'),fill='white');writer.append_data(np.asarray(im));frame_count+=1
  if frame_count==1:im.save(a.output/'preview.png')
 if i%200==199:print('STEP',i+1,'base',pos,'q',qn,flush=True)
 if a.mode=='policy' and terminated:break
writer.close()
if a.mode=='policy':
 np.savez_compressed(a.output/'policy_trace.npz',time=np.array([r[0] for r in policy_rows]),observations=np.array([r[1] for r in policy_rows]),actions=np.array([r[2] for r in policy_rows]),controls=np.array([r[3] for r in policy_rows]))
 (a.output/'checkpoint_identity.json').write_text(json.dumps(identity,indent=2))
assert rows,'No physical transitions'
z={key:np.array([r[k] for r in rows]) for k,key in enumerate(['time','q','qd','base_position','base_quaternion','effort','control'])}
np.savez_compressed(a.output/'trajectory.npz',**z)
header=['time_s']+['q_'+n+'_rad' for n in names]+['qd_'+n+'_rad_s' for n in names]+['base_x_m','base_y_m','base_z_m']+['quat_w','quat_x','quat_y','quat_z']+['measured_joint_effort_'+n+'_Nm' for n in names]+['ctrl_'+r['joint'] for r in source['actuators']]
np.savetxt(a.output/'trajectory.csv',np.column_stack(list(z.values())),delimiter=',',header=','.join(header),comments='')
result={'engine':'Isaac Sim / PhysX','isaacsim_version':'5.1.0.0','mode':a.mode,'terminal_status':terminal_status if a.mode=='policy' else {},'source_sha256':source['sha256'],'joint_names':names,'mass_audit_passed':True,'physics_dt_s':.005,'target_update_dt_s':.02,'servo':'PhysX implicit force drive with source SI gains and force limits','effort_semantics':'PhysX projected measured joint effort; not isolated motor torque','jit_control_dt_s':.02,'duration_s':float(z['time'][-1]),'frames':frame_count,'endpoint':endpoint,'wall_seconds':time.time()-start,'video_error':video_error,'render_flushes_per_frame':3,'max_render_body_position_error_m':max(render_pose_errors),'max_fk_position_error_m':max(r['max_link_position_error_m'] for r in fk_audit),'max_fk_orientation_error_rad':max(r['max_link_orientation_error_rad'] for r in fk_audit),'physics_backend':'PhysX CPU; RTX GPU rendering','physical_task_success':('not certified; frozen Phase U policy cross-engine diagnostic' if a.mode=='policy' else 'not evaluated; no balance or learned policy'),'root_displacement_m':(z['base_position'][-1]-z['base_position'][0]).tolist(),'joint_range_observed_rad':np.ptp(z['q'],axis=0).tolist(),'limitations':['explicit USD geometry, tessellated convex ellipsoid wheels; isotropic contact approximation not dynamically equivalent','implicit PhysX servo integration differs from MuJoCo RK4; source gains/target/force caps retained','fixed mode constrains root; free mode has no balance policy; policy mode uses frozen actor with finite-difference accelerometer adapter']}
(a.output/'kinematics_audit.json').write_text(json.dumps(fk_audit,indent=2))
(a.output/'result.json').write_text(json.dumps(result,indent=2));print('SMOKE_COMPLETE',json.dumps(result),flush=True)
app.close()
