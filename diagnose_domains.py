"""Bounded same-control/native-policy cross-engine diagnostics. No training."""
import argparse,os,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();spec=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=True)
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,mujoco,omni.usd
from pxr import UsdPhysics,PhysxSchema,UsdGeom,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation
from policy_runtime import Actor,Observation,control
from usd_model import build
from domains import ActionDelay,noisy_observation
from endpoint import Endpoint
root=Path(__file__).resolve().parent;cfg=json.loads((root/'policy/resolved_config.json').read_text());actor=Actor(root/'policy/actor.npz');summary=[]
for case in spec['cases']:
 out=a.output/case['name'];out.mkdir(exist_ok=True);dt=case.get('dt',.005);duration=case.get('duration',2.);assert duration<=4;stride=round(.02/dt);assert np.isclose(stride*dt,.02)
 omni.usd.get_context().new_stage();World.clear_instance();world=World(physics_dt=dt,rendering_dt=.02);world.get_physics_context().enable_fabric(False);stage=omni.usd.get_context().get_stage()
 m,conv=build(stage,root/'model/source.xml',fixed=case.get('fixed',False),wheel_sectors=case.get('converter_sectors',16),domain=case.get('domain'));m.opt.timestep=case.get('source_dt',.005);target_actor=Actor(root/case.get('actor','policy/actor.npz'));source_actor=Actor(root/case.get('source_actor','policy/actor.npz'))
 if case.get('no_contact',False):
  m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  for prim in stage.Traverse():
   if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
 for prim in stage.Traverse():
  if prim.HasAPI(UsdPhysics.MaterialAPI) and 'friction' in case:
   api=UsdPhysics.MaterialAPI(prim);api.CreateStaticFrictionAttr(case['friction']);api.CreateDynamicFrictionAttr(case['friction'])
  if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
   PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim).CreateHullVertexLimitAttr(case.get('hull_vertices',64))
   scale=case.get('mesh_normalization',1.)
   if scale!=1:
    mesh=UsdGeom.Mesh(prim);mesh.GetPointsAttr().Set((np.array(mesh.GetPointsAttr().Get())*scale).tolist());UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(1/scale))
 if case.get('source_isotropic',False):m.pair_friction[:,:2]=case['friction'];m.geom_friction[:,0]=case['friction']
 if case.get('wheel_sectors'):
  from collision_geometry import ellipsoid_sectors
  from usd_model import mesh
  from pxr import UsdShade,Sdf
  for prim in list(stage.Traverse()):
   if 'wheel_collision' not in prim.GetName():continue
   original_mesh=UsdGeom.Mesh(prim);shapes=ellipsoid_sectors([.07,.07,.025],case['wheel_sectors']);verts,faces=shapes[0];original_mesh.GetPointsAttr().Set(verts.tolist());original_mesh.GetFaceVertexCountsAttr().Set([3]*len(faces));original_mesh.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
   for si,(verts,faces) in enumerate(shapes[1:],1):
    obj=mesh(stage,str(prim.GetPath())+f'_sector{si}',verts,faces);new=obj.GetPrim()
    for attr in prim.GetAttributes():
     if attr.GetName().startswith(('xform','physx','physics')) and attr.Get() is not None:new.CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(attr.Get())
    for schema in prim.GetAppliedSchemas():new.AddAppliedSchema(schema)
    for rel in prim.GetRelationships():new.CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
 robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));world.reset();names=robot.dof_names;view=robot._articulation_view;pv=view._physics_view
 if case.get('target_drive_scales'):
  from drive_calibration import scaled_gains
  kp,kd=scaled_gains(names,*view.get_gains(),case['target_drive_scales']);view.set_gains(kps=kp,kds=kd)
  (out/'target_drive_calibration.json').write_text(json.dumps({'scales':case['target_drive_scales'],'runtime_gains':[v.tolist() for v in view.get_gains()],'runtime_max_forces':view.get_max_efforts().tolist(),'source_gains_unchanged':True},indent=2))
 from omni.physx import get_physx_cooking_interface
 from pxr import PhysicsSchemaTools
 cooked=[];union_vertices=[]
 for prim in stage.Traverse():
  if 'wheel_collision' not in prim.GetName():continue
  received=[]
  get_physx_cooking_interface().request_convex_collision_representation(stage_id=omni.usd.get_context().get_stage_id(),collision_prim_id=PhysicsSchemaTools.sdfPathToInt(prim.GetPath()),run_asynchronously=False,on_result=lambda status,hulls:received.extend(hulls))
  for hull in received:
   vv=np.array([list(x) for x in hull.vertices])/case.get('mesh_normalization',1.);union_vertices.extend(vv);rng=np.random.default_rng(0);directions=rng.normal(size=(10000,3));directions/=np.linalg.norm(directions,axis=1)[:,None];expected=np.linalg.norm(directions*np.array([.07,.07,.025]),axis=1);actual=np.max(directions@vv.T,axis=1);cooked.append({'prim':str(prim.GetPath()),'vertices':len(vv),'max_abs_support_error_m':float(np.max(np.abs(actual-expected))),'max_rel_support_error':float(np.max(np.abs(actual-expected)/expected)),'bounds':np.ptp(vv,axis=0).tolist()})
 if union_vertices:
  actual=np.max(directions@np.asarray(union_vertices).T,axis=1);cooked.append({'union_max_abs_support_error_m':float(np.max(np.abs(actual-expected))),'union_max_rel_support_error':float(np.max(np.abs(actual-expected)/expected))})
 (out/'cooked_wheels.json').write_text(json.dumps(cooked,indent=2))
 q0=np.array([m.key_qpos[0,m.jnt_qposadr[m.joint(n).id]] for n in names],np.float32);qa=np.array([m.jnt_qposadr[m.joint(n).id] for n in names]);va=np.array([m.jnt_dofadr[m.joint(n).id] for n in names]);ai=np.array([names.index(m.joint(j).name) for j in m.actuator_trnid[:,0]])
 robot.set_joint_positions(q0);robot.set_joint_velocities(np.zeros(5));robot.set_world_pose(m.key_qpos[0,:3],m.key_qpos[0,3:7]);robot.set_linear_velocity(np.array([2.,0,0]));robot.set_angular_velocity(np.zeros(3));prev=np.array([2.,0,0])
 if case.get('drive','implicit')=='explicit':view.set_gains(kps=np.zeros((1,5)),kds=np.zeros((1,5)))
 d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);d.qvel[0]=2.;d.ctrl[:]=control(np.zeros(4),m,cfg);mujoco.mj_forward(m,d)
 obsp=Observation(m);obsmp=Observation(m);op=obsp.initial(d.qpos[0]);om=obsmp.initial(d.qpos[0]);trp=[];trm=[];actor_rows=[];contact_rows=[];t0=time.time();source_failed=False;target_failed=False;source_monitor=Endpoint();target_monitor=Endpoint();ds=case.get('domain');delayer=ActionDelay(ds['delay_control_steps'] if ds else 0,np.zeros(4));rng=np.random.default_rng(ds.get('seed',0) if ds else 0);target_status={};source_status={}
 def row(t,pos,quat,q,qd,vel,omega,acc,ctrl):return np.r_[t,pos,quat,q,qd,vel,omega,acc,ctrl]
 for k in range(round(duration/.02)):
  actm=source_actor(om);actp=target_actor(noisy_observation(op,ds,rng) if ds else op)
  cm=control(actm,m,cfg);cp=control(delayer(actp),m,cfg)
  if case.get('mode','policy')=='replay':cp=cm.copy()
  if case.get('mode')=='zero':cm=cp=control(np.zeros(4),m,cfg)
  d.ctrl[:]=cm
  if not source_failed:
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
  rm=Rotation.from_quat(d.qpos[[4,5,6,3]]);om=obsmp.advance(d.qpos[:3],d.qpos[3:7],d.qpos[qa],d.qvel[va],names,d.qvel[:3],rm.apply(d.sensor('gyro_local').data),rm.apply(d.sensor('acc_local').data)+[0,0,-9.81],actm)
  for sub in range(stride):
   if case.get('drive','implicit')=='explicit':
    q=robot.get_joint_positions();qd=robot.get_joint_velocities();tau=np.zeros(5);u=m.actuator_gainprm[:,0]*cp+m.actuator_biasprm[:,1]*q[ai]+m.actuator_biasprm[:,2]*qd[ai];tau[ai]=np.clip(u,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]);robot.set_joint_efforts(tau.astype(np.float32))
   elif sub==0:
    qt=q0.copy();vt=np.zeros(5)
    for act,jidx in enumerate(ai):
     kp=-m.actuator_biasprm[act,1];kd=-m.actuator_biasprm[act,2];gain=m.actuator_gainprm[act,0]
     if kp:qt[jidx]=cp[act]*gain/kp
     else:vt[jidx]=cp[act]*gain/kd
    robot.apply_action(ArticulationAction(joint_positions=qt,joint_velocities=vt))
   if ds:robot.set_joint_efforts((-m.dof_damping[va]*robot.get_joint_velocities()).astype(np.float32))
   world.step(render=False);pos,quat=robot.get_world_pose();q=robot.get_joint_positions();qd=robot.get_joint_velocities();r=Rotation.from_quat(np.r_[quat[1:],quat[0]]);omega=robot.get_angular_velocity();vel=robot.get_linear_velocity()-np.cross(omega,r.apply(m.body_ipos[1]));acc=(vel-prev)/dt;prev=vel.copy()
   if case.get('sensor')=='tensor':
    ac=pv.get_link_accelerations()[0,0];offset=r.apply(m.body_ipos[1]);acc=ac[:3]-np.cross(ac[3:],offset)-np.cross(omega,np.cross(omega,offset))
  op=obsp.advance(pos,quat,q,qd,names,vel,omega,acc,actp)
  trp.append(row((k+1)*.02,pos,quat,q,qd,vel,omega,acc,cp));trm.append(row(d.time,d.qpos[:3],d.qpos[3:7],d.qpos[qa],d.qvel[va],d.qvel[:3],rm.apply(d.sensor('gyro_local').data),rm.apply(d.sensor('acc_local').data)+[0,0,-9.81],cm));actor_rows.append(np.r_[k*.02,om,op,actm,actp])
  if d.ncon and len(contact_rows)<4:
   contact_rows.append([{'geom':[m.geom(int(c.geom[0])).name,m.geom(int(c.geom[1])).name],'frame':c.frame.tolist(),'friction':c.friction.tolist()} for c in d.contact[:d.ncon]])
  if not np.isfinite(trp[-1]).all():break
  source_status=source_monitor.advance(d.qpos[:3],d.qpos[3:7],d.qvel[:3]);target_status=target_monitor.advance(pos,quat,vel)
  if source_status['terminated'] and not source_failed:source_failed=(k+1)*.02
  if target_status['terminated'] and not target_failed:target_failed=(k+1)*.02
  if case.get('stop_on_failure',True) and (target_failed or (source_failed and case.get('stop_on_source_failure',True))):break
 np.savez_compressed(out/'traces.npz',source=trm,target=trp,actor=actor_rows,joint_names=names);(out/'source_contacts.json').write_text(json.dumps(contact_rows,indent=2));(out/'case.json').write_text(json.dumps(case,indent=2))
 trp=np.array(trp);trm=np.array(trm);res={'case':case['name'],'duration':float(trp[-1,0]),'source_first_failure_s':source_failed,'target_first_failure_s':target_failed,'source_max_x_z':np.max(trm[:,[1,3]],axis=0).tolist(),'target_max_x_z':np.max(trp[:,[1,3]],axis=0).tolist(),'position_rmse_m':np.sqrt(np.mean((trp[:,1:4]-trm[:,1:4])**2,axis=0)).tolist(),'joint_velocity_rmse':np.sqrt(np.mean((trp[:,13:18]-trm[:,13:18])**2,axis=0)).tolist(),'wall_s':time.time()-t0,'source_endpoint':source_status,'target_endpoint':target_status,'target_speed_rmse':float(np.sqrt(np.mean((trp[:,18]-2)**2))),'target_screen_qualified':bool(target_status.get('apex_seen') and not target_failed and trp[-1,1]>4.)};summary.append(res);(out/'result.json').write_text(json.dumps(res,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print('CASE_DONE',json.dumps(res),flush=True);world.stop()
app.close()
