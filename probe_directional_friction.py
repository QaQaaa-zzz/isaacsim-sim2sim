"""Installed native contact capability probe. No training or model replacement."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
spec=json.loads((ROOT/'configs/isaac_training.json').read_text())
(a.output/'requested_training.json').write_text(json.dumps(spec,indent=2))
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import omni.usd
from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation,SingleRigidPrim
from omni.physx import get_physx_simulation_interface
from usd_model import build

world=World(physics_dt=.001,rendering_dt=.02);world.get_physics_context().enable_fabric(False)
stage=omni.usd.get_context().get_stage()
m,conversion=build(stage,ROOT/'model/source.xml',target_drive_gains=spec['target_drive_gains'])
robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));world.reset()
view=robot._articulation_view;kp,kd=view.get_gains();gains={}
for i,name in enumerate(robot.dof_names):
 gains[name]={'kp':float(kp[0,i]),'kd':float(kd[0,i])}
 if name in spec['target_drive_gains']:
  expected=spec['target_drive_gains'][name]
  np.testing.assert_allclose([kp[0,i],kd[0,i]],[expected['kp'],expected['kd']],rtol=1e-6,atol=1e-8)
stage.GetRootLayer().Export(str(a.output/'candidate_pd_scene.usda'))
api_report={'requested_gains':spec['target_drive_gains'],'runtime_gains':gains,
 'runtime_masses':dict(zip(view.body_names,map(float,view._physics_view.get_masses()[0]))),
 'usd_material_attributes':list(UsdPhysics.MaterialAPI.GetSchemaAttributeNames()),
 'physx_material_attributes':list(PhysxSchema.PhysxMaterialAPI.GetSchemaAttributeNames()),
 'contact_interface_methods':[name for name in dir(get_physx_simulation_interface()) if 'contact' in name.lower()],
 'source_model_gains_unchanged':bool(np.allclose(m.actuator_gainprm[:,0],[10,5,100,100])),
 'friction_matches_source':False,'native_training_launched':False}
(a.output/'capabilities.json').write_text(json.dumps(api_report,indent=2))
world.stop()

# A diagnostic rigid sliding block, not the full vehicle or wheel contact benchmark.
# Huge prescribed inertia suppresses rotation equally in both engines.
xml='''<mujoco><option gravity="0 0 -9.81" timestep="0.001" integrator="implicitfast" cone="elliptic" iterations="100" tolerance="1e-12"/>
<worldbody><geom name="floor" type="plane" size="10 10 .1"/>
<body name="block" pos="0 0 .05"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1000000 1000000 1000000"/>
<geom name="block_collision" type="box" size=".05 .05 .05"/></body></worldbody>
<contact><pair geom1="floor" geom2="block_collision" condim="6" friction="5 .5 .0001 .0001 .0001" solref=".008 1" solimp=".99 .99 .001 .5 2"/></contact></mujoco>'''
(a.output/'fixture.xml').write_text(xml)
rows=[]
for axis in [0,1]:
 mm=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(mm)
 for _ in range(300):mujoco.mj_step(mm,d)
 d.qvel[:]=0;d.qvel[axis]=2.;mujoco.mj_forward(mm,d)
 contact_frame=d.contact[0].frame.tolist();tr=[np.r_[0.,d.qpos[:3],d.qvel[:3]]]
 for k in range(20):mujoco.mj_step(mm,d);tr.append(np.r_[(k+1)*.001,d.qpos[:3],d.qvel[:3]])
 tr=np.asarray(tr);name=f'mujoco_direction_{axis}';np.savez_compressed(a.output/(name+'.npz'),trace=tr)
 rows.append(dict(name=name,engine='MuJoCo CPU contact fixture',axis=axis,mean_deceleration_m_s2=float((2-tr[-1,4+axis])/.02),source_contact_frame=contact_frame))

for static,dynamic in [(5.,5.),(5.,.5)]:
 for axis in [0,1]:
  omni.usd.get_context().new_stage();World.clear_instance();world=World(physics_dt=.001,rendering_dt=.02);world.get_physics_context().enable_fabric(False);stage=omni.usd.get_context().get_stage()
  UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
  mat=UsdShade.Material.Define(stage,'/World/Material');api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());api.CreateStaticFrictionAttr(static);api.CreateDynamicFrictionAttr(dynamic);api.CreateRestitutionAttr(0.)
  PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim()).CreateFrictionCombineModeAttr('max')
  floor=UsdGeom.Plane.Define(stage,'/World/Floor');floor.CreateAxisAttr('Z');floor.CreateWidthAttr(100);floor.CreateLengthAttr(100)
  block=UsdGeom.Cube.Define(stage,'/World/Block');block.CreateSizeAttr(.1);UsdGeom.Xformable(block).AddTranslateOp().Set(Gf.Vec3d(0,0,.05))
  for obj in [floor,block]:
   prim=obj.GetPrim();UsdPhysics.CollisionAPI.Apply(prim);coll=PhysxSchema.PhysxCollisionAPI.Apply(prim);coll.CreateContactOffsetAttr(.001);coll.CreateRestOffsetAttr(0.)
   UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
  prim=block.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim);mass=UsdPhysics.MassAPI.Apply(prim);mass.CreateMassAttr(1.);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e6,1e6,1e6));mass.CreateCenterOfMassAttr(Gf.Vec3f(0,0,0))
  rigid=PhysxSchema.PhysxRigidBodyAPI.Apply(prim);rigid.CreateLinearDampingAttr(0.);rigid.CreateAngularDampingAttr(0.)
  obj=world.scene.add(SingleRigidPrim('/World/Block',name='block'));world.reset()
  for _ in range(300):world.step(render=False)
  v=np.zeros(3);v[axis]=2;obj.set_linear_velocity(v);obj.set_angular_velocity(np.zeros(3));tr=[np.r_[0.,obj.get_world_pose()[0],obj.get_linear_velocity()]]
  for k in range(20):world.step(render=False);tr.append(np.r_[(k+1)*.001,obj.get_world_pose()[0],obj.get_linear_velocity()])
  tr=np.asarray(tr);name=f'physx_static_{static:g}_dynamic_{dynamic:g}_direction_{axis}';np.savez_compressed(a.output/(name+'.npz'),trace=tr)
  rows.append(dict(name=name,engine='PhysX native rigid contact',axis=axis,static=static,dynamic=dynamic,mean_deceleration_m_s2=float((2-tr[-1,4+axis])/.02)))
  world.stop()
for row in rows:assert np.isfinite(row['mean_deceleration_m_s2'])
(a.output/'sliding_results.json').write_text(json.dumps(rows,indent=2))
(a.output/'status.json').write_text(json.dumps(dict(completed=True,diagnostic_cases=6,pd_applied_and_verified=True,source_friction_implemented=False,training_transitions=0,best_model=None),indent=2))
print('PROBE_COMPLETE',json.dumps(rows),flush=True)
app.close()
