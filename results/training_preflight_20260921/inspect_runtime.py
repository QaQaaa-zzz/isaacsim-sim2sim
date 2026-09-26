"""Fresh default-converter readback. No policy rollout or training."""
import os,sys,json,hashlib
from pathlib import Path
root=Path('/home/qy/ISAAC——SIM');sys.path.insert(0,str(root));os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from usd_model import build
world=World(physics_dt=.005,rendering_dt=.02);world.get_physics_context().enable_fabric(False)
stage=omni.usd.get_context().get_stage();m,conversion=build(stage,root/'model/source.xml')
robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));world.reset()
view=robot._articulation_view;pv=view._physics_view;kp,kd=view.get_gains();masses=pv.get_masses()[0]
report={'variant':'default converter, no calibrated gains, no DR','training_started':False,'requested_training_transitions':10000000,'bodies':[],'joints':[],'source_pairs':[],'target_materials':[]}
for i,name in enumerate(view.body_names):
 src=float(m.body_mass[m.body(name).id]);target=float(masses[i]);report['bodies'].append(dict(name=name,source_mass_kg=src,physx_mass_kg=target,matches=bool(np.isclose(src,target,rtol=1e-6,atol=1e-8))))
for i,name in enumerate(robot.dof_names):
 acts=np.flatnonzero(m.actuator_trnid[:,0]==m.joint(name).id)
 p,d=(-m.actuator_biasprm[acts[0],1:3]) if len(acts) else (0.,0.)
 report['joints'].append(dict(name=name,source_kp=float(p),source_kd=float(d),physx_kp=float(kp[0,i]),physx_kd=float(kd[0,i]),matches=bool(np.allclose([p,d],[kp[0,i],kd[0,i]],rtol=1e-6,atol=1e-8))))
for i in range(m.npair):report['source_pairs'].append(dict(geoms=[m.geom(m.pair_geom1[i]).name,m.geom(m.pair_geom2[i]).name],friction=m.pair_friction[i].tolist()))
for prim in stage.Traverse():
 if prim.HasAPI(UsdPhysics.MaterialAPI):
  mat=UsdPhysics.MaterialAPI(prim);report['target_materials'].append(dict(path=str(prim.GetPath()),static=mat.GetStaticFrictionAttr().Get(),dynamic=mat.GetDynamicFrictionAttr().Get()))
report['mass_matches']=all(x['matches'] for x in report['bodies']);report['default_pd_numeric_matches']=all(x['matches'] for x in report['joints']);report['friction_matches']=False
report['friction_reason']='Source wheel-floor has two tangential coefficients [5,0.5]; target uses one isotropic static/dynamic coefficient 5. These are different dimensions of the contact model.'
identity=json.loads((root/'policy/identity.json').read_text());ck=Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928')
checks={str(root/'model/source.xml'):identity['xml_sha256'],str(Path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml')):identity['xml_sha256'],str(ck/'payload.pkl'):identity['payload_sha256'],str(root/'policy/resolved_config.json'):hashlib.sha256((ck.parent.parent/'resolved_config.json').read_bytes()).hexdigest()}
report['sha256']={}
for path,wanted in checks.items():
 actual=hashlib.sha256(Path(path).read_bytes()).hexdigest();assert actual==wanted,(path,actual,wanted);report['sha256'][path]=actual
report['config_canonical_sha256']=hashlib.sha256(json.dumps(json.loads((root/'policy/resolved_config.json').read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert report['config_canonical_sha256']==json.loads((root/'policy/identity.json').read_text())['config_sha256']
report['conditional_launch_gate_passed']=report['mass_matches'] and report['default_pd_numeric_matches'] and report['friction_matches']
(root/'results/training_preflight_20260921/runtime.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['mass_matches','default_pd_numeric_matches','friction_matches','conditional_launch_gate_passed','training_started']},indent=2),flush=True)
app.close()
