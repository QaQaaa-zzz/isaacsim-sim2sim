"""RTX visualization of recorded physical trajectories, explicitly labeled replay."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--cases',nargs='*');p.add_argument('--duration',type=float,default=3.);a=p.parse_args();root=Path(__file__).resolve().parent
os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':640,'height':480,'renderer':'RaytracedLighting','anti_aliasing':0,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np,omni.usd,imageio.v2 as imageio
from pxr import UsdLux,Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.viewports import set_camera_view
from omni.physx import get_physx_interface
from usd_model import build
from scene_visuals import enhance
from PIL import Image,ImageDraw
world=World(physics_dt=.005,rendering_dt=.04);world.get_physics_context().enable_fabric(False);stage=omni.usd.get_context().get_stage();m,_=build(stage,root/'model/source.xml');enhance(stage,m);light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(1100);light.CreateColorAttr(Gf.Vec3f(.9,.93,1));robot=world.scene.add(SingleArticulation('/World/Bike',name='bike'));camera=Camera('/World/Camera',frequency=-1,resolution=(640,480));camera.initialize();camera.set_focal_length(3);world.reset()
set_camera_view(eye=np.array([2.,-1.3,.8]),target=np.array([1.5,0,.3]),camera_prim_path='/World/Camera')
for _ in range(20):world.render()
for folder in sorted(a.input.iterdir()):
 if not(folder/'traces.npz').exists() or (a.cases and folder.name not in a.cases):continue
 trace=np.load(folder/'traces.npz');case=json.loads((folder/'case.json').read_text())
 if case.get('trace_format')=='mjx':
  raw=trace['trace'];qpos=raw[:,1:13];z=np.zeros((len(raw),18));z[:,0]=raw[:,0];z[:,1:4]=qpos[:,:3];z[:,4:8]=qpos[:,3:7];z[:,8:13]=qpos[:,[m.jnt_qposadr[m.joint(n).id] for n in robot.dof_names]];mapping=list(range(5))
 else:z=trace['target'];mapping=[list(trace['joint_names']).index(n) for n in robot.dof_names]
 frames=[];max_error=0.
 with imageio.get_writer(folder/'replay.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16) as writer:
  for t in np.arange(.04,a.duration+.001,.04):
   idx=min(np.searchsorted(z[:,0],t),len(z)-1);row=z[idx];pos=row[1:4];robot.set_world_pose(pos,row[4:8]);robot.set_joint_positions(row[8:13][mapping]);world.physics_sim_view.update_articulations_kinematic();get_physx_interface().update_transformations(False,True,True);set_camera_view(eye=pos+np.array([1.3,-2.6,1.1]),target=pos+np.array([.1,0,.42]),camera_prim_path='/World/Camera')
   # RTX needs additional updates after the first teleport of each case.
   for _ in range(30 if t==.04 else 3):world.render()
   rgba=camera.get_rgba()
   if rgba.ndim!=3:
    app.close();raise RuntimeError('Camera returned empty frame after warmup')
   max_error=max(max_error,float(np.max(np.abs(robot.get_world_pose()[0]-pos))));im=Image.fromarray(rgba[:,:,:3].astype(np.uint8));draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,65),fill=(13,20,29));draw.text((10,5),folder.name+' | '+case.get('recorded_engine','PhysX')+' trajectory replay',fill='white');draw.text((10,24),f't={min(t,z[-1,0]):.2f}s | '+case.get('contact_label',f'PhysX scalar friction={case.get("friction",5):.3g}; source anisotropy 5 / 0.5'),fill='white')
   if t>z[-1,0]+1e-6:draw.text((10,43),f'TERMINATED at {z[-1,0]:.2f}s; final pose held for comparison',fill=(255,120,100))
   else:draw.text((10,43),'Source mass / joint limits / actuator force caps preserved',fill=(160,200,230))
   writer.append_data(np.array(im))
   if idx==len(z)-1 and t<=z[-1,0]+.04:im.save(folder/'endpoint.png')
   if t==.04:im.save(folder/'preview.png')
 (folder/'render_verification.json').write_text(json.dumps({'video_kind':'replay of actual logged '+case.get('recorded_engine','PhysX')+' states, not a new dynamics rollout','physics_steps_during_render':0,'frames':len(np.arange(.04,a.duration+.001,.04)),'max_root_pose_replay_error_m':max_error,'final_state_hold_after_s':float(z[-1,0])},indent=2));print('RENDERED',folder,flush=True)
app.close()
