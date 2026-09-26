import argparse,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
r=json.loads((a.directory/'result.json').read_text());z=np.load(a.directory/'trajectory.npz')
assert r['engine']=='Isaac Sim / PhysX'
assert len(r['joint_names'])==5 and len(set(r['joint_names']))==5
assert z['time'].size>1 and np.all(np.diff(z['time'])>0)
for key in ['q','qd','base_position','base_quaternion','effort']:
 assert np.isfinite(z[key]).all(),key
assert r['mass_audit_passed']
reader=imageio.get_reader(a.directory/'simulation.mp4')
first=reader.get_data(0);last=reader.get_data(reader.count_frames()-1)
assert first.shape[:2]==(480,640)
assert float(first.std())>8,'blank frame'
assert np.abs(last.astype(float)-first.astype(float)).mean()>.1,'static video'
print(json.dumps({'verified':True,'samples':int(z['time'].size),'frames':reader.count_frames(),'video_frame_std':float(first.std()),'video_first_last_mean_abs_difference':float(np.abs(last.astype(float)-first.astype(float)).mean()),'physical_task_success':r.get('physical_task_success','not evaluated')},indent=2))
