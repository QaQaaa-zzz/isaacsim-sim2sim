"""Same frozen NumPy actor in native MuJoCo, engineering reference not MJX equivalence."""
from pathlib import Path
import numpy as np,json,mujoco
from policy_runtime import Actor,Observation,control
root=Path(__file__).resolve().parent;cfg=json.loads((root/'policy/resolved_config.json').read_text());m=mujoco.MjModel.from_xml_path(str(root/'model/source.xml'));m.opt.timestep=.005;d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);d.qvel[0]=2.;d.ctrl[:]=control(np.zeros(4),m,cfg);mujoco.mj_forward(m,d)
a=Actor(root/'policy/actor.npz');o=Observation(m);obs=o.initial(d.qpos[0]);names=[m.joint(i).name for i in range(1,m.njnt)];rows=[];obsrows=[]
for i in range(400):
 act=a(obs);d.ctrl[:]=control(act,m,cfg);obsrows.append(obs.copy())
 for _ in range(4):mujoco.mj_step(m,d)
 q=np.array([d.qpos[m.jnt_qposadr[m.joint(n).id]] for n in names]);qd=np.array([d.qvel[m.jnt_dofadr[m.joint(n).id]] for n in names]);gyro=d.sensor('gyro_local').data.copy();acc=d.sensor('acc_local').data.copy()
 from scipy.spatial.transform import Rotation
 r=Rotation.from_quat(d.qpos[[4,5,6,3]]);obs=o.advance(d.qpos[:3],d.qpos[3:7],q,qd,names,d.qvel[:3],r.apply(gyro),r.apply(acc)+[0,0,-9.81],act)
 rows.append((d.time,d.qpos[:3].copy(),d.qpos[3:7].copy(),act.copy(),q,qd))
p=root/'results/mujoco_checkpoint_reference';p.mkdir(exist_ok=True)
np.savez_compressed(p/'trajectory.npz',**{n:np.array([v[k] for v in rows]) for k,n in enumerate(['time','base_position','base_quaternion','actions','q','qd'])},observations=np.array(obsrows));print('max x/z',np.max([v[1] for v in rows],axis=0));print('final',rows[-1][:3])
