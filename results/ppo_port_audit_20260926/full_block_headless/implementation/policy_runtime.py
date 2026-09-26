"""Frozen deterministic Brax Actor and source-contract observation adapter."""
import numpy as np
from scipy.special import expit
from scipy.spatial.transform import Rotation
import mujoco
class Actor:
 def __init__(self,path):self.p=dict(np.load(path))
 def __call__(self,obs):
  p=self.p;x=(np.asarray(obs,dtype=np.float32)-p['mean'])/p['std']
  for i in range(4):
   x=x@p[f'hidden_{i}_kernel']+p[f'hidden_{i}_bias']
   if i<3:x=x*expit(x)
  return np.tanh(x[..., :4])
def control(action,m,cfg):
 a=np.clip(action,-1,1);out=np.zeros(4);out[0]=a[0]*m.actuator_ctrlrange[0,1];out[1]=cfg['action']['base_rear_speed']+a[1]*cfg['action']['rear_speed_delta']
 for i in [2,3]:
  j=m.actuator_trnid[i,0];initial=m.key_qpos[0,m.jnt_qposadr[j]];lo,hi=m.actuator_ctrlrange[i];out[i]=initial+a[i]*((hi-initial) if a[i]>=0 else (initial-lo))
 return np.clip(out,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
class Observation:
 def __init__(self,m):self.m=m;self.d=mujoco.MjData(m);self.frames=np.zeros((3,25),np.float32);self.count=0;self.signal=False;self.consumed=False
 def initial(self,x):self.signal=2.5<=x<=4.;self.consumed=x>4.;return np.r_[self.frames.ravel(),float(self.signal)].astype(np.float32)
 def privileged(self,actor_obs,vel,omega,qd,names):
  """Source 106D critic contract; qvel free angular coordinates are body-local."""
  m=self.m;d=self.d;r=Rotation.from_quat(d.qpos[[4,5,6,3]])
  d.qvel[:3]=vel;d.qvel[3:6]=r.inv().apply(omega)
  for i,n in enumerate(names):d.qvel[m.jnt_dofadr[m.joint(n).id]]=qd[i]
  prohibited=False;step=m.geom('step');left=step.pos[0]-step.size[0];right=step.pos[0]+step.size[0];top=step.pos[2]+step.size[2]
  for g in range(m.ngeom):
   if m.geom_bodyid[g]==0 or not(m.geom_contype[g] or m.geom_conaffinity[g]) or 'wheel' in m.geom(g).name:continue
   mat=d.geom_xmat[g].reshape(3,3);size=m.geom_size[g];typ=m.geom_type[g]
   if typ==mujoco.mjtGeom.mjGEOM_BOX:support=np.sum(np.abs(mat)*size,axis=1)
   elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:support=np.linalg.norm(mat*size,axis=1)
   elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:support=size[1]*np.abs(mat[:,2])+size[0]*np.linalg.norm(mat[:,:2],axis=1)
   else:raise ValueError(typ)
   lower=d.geom_xpos[g]-support;upper=d.geom_xpos[g]+support
   overlap=upper[0]>=left and lower[0]<=right and upper[1]>=-step.size[1] and lower[1]<=step.size[1]
   prohibited|=lower[2]-(top if overlap else 0)<-.002
  return np.r_[actor_obs,d.qpos,d.qvel,r.as_euler('xyz'),vel,float(prohibited)].astype(np.float32)
 def advance(self,pos,quat,q,qd,names,vel,omega,acceleration,last_action):
  m=self.m;d=self.d;d.qpos[:3]=pos;d.qpos[3:7]=quat
  for i,n in enumerate(names):d.qpos[m.jnt_qposadr[m.joint(n).id]]=q[i]
  mujoco.mj_kinematics(m,d)
  front=-np.inf
  for g in range(m.ngeom):
   if m.geom_bodyid[g]==0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
   direction=d.geom_xmat[g].reshape(3,3)[0];sz=m.geom_size[g];typ=m.geom_type[g]
   if typ==mujoco.mjtGeom.mjGEOM_BOX:support=np.sum(sz*np.abs(direction))
   elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:support=np.linalg.norm(sz*direction)
   elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:support=sz[1]*abs(direction[2])+sz[0]*np.linalg.norm(direction[:2])
   else:raise ValueError(typ)
   front=max(front,d.geom_xpos[g,0]+support)
  r=Rotation.from_quat(np.r_[quat[1:],quat[0]]);ids=[names.index(n) for n in ['steering_joint','hip_joint','knee_joint']];wids=[names.index(n) for n in ['frontwheel_joint','rearwheel_joint']]
  frame=np.r_[r.inv().apply([0,0,-1]),r.inv().apply(omega),r.inv().apply(acceleration-[0,0,-9.81]),q[ids],qd[ids],qd[wids],last_action,vel[0],3.6-front,pos[2],1].astype(np.float32)
  self.frames=np.concatenate([self.frames[1:],frame[None]]);self.count=min(3,self.count+1);self.frames[:,-1]=(np.arange(3)>=3-self.count)
  inside=2.5<=pos[0]<=4.;old=self.signal;self.signal=inside and not self.consumed;self.consumed=self.consumed or (old and not inside) or pos[0]>4
  return np.r_[self.frames.ravel(),self.signal].astype(np.float32)
