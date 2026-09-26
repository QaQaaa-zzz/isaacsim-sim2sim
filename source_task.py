"""CPU MuJoCo environment using unchanged JIT reward/event functions read-only.
This is a new engineering training backend, not continuation of original PPO state.

Angular sampling defaults to the historical CPU native sensors. CPU and MJX-Warp
native_sensor sampling phases are not interchangeable. Explicit end_state uses
the final free-joint body angular velocity and final orientation for Actor and
reward, matching the PhysX bridge's sampling contract for future paired work;
it does not establish compatibility of a frozen native-sensor Actor.
"""
import os,sys,json
sys.dont_write_bytecode=True;os.environ.setdefault('JAX_PLATFORMS','cpu');sys.path.insert(0,'/home/qy/DVGC/JIT/src')
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from jit_dvgc.config import load_config
from jit_dvgc import rewards,semantics
# Same original functions, NumPy evaluation. Cross-backend parity tested separately.
rewards.jp=np;semantics.jp=np
from policy_runtime import Observation,control
from domains import apply_domain,identity_domain,ActionDelay,noisy_observation
ROOT=Path(__file__).resolve().parent
CFG=load_config(ROOT/'policy/resolved_config.json',runtime_only=True)
import hashlib
assert hashlib.sha256((ROOT/'model/source.xml').read_bytes()).hexdigest()==CFG.model['xml_sha256'], 'source model identity mismatch'
RAW=json.loads((ROOT/'policy/resolved_config.json').read_text())

class SourceTask:
 def __init__(self):self.cfg=CFG
 def reset(self,domain=None,seed=0):
  self.domain=identity_domain() if domain is None else domain
  self.angular_velocity_mode=self.domain.get('angular_velocity_mode','native_sensor')
  if self.angular_velocity_mode not in ('native_sensor','end_state'):raise ValueError(f'Unknown angular_velocity_mode: {self.angular_velocity_mode!r}')
  self.rng=np.random.default_rng(seed);self.m=mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'));apply_domain(self.m,self.domain);self.d=mujoco.MjData(self.m);mujoco.mj_resetDataKeyframe(self.m,self.d,0)
  # Rebuild derived inertia constants after density/COM changes, then restore reset.
  mujoco.mj_setConst(self.m,self.d);mujoco.mj_resetDataKeyframe(self.m,self.d,0)
  self.d.qvel[0]=CFG.reset.initial_forward_velocity;self.d.ctrl[:]=control(np.zeros(4),self.m,RAW);mujoco.mj_forward(self.m,self.d)
  self.names=[self.m.joint(j).name for j in range(1,self.m.njnt)];self.qa=[self.m.jnt_qposadr[self.m.joint(n).id] for n in self.names];self.va=[self.m.jnt_dofadr[self.m.joint(n).id] for n in self.names]
  self.observer=Observation(self.m);self.events=semantics.initial_event_state(np.asarray(self.d.qpos[0]),CFG);self.last_action=np.zeros(4);self.episode_return=0.;self.delay=ActionDelay(self.domain['delay_control_steps'],np.zeros(4));self.steps=0
  return self.observer.initial(self.d.qpos[0])
 def step(self,action):
  action=np.asarray(action,np.float32);self.d.ctrl[:]=control(self.delay(action),self.m,RAW)
  for _ in range(round(.02/self.m.opt.timestep)):
   previous_velocity=self.d.qvel[:3].copy();mujoco.mj_step(self.m,self.d)
  acceleration=(self.d.qvel[:3]-previous_velocity)/self.m.opt.timestep if self.domain.get('accelerometer_mode')=='finite_difference' else None
  d=self.d;r=Rotation.from_quat(d.qpos[[4,5,6,3]]);roll,pitch,yaw=r.as_euler('xyz')
  omega=r.apply(d.qvel[3:6]) if self.angular_velocity_mode=='end_state' else d.sensor('ang_global').data.copy()
  actor_omega=omega if self.angular_velocity_mode=='end_state' else r.apply(d.sensor('gyro_local').data)
  previous=self.events;inputs=semantics.TerminalInputs(previous.episode_step,np.asarray(not np.isfinite(np.r_[d.qpos,d.qvel,action]).all()),roll,pitch,np.asarray(False),np.asarray(d.qpos[0]<self.m.key_qpos[0,0]-CFG.physical_limits.max_backward_distance),previous.stuck,yaw,previous.jump_zone_seen);pre=semantics.classify_terminal(inputs,CFG)
  self.events=semantics.advance_events(previous,semantics.PhaseUSignals(d.qpos[0],d.qpos[2],d.qvel[0],d.qvel[2],pre.terminated),CFG);terminal=semantics.classify_terminal(inputs.replace(stuck=self.events.stuck,jump_zone_seen=self.events.jump_zone_seen),CFG)
  state=rewards.RewardState(*d.qpos[:3],roll,pitch,yaw,*d.qvel[:3],*omega,d.qvel[self.m.jnt_dofadr[self.m.joint('hip_joint').id]],d.qvel[self.m.jnt_dofadr[self.m.joint('knee_joint').id]],*d.actuator_force[2:4])
  ri=rewards.RewardInputs(state,action,self.last_action,self.events.jump_signal,self.events.apex_seen & ~previous.apex_seen,np.asarray(False),terminal.physical_failure,terminal.roll_limit|terminal.pitch_limit,terminal.jump_zone_missed,terminal.stuck,terminal.yaw_limit,terminal.timeout)
  rr=rewards.phase_u_reward(ri,CFG.reward,CFG.physical_limits);reward=float(rr.total)
  if terminal.stuck or terminal.yaw_limit:reward=CFG.reward.failed_episode_return-self.episode_return
  self.episode_return+=reward;self.steps+=1
  obs=self.observer.advance(d.qpos[:3],d.qpos[3:7],d.qpos[self.qa],d.qvel[self.va],self.names,d.qvel[:3],actor_omega,(r.apply(d.sensor('acc_local').data)+[0,0,-9.81]) if acceleration is None else acceleration,action)
  self.last_action=action.copy();info=dict(x=float(d.qpos[0]),z=float(d.qpos[2]),vx=float(d.qvel[0]),roll=float(roll),pitch=float(pitch),yaw=float(yaw),apex=bool(self.events.apex_seen),zone_seen=bool(self.events.jump_zone_seen),end_code=int(terminal.end_code),return_=self.episode_return,components={k:float(v) for k,v in rr.components.as_dict().items()})
  if self.angular_velocity_mode=='end_state':info['angular_velocity_mode']='end_state'
  return noisy_observation(obs,self.domain,self.rng),reward,bool(terminal.terminated|terminal.truncated),info
