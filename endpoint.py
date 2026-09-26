"""Original Phase U event and termination contract applied to either engine."""
import sys,os
sys.dont_write_bytecode=True;os.environ.setdefault('JAX_PLATFORMS','cpu');sys.path.insert(0,'/home/qy/DVGC/JIT/src')
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from jit_dvgc import semantics
from jit_dvgc.config import load_config
semantics.jp=np
CFG=load_config(Path(__file__).resolve().parent/'policy/resolved_config.json',runtime_only=True)
class Endpoint:
 def __init__(self,x=1.5):self.initial_x=x;self.events=semantics.initial_event_state(np.asarray(x),CFG)
 def advance(self,pos,quat,velocity):
  angles=Rotation.from_quat(np.r_[quat[1:],quat[0]]).as_euler('xyz');e=self.events;inputs=semantics.TerminalInputs(e.episode_step,np.asarray(not np.isfinite(np.r_[pos,quat,velocity]).all()),angles[0],angles[1],np.asarray(False),np.asarray(pos[0]<self.initial_x-CFG.physical_limits.max_backward_distance),e.stuck,angles[2],e.jump_zone_seen);pre=semantics.classify_terminal(inputs,CFG)
  self.events=semantics.advance_events(e,semantics.PhaseUSignals(pos[0],pos[2],velocity[0],velocity[2],pre.terminated),CFG);t=semantics.classify_terminal(inputs.replace(stuck=self.events.stuck,jump_zone_seen=self.events.jump_zone_seen),CFG)
  return {'terminated':bool(t.terminated),'truncated':bool(t.truncated),'apex_seen':bool(self.events.apex_seen),'jump_zone_seen':bool(self.events.jump_zone_seen),'end_code':int(t.end_code),'roll_limit':bool(t.roll_limit),'pitch_limit':bool(t.pitch_limit),'yaw_limit':bool(t.yaw_limit),'stuck':bool(t.stuck)}
