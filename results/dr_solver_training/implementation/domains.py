"""Declarative reproducible randomization, independent of policy/checkpoint names."""
import numpy as np
from collections import deque

def identity_domain():
 return dict(contact_mode='identity',density_scale=1.,com_shift_m=[0.,0.,0.],kp_scale=1.,kd_scale=1.,force_scale=1.,friction_scale=1.,friction=None,joint_damping=0.,gyro_noise_std=0.,acc_noise_std=0.,delay_control_steps=0,simulation_dt=.005,integrator='RK4',accelerometer_mode='native')

def sample_domain(spec,seed,role):
 rng=np.random.default_rng(np.random.SeedSequence([int(seed),int(spec['seed_namespaces'][role])]))
 names=list(spec['mixture']);mode=rng.choice(names,p=[spec['mixture'][n] for n in names]);d=identity_domain();d.update(seed=int(seed),role=role,contact_mode=str(mode))
 if mode=='identity':return d
 ranges=spec['ranges']
 for key in ['density_scale','kp_scale','kd_scale','force_scale','friction_scale','joint_damping','gyro_noise_std','acc_noise_std']:d[key]=float(rng.uniform(*ranges[key]))
 d['com_shift_m']=rng.uniform(*ranges['com_shift_m'],size=3).tolist();d['friction']=float(np.exp(rng.uniform(*np.log(ranges['isotropic_friction'])))) if mode=='isotropic' else None
 d['delay_control_steps']=int(rng.choice(spec['delay_control_steps']));d['simulation_dt']=float(rng.choice(spec['simulation_dt_choices']));d['integrator']=str(rng.choice(spec.get('integrator_choices',['RK4'])));d['accelerometer_mode']=str(rng.choice(spec.get('accelerometer_mode_choices',['native'])))
 return d

def apply_domain(m,d):
 # Called on a fresh model only, never accumulated across episodes.
 m.body_mass[1:]*=d['density_scale'];m.body_inertia[1:]*=d['density_scale'];m.body_ipos[1:]+=np.array(d['com_shift_m'])
 m.actuator_gainprm[:,0]*=np.where(m.actuator_biasprm[:,1]!=0,d['kp_scale'],d['kd_scale']);m.actuator_biasprm[:,1]*=d['kp_scale'];m.actuator_biasprm[:,2]*=d['kd_scale'];m.actuator_forcerange[:]*=d['force_scale'];m.dof_damping[6:]+=d['joint_damping']
 if d['contact_mode']=='isotropic':m.geom_friction[:,0]=d['friction'];m.pair_friction[:,:2]=d['friction']
 else:m.geom_friction[:,0]*=d['friction_scale'];m.pair_friction[:,:2]*=d['friction_scale']
 m.opt.timestep=d['simulation_dt']
 if d.get('integrator'):
  import mujoco
  m.opt.integrator=getattr(mujoco.mjtIntegrator,'mjINT_'+d['integrator'].upper())

class ActionDelay:
 def __init__(self,steps,initial):self.steps=steps;self.initial=np.asarray(initial).copy();self.reset()
 def reset(self):self.queue=deque(self.initial.copy() for _ in range(self.steps))
 def __call__(self,action):self.queue.append(np.asarray(action).copy());return self.queue.popleft()

def noisy_observation(obs,domain,rng):
 x=obs.copy();frames=x[:75].reshape(3,25);valid=frames[:,-1]>0
 frames[valid,3:6]+=rng.normal(0,domain['gyro_noise_std'],size=(sum(valid),3));frames[valid,6:9]+=rng.normal(0,domain['acc_noise_std'],size=(sum(valid),3));return x
