"""Bounded runtime integrity/resource probe for the source PPO environment."""
import argparse,json,os,sys,time,resource
from pathlib import Path
os.environ.update(JAX_PLATFORMS='cpu',OMNI_KIT_ACCEPT_EULA='YES',OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num-envs',type=int,default=384);p.add_argument('--steps',type=int,default=8);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
try:
 import numpy as np
 from physx_task import PhysxTask
 from policy_runtime import Actor
 spec=json.loads(a.config.read_text());print('BUILD_START',a.num_envs,flush=True)
 env=PhysxTask(spec,a.num_envs,spec['ppo']['seed']);actor=Actor(Path(__file__).resolve().parent/'policy/actor.npz');init_s=time.monotonic()-started
 print('BUILD_COMPLETE',init_s,flush=True)
 rows=[];t0=time.monotonic()
 for k in range(a.steps):
  _,_,done,info=env.step(actor(env.obs));rows.append(info);assert env.privileged_obs.shape==(a.num_envs,106) and np.isfinite(env.privileged_obs).all();env.reset(np.flatnonzero(done));print('STEP',k+1,flush=True)
 wall=time.monotonic()-t0
 (a.output/'runtime_audit.json').write_text(json.dumps(env.audit,indent=2))
 (a.output/'reward_fixtures.json').write_text(json.dumps(rows[:2]))
 result=dict(passed=True,num_envs=a.num_envs,steps=a.steps,transitions=a.steps*a.num_envs,initialization_s=init_s,sampling_s=wall,transitions_per_s=a.steps*a.num_envs/wall,max_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,all_finite=True,critic_dimensions=106)
 (a.output/'status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:app.close()
