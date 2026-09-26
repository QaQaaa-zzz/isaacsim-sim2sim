import os,sys,json
from pathlib import Path
root=Path('/home/qy/ISAAC——SIM');sys.path.insert(0,str(root));os.environ['OMNI_KIT_ACCEPT_EULA']='YES';os.environ['JAX_PLATFORMS']='cpu'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4})
import numpy as np
from physx_task import PhysxTask
from policy_runtime import Actor
s=json.loads((root/'configs/isaac_training.json').read_text());env=PhysxTask(s,8,123);actor=Actor(root/'policy/actor.npz');traces=[]
for repeat in range(2):
 env.hard_reset();env.reset(np.arange(8),seeds=s['best_selection']['seeds']);rows=[];infos=[]
 for k in range(25):
  _,r,done,info=env.step(actor(env.obs));rows.append(env.trace.copy());infos.append(info)
 traces.append(np.asarray(rows))
 if repeat==0:(root/'results/isaac_training_engineering/reward_fixtures.json').write_text(json.dumps(infos))
err=float(np.max(np.abs(traces[0]-traces[1])));np.savez_compressed(root/'results/isaac_training_engineering/reset_probe.npz',first=traces[0],second=traces[1]);(root/'results/isaac_training_engineering/reset_probe.json').write_text(json.dumps(dict(max_abs_trace_difference=err,repeated_steps=25),indent=2));print('RESET_ERROR',err,flush=True);app.close()
