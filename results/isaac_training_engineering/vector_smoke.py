import sys,os,json,time
from pathlib import Path
root=Path('/home/qy/ISAAC——SIM');sys.path.insert(0,str(root));os.environ['OMNI_KIT_ACCEPT_EULA']='YES';os.environ['JAX_PLATFORMS']='cpu'
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'extra_args':['--/app/settings/loadUserConfig=false','--/app/settings/persistent=false']})
import numpy as np
from physx_task import PhysxTask
from policy_runtime import Actor
spec=json.loads((root/'configs/isaac_training.json').read_text());env=PhysxTask(spec,8,123);actor=Actor(root/'policy/actor.npz');out=root/'results/isaac_training_engineering'
(out/'runtime_audit.json').write_text(json.dumps(env.audit,indent=2));obs=env.obs.copy();t0=time.time();steps=0;episodes=[];first=[]
for k in range(100):
 actions=actor(obs);obs,r,done,info=env.step(actions);steps+=env.n
 if k<5:first.append(dict(action=actions[0].tolist(),info=info[0],trace=env.trace[0].tolist()))
 for i in np.flatnonzero(done):episodes.append(info[i])
 env.reset(np.flatnonzero(done));obs=env.obs.copy()
 if k%20==19:print('SMOKE',k+1,'rate',steps/(time.time()-t0),flush=True)
(out/'vector_smoke.json').write_text(json.dumps(dict(completed=True,transitions=steps,wall_s=time.time()-t0,rate=steps/(time.time()-t0),episodes=episodes,first_steps=first),indent=2));print('COMPLETE',steps,time.time()-t0,flush=True);app.close()
