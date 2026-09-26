"""Read-only original MJX-Warp runtime baseline. Writes only to Isaac workspace."""
import os,sys,json,argparse
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['JAX_COMPILATION_CACHE_DIR']='/home/qy/ISAAC——SIM/cache/jax';sys.dont_write_bytecode=True
sys.path.insert(0,'/home/qy/DVGC/JIT/src')
import jax,numpy as np
from jit_dvgc.config import load_config
from jit_dvgc.env import TwoPhaseBikeEnv
from policy_runtime import Actor
root=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('--actor',type=Path,default=root/'policy/actor.npz');parser.add_argument('--output',type=Path,default=root/'results/domain_audit/playground_reference');args=parser.parse_args();c=load_config(root/'policy/resolved_config.json',runtime_only=True);env=TwoPhaseBikeEnv(c);reset=jax.jit(env.reset);step=jax.jit(env.step);a=Actor(args.actor);s=reset(jax.random.PRNGKey(0));jax.block_until_ready(s);rows=[]
print('RESET',np.array(s.data.qpos),s.info['reset_source_airborne_rsi'],flush=True)
assert not bool(s.info['reset_source_airborne_rsi'])
for i in range(150):
 action=a(np.array(s.obs['state']));s=step(s,action);jax.block_until_ready(s);rows.append(np.r_[(i+1)*.02,np.array(s.data.qpos),np.array(s.data.qvel),np.array(s.obs['state']),float(s.reward),int(s.info['end_code']),int(s.info['events'].apex_seen)])
 if i%25==24:print('STEP',i+1,np.array(s.data.qpos[:3]),flush=True)
 if bool(s.done):break
out=args.output;out.mkdir(exist_ok=False);np.savez_compressed(out/'trace.npz',trace=rows);(out/'result.json').write_text(json.dumps({'engine':'original JIT MJX-Warp','actor':str(args.actor),'steps':len(rows),'max_x':float(np.max(np.array(rows)[:,1])),'max_z':float(np.max(np.array(rows)[:,3])),'apex_seen':bool(s.info['events'].apex_seen),'end_code':int(s.info['end_code'])},indent=2))
