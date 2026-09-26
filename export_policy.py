import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,hashlib,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/qy/DVGC/JIT/src')
from jit_dvgc.checkpoint import load_checkpoint,CheckpointIdentity
from jit_dvgc.constants import ACTOR_FRAME_FIELDS,ACTOR_TASK_FIELDS,ACTION_ORDER
p=Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928')
i=json.loads((p/'identity.json').read_text());payload=load_checkpoint(p,expected=CheckpointIdentity(i['config_sha256'],i['xml_sha256'],ACTOR_FRAME_FIELDS,ACTOR_TASK_FIELDS,ACTION_ORDER))
print('actor',__import__('jax').tree_util.tree_map(lambda x:x.shape,payload.actor_params));print('norm',payload.observation_normalizer)
root=Path('/home/qy/ISAAC——SIM/policy');(root/'identity.json').write_text(json.dumps(i,indent=2));(root/'resolved_config.json').write_bytes((p.parent.parent/'resolved_config.json').read_bytes())
params=payload.actor_params['params'];arr={}
for name,v in params.items():
 for key,x in v.items():arr[name+'_'+key]=np.asarray(x)
norm=payload.observation_normalizer
arr['mean']=np.asarray(norm.mean['state']);arr['std']=np.asarray(norm.std['state'])
np.savez(root/'actor.npz',**arr)
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
import jax
net=networks.make_ppo_networks({'state':76,'privileged_state':106},4,policy_hidden_layer_sizes=(256,256,256),value_hidden_layer_sizes=(256,256,256),policy_obs_key='state',value_obs_key='privileged_state',distribution_type='tanh_normal',preprocess_observations_fn=running_statistics.normalize)
fn=networks.make_inference_fn(net)((norm,payload.actor_params,payload.critic_params),deterministic=True)
obs=np.random.default_rng(420).normal(size=(64,76)).astype(np.float32);obs[0]=0
out=np.asarray(jax.vmap(lambda o:fn({'state':o},jax.random.PRNGKey(0))[0])(obs));np.savez(root/'reference_inference.npz',observations=obs,actions=out)
print('EXPORT_DONE',arr.keys())
