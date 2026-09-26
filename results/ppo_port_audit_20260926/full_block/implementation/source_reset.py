"""Exact JAX draws and full-reset RNG advancement from Playground wrapper."""
import numpy as np
import jax
import jax.numpy as jp

def initial_env_keys(seed,n):
    _,local=jax.random.split(jax.random.PRNGKey(seed))
    local=jax.random.fold_in(local,0)
    _,key_env,_=jax.random.split(local,3)
    return jax.random.split(key_env,n)

class SourceResetStream:
    def __init__(self,keys,cfg,ground_x,ground_z):
        self.split=jax.jit(jax.vmap(jax.random.split))
        def sample(key):
            decision,state=jax.random.split(key)
            air=jax.random.bernoulli(decision,cfg['airborne_rsi_probability'])
            ks=jax.random.split(state,4)
            values=jp.array([jax.random.uniform(k,(),minval=cfg['airborne_rsi_'+field+'_min'],maxval=cfg['airborne_rsi_'+field+'_max']) for k,field in zip(ks,['x','z','vx','vz'])])
            return jp.r_[jp.where(air,values,jp.array([ground_x,ground_z,cfg['initial_forward_velocity'],0.])),air.astype(jp.float32)]
        self.sample=jax.jit(jax.vmap(sample))
        pair=self.split(keys);self.keys=pair[:,0];self.pending=np.asarray(self.sample(pair[:,1]))
    def advance(self):
        pair=self.split(self.keys);self.keys=pair[:,0]
        nested=self.split(pair[:,1])
        self.pending=np.asarray(self.sample(nested[:,1]))
