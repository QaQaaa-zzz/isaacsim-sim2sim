import os
os.environ['JAX_PLATFORMS']='cpu'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import jax
import jax.numpy as jp
from source_reset import SourceResetStream

def test_full_reset_key_schedule_and_source_ranges():
    cfg={'airborne_rsi_probability':.08,'airborne_rsi_x_min':2.7,'airborne_rsi_x_max':2.9,'airborne_rsi_z_min':.38,'airborne_rsi_z_max':.45,'airborne_rsi_vx_min':1.8,'airborne_rsi_vx_max':2.2,'airborne_rsi_vz_min':3.,'airborne_rsi_vz_max':3.6,'initial_forward_velocity':2.}
    keys=jax.random.split(jax.random.PRNGKey(42),384)
    stream=SourceResetStream(keys,cfg,1.5,.15)
    outer=jax.vmap(jax.random.split)(keys)
    def expected(key):
        decision,state=jax.random.split(key);air=jax.random.bernoulli(decision,.08);draws=jax.random.split(state,4)
        values=[jax.random.uniform(k,(),minval=lo,maxval=hi) for k,lo,hi in zip(draws,[2.7,.38,1.8,3.],[2.9,.45,2.2,3.6])]
        return jp.r_[jp.where(air,jp.array(values),jp.array([1.5,.15,2.,0.])),air.astype(jp.float32)]
    np.testing.assert_array_equal(stream.pending,np.asarray(jax.vmap(expected)(outer[:,1])))
    for _ in range(3):
        outer=jax.vmap(jax.random.split)(outer[:,0]);inner=jax.vmap(jax.random.split)(outer[:,1])
        stream.advance()
        np.testing.assert_array_equal(stream.pending,np.asarray(jax.vmap(expected)(inner[:,1])))
