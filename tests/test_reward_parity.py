import os,sys,unittest,importlib
os.environ['JAX_PLATFORMS']='cpu';sys.dont_write_bytecode=True;sys.path.insert(0,'/home/qy/DVGC/JIT/src')
import numpy as np,jax
from pathlib import Path
from jit_dvgc.config import load_config
from jit_dvgc import rewards,semantics
class RewardParity(unittest.TestCase):
 def test_original_jax_numpy_functions(self):
  importlib.reload(rewards);importlib.reload(semantics);cfg=load_config(Path(__file__).resolve().parents[1]/'policy/resolved_config.json',runtime_only=True);rng=np.random.default_rng(8);fixtures=[];expected=[]
  for k in range(48):
   values=rng.normal(size=16).astype(np.float32);values[2]=rng.uniform(.1,.8);state=rewards.RewardState(*values);inputs=rewards.RewardInputs(state,rng.uniform(-1,1,4).astype(np.float32),rng.uniform(-1,1,4).astype(np.float32),*[np.asarray(x) for x in rng.integers(0,2,9).astype(bool)]);fixtures.append(inputs);r=rewards.phase_u_reward(inputs,cfg.reward,cfg.physical_limits);expected.append([float(r.total)]+[float(v) for v in r.components.values()])
  rewards.jp=np;actual=[]
  for inputs in fixtures:
   r=rewards.phase_u_reward(inputs,cfg.reward,cfg.physical_limits);actual.append([float(r.total)]+[float(v) for v in r.components.values()])
  np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-4)
  fixtures=[];expected=[]
  for k in range(48):
   p=semantics.initial_event_state(np.asarray(1.5),cfg);sig=semantics.PhaseUSignals(np.asarray(rng.uniform(2.,4.2)),np.asarray(rng.uniform(.1,.8)),np.asarray(rng.uniform(0,3)),np.asarray(rng.uniform(-2,2)),np.asarray(k%3==0));fixtures.append((p,sig));out=semantics.advance_events(p,sig,cfg);expected.append([np.array(v) for v in jax.tree_util.tree_leaves(out)])
  semantics.jp=np
  for pair,exp in zip(fixtures,expected):
   out=semantics.advance_events(*pair,cfg)
   for a,b in zip(jax.tree_util.tree_leaves(out),exp):np.testing.assert_allclose(a,b)
if __name__=='__main__':unittest.main()
