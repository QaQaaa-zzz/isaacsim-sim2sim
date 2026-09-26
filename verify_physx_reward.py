"""Recompute recorded real-PhysX reward components using original JAX functions."""
import os,sys,json
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';sys.dont_write_bytecode=True;sys.path.insert(0,'/home/qy/DVGC/JIT/src')
import numpy as np
from jit_dvgc.config import load_config
from jit_dvgc import rewards
ROOT=Path(__file__).resolve().parent
cfg=load_config(ROOT/'policy/resolved_config.json',runtime_only=True)
fixtures=json.loads((ROOT/'results/isaac_training_engineering/reward_fixtures.json').read_text());max_error=0.;count=0
for frame in fixtures:
    for row in frame:
        state=rewards.RewardState(*map(np.float32,row['reward_state']))
        ri=rewards.RewardInputs(state,np.array(row['action'],np.float32),np.array(row['last_action'],np.float32),*[np.asarray(x) for x in row['reward_flags']])
        result=rewards.phase_u_reward(ri,cfg.reward,cfg.physical_limits)
        expected=np.array([float(result.total)]+[float(v) for v in result.components.values()])
        actual=np.array([row['reward_unadjusted']]+list(row['components'].values()))
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-4)
        reward=cfg.reward.failed_episode_return-row['return_before'] if row['reward_flags'][6] or row['reward_flags'][7] else float(result.total)
        np.testing.assert_allclose(row['reward'],reward,rtol=2e-5,atol=2e-4)
        np.testing.assert_allclose(sum(row['components'].values()),float(result.unclipped_total),rtol=2e-5,atol=2e-4)
        assert max(abs(row['reward_state'][-2]),abs(row['reward_state'][-1]))<=30.00001
        max_error=max(max_error,float(np.max(np.abs(actual-expected))));count+=1
report=dict(original_jax_reward_components_checked=count,max_absolute_difference=max_error,passed=True,torque_source='actual applied capped hip/knee PD effort, tensor readback checked in environment')
(ROOT/'results/isaac_training_engineering/reward_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
