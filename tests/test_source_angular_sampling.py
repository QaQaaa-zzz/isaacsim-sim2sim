"""CPU-only angular sampling contract; at most 44 actual mj_step calls per run.

The frozen backup guards legacy CPU values. CPU native_sensor is not an oracle
for the frozen MJX-Warp sensor sampling phase.
"""
import os
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import source_task
from domains import identity_domain


@pytest.fixture(scope='module', autouse=True)
def cpu_physics_budget():
    original = mujoco.mj_step
    calls = []

    def counted_step(*args, **kwargs):
        assert len(calls) < 44, 'CPU engineering test exceeded its declared physical-step budget'
        calls.append(None)
        return original(*args, **kwargs)

    mujoco.mj_step = counted_step
    try:
        yield calls
    finally:
        mujoco.mj_step = original
        print(f'CPU_PHYSICS_STEPS={len(calls)}')


def test_default_and_explicit_native_match_frozen_cpu_values_for_multiple_steps():
    backup = ROOT / 'results/migration_audit_20260926/implementation_before/source_task.py'
    namespace = {'__file__': str(ROOT / 'source_task.py'), '__name__': 'source_task_before'}
    exec(compile(backup.read_text(), str(backup), 'exec'), namespace)
    frozen = namespace['SourceTask']()
    implicit = source_task.SourceTask()
    explicit = source_task.SourceTask()
    observations = [frozen.reset(seed=19), implicit.reset(seed=19),
                    explicit.reset({**identity_domain(), 'angular_velocity_mode': 'native_sensor'}, seed=19)]
    for observation in observations[1:]:
        np.testing.assert_array_equal(observation, observations[0])
    for action in ([.03, -.02, .01, -.01], [-.02, .01, -.01, .02], [.01, 0., 0., 0.]):
        expected = frozen.step(action)
        for env in (implicit, explicit):
            observation, reward, done, info = env.step(action)
            np.testing.assert_array_equal(observation, expected[0])
            assert reward == expected[1]
            assert done == expected[2]
            assert info == expected[3]
            np.testing.assert_array_equal(env.d.qpos, frozen.d.qpos)
            np.testing.assert_array_equal(env.d.qvel, frozen.d.qvel)
        if expected[2]:
            break


def test_end_state_gyro_and_reward_use_final_body_velocity_and_world_rotation():
    native, aligned = source_task.SourceTask(), source_task.SourceTask()
    for env, mode in ((native, 'native_sensor'), (aligned, 'end_state')):
        env.reset({**identity_domain(), 'angular_velocity_mode': mode}, seed=23)
        env.d.qpos[2] = 1.0
        env.d.qpos[3:7] = Rotation.from_euler('xyz', [.15, -.2, .3]).as_quat()[[3, 0, 1, 2]]
        env.d.qvel[3:6] = [.8, -.5, 1.1]
        env.d.qpos[env.m.joint('steering_joint').qposadr[0]] = .1
        mujoco.mj_forward(env.m, env.d)
    original_obs, _, original_done, _ = native.step([.2, 0., .1, -.1])
    observation, _, done, info = aligned.step([.2, 0., .1, -.1])
    np.testing.assert_array_equal(aligned.d.qpos, native.d.qpos)
    np.testing.assert_array_equal(aligned.d.qvel, native.d.qvel)
    assert done == original_done
    frame = observation[:75].reshape(3, 25)[-1]
    assert frame[-1] == 1
    body = aligned.d.qvel[3:6].copy()
    np.testing.assert_allclose(frame[3:6], body, rtol=1e-6, atol=1e-7)
    # All observation fields other than the new angular sample retain their values.
    keep = np.ones(76, dtype=bool)
    keep[53:56] = False
    np.testing.assert_array_equal(observation[keep], original_obs[keep])
    world = np.empty(3)
    mujoco.mju_rotVecQuat(world, body, aligned.d.qpos[3:7])
    state = source_task.rewards.RewardState(*np.zeros(16)).replace(pitch_rate=world[1])
    reward_input = source_task.rewards.RewardInputs(
        state, np.zeros(4), np.zeros(4), *[np.asarray(False) for _ in range(9)])
    expected = source_task.rewards.phase_u_reward(reward_input, source_task.CFG.reward,
                                                source_task.CFG.physical_limits)
    assert info['components']['pitch_rate'] == pytest.approx(float(expected.components.pitch_rate), abs=1e-12)
    assert info['angular_velocity_mode'] == 'end_state'
    # Derive Euler rates independently by quaternion integration, without a physics step.
    quat = aligned.d.qpos[3:7].copy()
    before = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')
    mujoco.mju_quatIntegrate(quat, body, 1e-7)
    euler_rate = (Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz') - before) / 1e-7
    assert np.linalg.norm(euler_rate - body) > .05
    assert np.linalg.norm(world - body) > .05


@pytest.mark.parametrize('mode', ['world', '', None])
def test_unknown_angular_sampling_mode_is_rejected(mode):
    with pytest.raises(ValueError, match='angular_velocity_mode'):
        source_task.SourceTask().reset({**identity_domain(), 'angular_velocity_mode': mode})
