"""Safety limits and recording semantics; these tests never start Isaac."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / 'probe_physx_migration.py'
    assert path.is_file(), 'The bounded migration probe is not implemented yet'
    return importlib.import_module('probe_physx_migration')


def spec():
    return {
        'physics_dt': 0.001, 'control_dt': 0.02,
        'actuator_mode': {
            'rearwheel_joint': 'native_implicit',
            'steering_joint': 'native_implicit',
            'hip_joint': 'explicit_PD_capped',
            'knee_joint': 'explicit_PD_capped',
        },
    }


def test_valid_two_second_probe_has_twenty_physics_substeps(probe):
    assert probe.validate_probe(spec(), 100, 7) == 20


@pytest.mark.parametrize('steps', [0, -1, 101, True])
def test_rejects_unbounded_or_invalid_control_step_count(probe, steps):
    with pytest.raises(ValueError):
        probe.validate_probe(spec(), steps, 7)


def test_rejects_duration_over_two_seconds_even_with_one_hundred_steps(probe):
    value = spec()
    value['control_dt'] = 0.03
    with pytest.raises(ValueError, match='2 seconds'):
        probe.validate_probe(value, 100, 7)


@pytest.mark.parametrize('dt', [0, -0.001, float('nan'), 0.003, 0.03])
def test_rejects_invalid_or_nondivisible_physics_step(probe, dt):
    value = spec()
    value['physics_dt'] = dt
    with pytest.raises(ValueError):
        probe.validate_probe(value, 100, 7)


@pytest.mark.parametrize('change', ['explicit_rear', 'missing_hip', 'unknown_joint'])
def test_rejects_actuator_modes_that_physx_task_would_silently_ignore(probe, change):
    value = spec()
    if change == 'explicit_rear':
        value['actuator_mode']['rearwheel_joint'] = 'explicit_PD_capped'
    elif change == 'missing_hip':
        del value['actuator_mode']['hip_joint']
    else:
        value['actuator_mode']['frontwheel_joint'] = 'native_implicit'
    with pytest.raises(ValueError, match='actuator_mode'):
        probe.validate_probe(value, 100, 7)


def test_rejects_over_budget_physics_substeps(probe):
    value = spec()
    value['physics_dt'] = 0.00001
    with pytest.raises(ValueError, match='physics substeps'):
        probe.validate_probe(value, 100, 7)


def test_existing_output_is_preserved(probe, tmp_path):
    output = tmp_path / 'old_run'
    output.mkdir()
    original = output / 'sentinel'
    original.write_text('frozen')
    with pytest.raises(FileExistsError):
        probe.prepare_output(output)
    assert original.read_text() == 'frozen'
    assert list(output.iterdir()) == [original]


class TerminalEnvironment:
    """Replace only external physics, keeping real probe recording and stop logic."""
    n = 1
    names = ['rearwheel_joint', 'steering_joint', 'hip_joint', 'frontwheel_joint', 'knee_joint']

    def __init__(self):
        self.obs = np.zeros((1, 76), np.float32)
        self.trace = np.zeros((1, 31))
        self.calls = 0
        self.world = SimpleNamespace(step=lambda render=False: None)

    def step(self, actions):
        self.calls += 1
        if self.calls > 2:
            raise AssertionError('Probe simulated beyond actual failure')
        for _ in range(20):
            self.world.step(render=False)
        self.obs[:] = self.calls
        self.trace[:] = self.calls
        done = np.array([self.calls == 2])
        info = [{'end_code': 3 if done[0] else 0, 'terminated': bool(done[0]),
                 'truncated': False, 'reward': -400.0 if done[0] else 2.0}]
        return self.obs.copy(), np.array([info[0]['reward']]), done, info


def test_stops_on_terminal_step_keeps_pre_and_post_obs_and_counts_real_substeps(probe):
    env = TerminalEnvironment()
    result = probe.run_rollout(env, lambda observation: np.zeros((1, 4)),
                               steps=100, control_dt=0.02, substeps=20)
    assert result['physics_substeps'] == 40
    assert result['endpoint_reason'] == 'environment_done'
    np.testing.assert_array_equal(result['arrays']['done'], [False, True])
    np.testing.assert_array_equal(result['arrays']['target'][:, 0], [.02, .04])
    np.testing.assert_array_equal(result['arrays']['observations'][:, 0], [0, 1])
    np.testing.assert_array_equal(result['arrays']['next_observations'][:, 0], [1, 2])
    np.testing.assert_array_equal(result['arrays']['rewards'], [2, -400])
    assert len(result['info']) == 2


def test_world_to_body_angular_record_uses_full_orientation(probe):
    half = np.sqrt(0.5)
    view = SimpleNamespace(
        get_world_poses=lambda: (np.zeros((1, 3)), np.array([[half, 0, 0, half]])),
        get_angular_velocities=lambda: np.array([[0.0, 1.0, 0.0]]),
    )
    result = probe.angular_state(view)
    np.testing.assert_allclose(result['world_rad_s'], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(result['body_rad_s'], [1, 0, 0], atol=1e-12)


@pytest.mark.parametrize('field', ['reward', 'reset', 'episode_horizon'])
def test_rejects_config_values_not_used_by_the_frozen_task(probe, field):
    raw = {'reward': {'speed_coeff': 1.0}, 'reset': {'initial_forward_velocity': 2.0},
           'ppo': {'episode_horizon': 400}}
    value = spec()
    value[field] = {'changed': True} if field != 'episode_horizon' else 100
    with pytest.raises(ValueError, match=field):
        probe.validate_runtime_config(value, raw)


def test_invalid_action_stops_before_any_physics_step(probe):
    env = TerminalEnvironment()
    result = probe.run_rollout(env, lambda observation: np.full((1, 4), np.nan),
                               steps=100, control_dt=0.02, substeps=20)
    assert result['endpoint_reason'] == 'probe_error'
    assert result['physics_substeps'] == 0
    assert result['control_steps'] == 0
    assert env.calls == 0


def test_partial_physics_failure_preserves_completed_transition_and_actual_count(probe):
    env = TerminalEnvironment()
    attempted = 0

    def physics(render=False):
        nonlocal attempted
        attempted += 1
        if attempted == 24:
            raise RuntimeError('external physics failed')

    env.world.step = physics
    result = probe.run_rollout(env, lambda observation: np.zeros((1, 4)),
                               steps=100, control_dt=0.02, substeps=20)
    assert result['endpoint_reason'] == 'probe_error'
    assert result['physics_substeps'] == 23
    assert result['physics_substeps_attempted'] == 24
    assert result['control_steps'] == 1
    np.testing.assert_array_equal(result['arrays']['target'][:, 0], [.02])
    assert 'external physics failed' in result['error']
    assert env.world.step is physics


def test_inconsistent_physics_step_count_is_an_error(probe):
    result = probe.run_rollout(TerminalEnvironment(), lambda observation: np.zeros((1, 4)),
                               steps=100, control_dt=0.02, substeps=19)
    assert result['endpoint_reason'] == 'probe_error'
    assert result['physics_substeps'] == 20
