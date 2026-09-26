"""Pure numerical tests for bounded phase torque synthesis, zero physics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('drive_response_adapter', ROOT / 'drive_response_adapter.py')
adapter_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter_module)


def integrate(q, v, torques, inertia, h, n):
    for torque in torques:
        for _ in range(n):
            v += h * torque / inertia
            q += h * v
    return q, v


@pytest.mark.parametrize('steps', [2, 4, 10, 20])
@pytest.mark.parametrize('initial', [(0., 0.), (.8, -3.), (-.7, 4.)])
def test_two_phase_synthesis_matches_both_endpoints(steps, initial):
    q, v = initial
    h, inertia, expected = .005 / steps, .0004165, [2.3, -1.4]
    goal_q, goal_v = integrate(q, v, expected, inertia, h, steps // 2)
    plan = adapter_module.two_phase_torques(q, v, goal_q, goal_v, inertia, .005, h, 6.)
    np.testing.assert_allclose(plan['requested_torques'], expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(integrate(q, v, plan['capped_torques'], inertia, h, steps // 2), [goal_q, goal_v], atol=1e-12)
    assert not plan['any_clipped']


def test_force_clipping_is_explicit_and_removes_endpoint_guarantee():
    plan = adapter_module.two_phase_torques(0., 0., .2, 0., .0004165, .005, .0005, 6.)
    assert plan['any_clipped']
    assert np.max(np.abs(plan['capped_torques'])) <= 6.
    assert np.max(np.abs(plan['terminal_error_after_clip'])) > .1


def test_rear_rk4_stiff_servo_has_displacement_with_zero_end_speed():
    r = adapter_module.rk4_servo_target(0., 0., 12., 0., 5., .0004165, 6., .005, velocity_mode=True)
    np.testing.assert_allclose(r['q_end'], .06002400960384154, atol=1e-12)
    np.testing.assert_allclose(r['v_end'], 0., atol=1e-12)
    np.testing.assert_array_equal(r['stage_torques'], [6., -6., 6., -6.])
    plan = adapter_module.two_phase_torques(0., 0., r['q_end'], r['v_end'], .0004165, .005, .0005, 6.)
    np.testing.assert_allclose(plan['requested_torques'], [4., -4.], atol=1e-12)


def make_adapter():
    return adapter_module.PhaseDriveAdapter(
        ['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint'],
        ['steering_joint', 'rearwheel_joint', 'hip_joint', 'knee_joint'],
        {'rearwheel_joint': .0004165, 'steering_joint': .001},
        {'rearwheel_joint': {'kp': 0., 'kd': 5.}, 'steering_joint': {'kp': 10., 'kd': .1}},
        {'rearwheel_joint': 6., 'steering_joint': 5.})


def test_phase_scheduler_uses_real_current_state_once_per_macro():
    a = make_adapter()
    ctrl, q, qd = np.array([0., 12., -.8, 1.]), np.zeros(5), np.zeros(5)
    efforts = []
    for k in range(10):
        tau, audit = a.compute_efforts(k, q, qd, ctrl)
        efforts.append(tau.copy())
        qd[0] += .0005 * tau[0] / .0004165
        q[0] += .0005 * qd[0]
    np.testing.assert_allclose(np.array(efforts)[:5, 0], 4.)
    np.testing.assert_allclose(np.array(efforts)[5:, 0], -4.)
    np.testing.assert_array_equal(np.array(efforts)[:, 2:], 0.)
    np.testing.assert_allclose([q[0], qd[0]], [.06002400960384154, 0.], atol=1e-12)
    assert audit['phase_index'] == 1
    qd[0] = 3.
    tau, audit = a.compute_efforts(10, q, qd, ctrl)
    assert audit['plans']['rearwheel_joint']['initial_v'] == 3.


def test_scheduler_rejects_unannounced_input_switch_and_skipped_steps():
    a = make_adapter()
    q = np.zeros(5); ctrl = np.array([0., 12., -.8, 1.])
    a.compute_efforts(0, q, q, ctrl)
    with pytest.raises(ValueError, match='constant'):
        a.compute_efforts(1, q, q, np.array([0., 6., -.8, 1.]))
    with pytest.raises(ValueError, match='consecutive'):
        a.compute_efforts(2, q, q, ctrl)
    a.reset()
    a.compute_efforts(0, q, q, ctrl)


@pytest.mark.parametrize('h', [.003, .001, float('nan'), 0.])
def test_requires_even_integer_microsteps(h):
    with pytest.raises(ValueError):
        adapter_module.two_phase_torques(0, 0, .01, 0, .001, .005, h, 6.)
