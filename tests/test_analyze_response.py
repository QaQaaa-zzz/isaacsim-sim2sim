import numpy as np
import pytest

from analyze_response import paired_samples, response_metrics, admissible, validate_trace


def trace(dt, error=0.):
    t = np.arange(round(.12 / dt) + 1) * dt
    return dict(time_s=t, joint_names=np.array(['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']),
                joint_q=np.tile(t[:, None], (1, 5)) + error,
                joint_qd=np.ones((len(t), 5)), omega_body_end=np.zeros((len(t), 3)),
                ctrl=np.zeros((len(t), 4)), qpos=np.zeros((len(t), 12)), qvel=np.zeros((len(t), 11)))


def test_common_samples_use_real_10ms_states_without_interpolation():
    a, b = trace(.005), trace(.002)
    ia, ib = paired_samples(a['time_s'], b['time_s'])
    np.testing.assert_allclose(a['time_s'][ia], np.arange(1, 13) * .01)
    np.testing.assert_allclose(a['time_s'][ia], b['time_s'][ib])
    b['time_s'][5] += .0001
    with pytest.raises(ValueError, match='missing'):
        paired_samples(a['time_s'], b['time_s'])


def test_matching_response_zero_and_named_order_must_match():
    a, b = trace(.005), trace(.001)
    assert response_metrics(a, b)['loss'] == pytest.approx(0.)
    b['joint_names'] = b['joint_names'][::-1]
    with pytest.raises(ValueError, match='joint'):
        response_metrics(a, b)


def test_smaller_velocity_error_cannot_hide_larger_angle_error():
    baseline = dict(q_rmse_rad=.01, qd_rmse_rad_s=10., omega_rmse_rad_s=.01)
    candidate = dict(q_rmse_rad=.3, qd_rmse_rad_s=1., omega_rmse_rad_s=.01)
    assert not admissible(candidate, baseline)
    assert admissible(baseline, baseline)


def test_input_mismatch_rejected():
    a, b = trace(.005), trace(.001)
    b['ctrl'][10, 0] = .1
    with pytest.raises(ValueError, match='input'):
        response_metrics(a, b)


def test_all_substep_inputs_and_initial_state_are_required():
    a = trace(.001)
    spec = dict(physics_dt=.005, initial=dict(qpos=[0.] * 12, qvel=[0.] * 11))
    case = dict(segments=[dict(duration_s=.12, ctrl=[0.] * 4)])
    validate_trace(a, spec, case)
    a['ctrl'][7, 0] = .1
    with pytest.raises(ValueError, match='input'):
        validate_trace(a, spec, case)
    a['ctrl'][7, 0] = 0.
    del a['qpos']
    with pytest.raises(ValueError, match='qpos'):
        validate_trace(a, spec, case)
