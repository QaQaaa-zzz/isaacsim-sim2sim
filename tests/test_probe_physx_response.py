"""Pure interface/recording checks; these tests never initialize Isaac."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ACTUATORS = ['steering_joint', 'rearwheel_joint', 'hip_joint', 'knee_joint']
NAMES = ['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[1] / 'probe_physx_response.py'
    assert path.is_file(), 'The independent PhysX response probe is not implemented yet'
    return importlib.import_module('probe_physx_response')


def candidate():
    return dict(name='candidate_a', physics_dt=.001, drive_mode='hybrid',
                target_drive_gains={name: dict(kp=0. if name == 'rearwheel_joint' else 10., kd=.4)
                                    for name in ACTUATORS})


def cases():
    return [dict(name='step', enabled_joints=ACTUATORS, segments=[
        dict(duration_s=.01, ctrl=[0, 0, -1.2, 2.5]),
        dict(duration_s=.02, ctrl=[0, 12, -1.2, 2.5])])]


def test_candidates_count_real_steps_and_reject_bad_selection(probe):
    values, total = probe.validate_candidates([candidate()], cases(), ACTUATORS)
    assert total == 30
    assert values[0]['case_names'] == ['step']
    bad = candidate()
    bad['case_names'] = ['not_a_case']
    with pytest.raises(ValueError, match='case_names'):
        probe.validate_candidates([bad], cases(), ACTUATORS)


@pytest.mark.parametrize('key,value', [('name', '../escape'), ('physics_dt', .003),
                                      ('physics_dt', True), ('drive_mode', 'unknown')])
def test_reject_invalid_candidate_before_isaac_import(probe, key, value):
    bad = candidate()
    bad[key] = value
    with pytest.raises(ValueError):
        probe.validate_candidates([bad], cases(), ACTUATORS)


def test_reject_wrong_gain_mapping_negative_gain_and_over_budget(probe):
    bad = candidate()
    bad['target_drive_gains']['hip_joint']['kd'] = -1.
    with pytest.raises(ValueError, match='gain'):
        probe.validate_candidates([bad], cases(), ACTUATORS)
    bad = candidate()
    del bad['target_drive_gains']['knee_joint']
    with pytest.raises(ValueError, match='gain'):
        probe.validate_candidates([bad], cases(), ACTUATORS)
    with pytest.raises(ValueError, match='budget'):
        probe.validate_candidates([candidate()], cases(), ACTUATORS, budget=29)


def test_control_timeline_uses_input_for_just_completed_interval(probe):
    time, ctrl = probe.control_timeline(cases()[0], .005)
    np.testing.assert_allclose(time, np.arange(7) * .005)
    np.testing.assert_array_equal(ctrl[:, 1], [0, 0, 0, 12, 12, 12, 12])


def test_com_and_body_angular_conversion_round_trip_under_compound_orientation(probe):
    quat = Rotation.from_euler('xyz', [.4, -.3, .7]).as_quat()[[3, 0, 1, 2]]
    body_omega = np.array([.8, -.5, 1.1])
    origin = np.array([2., -.2, .3])
    offset = np.array([.027, -.0007, .05])
    com, world = probe.initial_root_velocities(quat, origin, body_omega, offset)
    restored, body = probe.root_velocities_from_physx(quat, com, world, offset)
    np.testing.assert_allclose(restored, origin, atol=1e-12)
    np.testing.assert_allclose(body, body_omega, atol=1e-12)
    assert np.linalg.norm(com - origin) > .01
    assert np.linalg.norm(world - body_omega) > .1


def test_hybrid_disables_unselected_joint_and_preserves_name_mapping(probe):
    names = ['knee_joint', 'frontwheel_joint', 'steering_joint', 'rearwheel_joint', 'hip_joint']
    kp, kd, explicit = probe.drive_arrays(names, ACTUATORS, candidate()['target_drive_gains'],
                                         ['rearwheel_joint', 'hip_joint'], 'hybrid')
    np.testing.assert_array_equal(kp, [0, 0, 0, 0, 0])
    np.testing.assert_array_equal(kd, [0, 0, 0, .4, 0])
    np.testing.assert_array_equal(explicit, [False, False, False, False, True])


def test_explicit_effort_uses_velocity_rear_and_capped_position_pd(probe):
    gains = candidate()['target_drive_gains']
    gains['hip_joint'] = dict(kp=100., kd=6.)
    q = np.zeros(5); qd = np.zeros(5)
    effort = probe.explicit_efforts(NAMES, ACTUATORS, gains, np.array([False, False, False, True, False]),
                                   q, qd, np.array([.2, 12, 1., 2.5]), np.array([[-6, 6], [-6, 6], [-30, 30], [-30, 30]]))
    np.testing.assert_array_equal(effort, [0, 0, 0, 30, 0])
    mask = np.array([True, False, False, False, False])
    effort = probe.explicit_efforts(NAMES, ACTUATORS, gains, mask, q, qd,
                                   np.array([.2, 12, 1., 2.5]), np.array([[-6, 6], [-6, 6], [-30, 30], [-30, 30]]))
    np.testing.assert_allclose(effort, [4.8, 0, 0, 0, 0])


def test_existing_output_is_not_overwritten(probe, tmp_path):
    path = tmp_path / 'frozen'
    path.mkdir()
    with pytest.raises(FileExistsError):
        probe.prepare_output(path)


def phase_candidate():
    value = candidate()
    value.update(physics_dt=.0005, drive_mode='phase', phase_adapter={
        'macro_dt': .005, 'effective_inertias': {'rearwheel_joint': .0004165, 'steering_joint': .001}})
    value['target_drive_gains']['rearwheel_joint']['kd'] = 5.
    return value


def test_phase_candidate_requires_even_microsteps_and_macro_aligned_inputs(probe):
    values, total = probe.validate_candidates([phase_candidate()], cases(), ACTUATORS)
    assert total == 60
    bad = phase_candidate()
    bad['physics_dt'] = .001
    with pytest.raises(ValueError, match='even'):
        probe.validate_candidates([bad], cases(), ACTUATORS)
    changed = cases()
    changed[0]['segments'][0]['duration_s'] = .011
    with pytest.raises(ValueError, match='macro'):
        probe.validate_candidates([phase_candidate()], changed, ACTUATORS)


def test_phase_disables_native_front_drives_and_only_selected_adapter_joints(probe):
    value = phase_candidate()
    kp, kd, explicit = probe.drive_arrays(NAMES, ACTUATORS, value['target_drive_gains'],
                                         ['rearwheel_joint', 'hip_joint'], 'phase')
    np.testing.assert_array_equal(kp, 0)
    np.testing.assert_array_equal(kd, 0)
    np.testing.assert_array_equal(explicit, [False, False, False, True, False])
    caps = np.array([[-5, 5], [-6, 6], [-30, 30], [-30, 30]])
    adapter = probe.make_phase_adapter(NAMES, ACTUATORS, value, ['rearwheel_joint', 'hip_joint'], caps)
    for k in range(10):
        effort, audit = adapter.compute_efforts(k, np.zeros(5), np.zeros(5), np.array([.2, 12, 1, 2.5]))
        assert set(audit['plans']) == {'rearwheel_joint'}
        assert effort[0] == pytest.approx(4. if k < 5 else -4.)
        np.testing.assert_array_equal(effort[1:], 0)


def test_force_cap_readback_is_checked_by_joint_name(probe):
    caps = np.array([[-5, 5], [-6, 6], [-30, 30], [-30, 30]])
    probe.verify_force_caps(NAMES, ACTUATORS, np.array([6, 5, 100, 30, 30]), caps)
    with pytest.raises(ValueError, match='force'):
        probe.verify_force_caps(NAMES, ACTUATORS, np.array([60, 5, 100, 30, 30]), caps)
