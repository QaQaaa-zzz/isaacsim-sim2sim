"""Contract tests for paired actuator experiments; no simulator steps."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('response_protocol', ROOT / 'response_protocol.py')
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)


def declaration():
    return {
        'schema_version': 1, 'physics_dt': .005, 'integrator': 'RK4',
        'gravity': [0, 0, 0], 'contact_enabled': False,
        'initial': {'qpos': [1.5, 0, .15, 1, 0, 0, 0, 0, 0, 0, -.8, 1.],
                    'qvel': [0.] * 11},
        'cases': [{'name': 'rear_step', 'enabled_joints': ['rearwheel_joint'],
                   'segments': [{'duration_s': .01, 'ctrl': [0, 0, -.8, 1.]},
                                {'duration_s': .02, 'ctrl': [0, 12, -.8, 1.]}]}],
    }


def test_piecewise_controls_switch_before_boundary_step():
    case = protocol.validate_protocol(declaration())[0]
    assert case['steps'] == 6
    np.testing.assert_array_equal(case['controls'][:, 1], [0, 0, 12, 12, 12, 12])
    assert case['enabled_joints'] == ['rearwheel_joint']


def test_target_finer_dt_preserves_same_input_timing():
    case = protocol.validate_protocol(declaration(), physics_dt=.001)[0]
    assert case['steps'] == 30
    np.testing.assert_array_equal(case['controls'][:10, 1], 0.)
    np.testing.assert_array_equal(case['controls'][10:, 1], 12.)


@pytest.mark.parametrize('mutate', [
    lambda s: s.update(physics_dt=float('nan')),
    lambda s: s.update(contact_enabled=0),
    lambda s: s['initial'].update(qvel=[0.] * 10),
    lambda s: s['initial']['qpos'].__setitem__(3, 2.),
    lambda s: s['cases'][0].update(name='../unsafe'),
    lambda s: s['cases'][0].update(enabled_joints=['frontwheel_joint']),
    lambda s: s['cases'][0]['segments'][0].update(duration_s=.006),
    lambda s: s['cases'][0]['segments'][0].update(ctrl=[0, 12, float('inf'), 1]),
    lambda s: s.update(model_path='/home/qy/DVGC/other.xml'),
    lambda s: s.update(initial={}),
    lambda s: s['cases'].append(copy.deepcopy(s['cases'][0])),
])
def test_rejects_unreproducible_declarations(mutate):
    spec = declaration()
    mutate(spec)
    with pytest.raises(ValueError):
        protocol.validate_protocol(spec)


def test_total_budget_applies_across_cases():
    spec = declaration()
    spec['physics_dt'] = .00005
    spec['cases'][0]['segments'] = [{'duration_s': .05, 'ctrl': [0, 12, -.8, 1.]}]
    spec['cases'] *= 3
    spec['cases'] = [{**c, 'name': f'case_{i}'} for i, c in enumerate(spec['cases'])]
    with pytest.raises(ValueError, match='2000'):
        protocol.validate_protocol(spec)


def test_declared_duration_cap_is_point_two_seconds():
    spec = declaration()
    spec['cases'][0]['segments'] = [{'duration_s': .2, 'ctrl': [0, 12, -.8, 1.]}]
    assert protocol.validate_protocol(spec)[0]['steps'] == 40
    spec['cases'][0]['segments'][0]['duration_s'] = .205
    with pytest.raises(ValueError, match='0.2 seconds'):
        protocol.validate_protocol(spec)


def test_case_override_and_default_all_actuators():
    spec = declaration()
    del spec['cases'][0]['enabled_joints']
    initial = copy.deepcopy(spec['initial'])
    initial['qvel'][0] = 2.
    spec['cases'][0]['initial'] = initial
    case = protocol.validate_protocol(spec)[0]
    assert case['initial']['qvel'][0] == 2.
    assert len(case['enabled_joints']) == 4
    assert spec['initial']['qvel'][0] == 0.


def test_prepare_model_changes_only_declared_diagnostic_contract():
    import mujoco
    spec = declaration()
    case = protocol.validate_protocol(spec)[0]
    m, d, meta = protocol.prepare_model(spec, case)
    np.testing.assert_allclose(d.qpos, spec['initial']['qpos'])
    np.testing.assert_allclose(d.qvel, spec['initial']['qvel'])
    assert m.opt.timestep == .005
    assert m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    np.testing.assert_array_equal(m.opt.gravity, 0)
    assert meta['joint_names'] == ['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']
    for i, joint in enumerate(meta['actuator_joint_names']):
        if joint != 'rearwheel_joint':
            np.testing.assert_array_equal(m.actuator_gainprm[i], 0)
            np.testing.assert_array_equal(m.actuator_biasprm[i], 0)
    assert m.actuator_gainprm[1, 0] == 5
    assert d.time == 0


def test_ctrl_range_is_checked_against_compiled_model():
    spec = declaration()
    spec['cases'][0]['segments'][0]['ctrl'][1] = 41
    case = protocol.validate_protocol(spec)[0]
    with pytest.raises(ValueError, match='ctrlrange'):
        protocol.prepare_model(spec, case)
