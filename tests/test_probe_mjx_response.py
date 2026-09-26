"""Offline response-recording tests; no JAX or simulator physics is started."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('probe_mjx_response', ROOT / 'probe_mjx_response.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_free_joint_body_angular_velocity_rotates_to_world_once():
    qpos = np.zeros(12)
    qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    qvel = np.zeros(11)
    qvel[3:6] = [1, 2, 3]
    np.testing.assert_allclose(probe._world_angular(qpos, qvel), [-2, 1, 3], atol=1e-14)


def test_native_sensor_stage_and_end_state_are_kept_distinct():
    import mujoco
    from response_protocol import SOURCE_MODEL
    m = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qvel[3:6] = [1, 2, 3]
    # Synthetic sentinels in a record-only test expose accidental sensor refresh.
    d.sensor('gyro_local').data[:] = [-4, -5, -6]
    d.sensor('ang_global').data[:] = [7, 8, 9]
    meta = {'qa': [7, 8, 9, 10, 11], 'va': [6, 7, 8, 9, 10]}
    r = probe.state_record(d, m, meta, [0, 12, -.8, 1.], .005)
    np.testing.assert_array_equal(r['gyro_body_native'], [-4, -5, -6])
    np.testing.assert_array_equal(r['omega_world_native'], [7, 8, 9])
    np.testing.assert_array_equal(r['omega_body_end'], [1, 2, 3])
    np.testing.assert_array_equal(r['omega_world_end'], [1, 2, 3])
    assert r['time_s'] == r['time'] == .005


def test_joint_pose_difference_is_independent_of_end_qd(tmp_path):
    rows = [{'time': np.asarray(t), 'joint_q': np.array([q]), 'joint_qd': np.array([0.])}
            for t, q in [(0., 0.), (.005, .06), (.01, .12)]]
    result = probe.save_trace(tmp_path / 'trace.npz', rows,
                              {'joint_names': ['rearwheel_joint'], 'actuator_names': ['rear']})
    assert np.isnan(result['joint_interval_velocity'][0, 0])
    np.testing.assert_allclose(result['joint_interval_velocity'][1:, 0], 12.)
    np.testing.assert_array_equal(result['joint_qd'], 0.)
