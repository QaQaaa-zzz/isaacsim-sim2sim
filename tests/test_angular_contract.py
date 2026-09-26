"""Independent MuJoCo sensor oracles for the existing PhysX observation bridge.

These tests use synchronized mj_forward data. They do not assert that the
frozen MJX-Warp step returns synchronized sensor values.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_runtime import Observation


@pytest.mark.parametrize("angles", [[0, 0, 0], [0, 0, np.pi / 2], [.4, -.3, .7]])
def test_body_gyro_and_critic_match_mujoco_sensors_in_radians(angles):
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1] / "model/source.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 2.0
    data.qpos[3:7] = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
    data.qvel[:6] = [1.2, -.7, .4, .8, -.5, 1.1]
    data.qvel[6:] = [-12., .2, 13., -.3, .4]
    mujoco.mj_forward(model, data)
    # Deliberately reorder joints; the bridge must identify them by name.
    names = ["knee_joint", "frontwheel_joint", "steering_joint", "rearwheel_joint", "hip_joint"]
    qa = [model.joint(name).qposadr[0] for name in names]
    va = [model.joint(name).dofadr[0] for name in names]
    world_omega = data.sensor("ang_global").data.copy()
    observer = Observation(model)
    observer.initial(data.qpos[0])
    actor = observer.advance(data.qpos[:3], data.qpos[3:7], data.qpos[qa], data.qvel[va],
                             names, data.qvel[:3], world_omega, np.zeros(3), np.zeros(4))
    frame = actor[:75].reshape(3, 25)[-1]
    np.testing.assert_allclose(frame[3:6], data.sensor("gyro_local").data, atol=1e-6)
    np.testing.assert_allclose(frame[12:17], [.2, -.3, .4, 13., -12.], atol=1e-6)
    critic = observer.privileged(actor, data.qvel[:3], world_omega, data.qvel[va], names)
    np.testing.assert_allclose(critic[88:99], data.qvel, atol=1e-6)


def test_body_angular_velocity_is_not_euler_angle_derivative():
    quat = Rotation.from_euler("xyz", [.4, -.3, .7]).as_quat()[[3, 0, 1, 2]]
    before = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz")
    body_omega = np.array([.8, -.5, 1.1])
    after_quat = quat.copy()
    mujoco.mju_quatIntegrate(after_quat, body_omega, 1e-7)
    after = Rotation.from_quat(after_quat[[1, 2, 3, 0]]).as_euler("xyz")
    assert np.linalg.norm((after - before) / 1e-7 - body_omega) > .1
