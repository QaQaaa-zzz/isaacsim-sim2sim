"""Read-only model/API coordinate and sensor-time audit. Never imports Isaac Sim."""
import argparse
import json
import importlib.metadata
import hashlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/home/qy/ISAAC——SIM')
OUT = ROOT / 'results/migration_audit_20260926'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_COMPILATION_CACHE_DIR'] = str(OUT / 'cache/jax')
os.environ['WARP_CACHE_PATH'] = str(OUT / 'cache/warp')
sys.path.insert(0, str(ROOT))
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from policy_runtime import Observation, control

p = argparse.ArgumentParser()
p.add_argument('--engine', choices=['cpu', 'warp'], default='cpu')
args = p.parse_args()
cfg = json.loads((ROOT/'policy/resolved_config.json').read_text())
m = mujoco.MjModel.from_xml_path(str(ROOT/'model/source.xml'))
m.opt.timestep = .005
names = [m.joint(j).name for j in range(1, m.njnt)]
qa = [m.joint(n).qposadr[0] for n in names]
va = [m.joint(n).dofadr[0] for n in names]
d = mujoco.MjData(m)

coordinate_rows = []
for euler, local in [([0, 0, np.pi/2], [1., 0., 0.]), ([.3, -.2, .5], [.7, -.4, .2])]:
    mujoco.mj_resetDataKeyframe(m, d, 0)
    r = Rotation.from_euler('xyz', euler)
    d.qpos[3:7] = r.as_quat()[[3, 0, 1, 2]]
    d.qpos[2] = 2.
    d.qvel[:3] = [2., -.2, .3]
    d.qvel[3:6] = local
    mujoco.mj_forward(m, d)
    omega = d.sensor('ang_global').data.copy()
    observer = Observation(m)
    actor = observer.advance(d.qpos[:3], d.qpos[3:7], d.qpos[qa], d.qvel[va], names,
                             d.qvel[:3], omega, np.zeros(3), np.zeros(4))
    critic = observer.privileged(actor, d.qvel[:3], omega, d.qvel[va], names)
    com_vel = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, com_vel, 0)
    correction = np.cross(omega, r.apply(m.body_ipos[1]))
    np.testing.assert_allclose(actor[53:56], d.sensor('gyro_local').data, atol=1e-6)
    np.testing.assert_allclose(critic[76+m.nq+3:76+m.nq+6], d.qvel[3:6], atol=1e-6)
    np.testing.assert_allclose(com_vel[3:]-correction, d.qvel[:3], atol=1e-12)
    coordinate_rows.append(dict(euler_rad=euler, qvel_local=d.qvel[3:6].tolist(),
        gyro_local=d.sensor('gyro_local').data.tolist(), ang_global=omega.tolist(),
        actor_gyro=actor[53:56].tolist(), critic_omega=critic[76+m.nq+3:76+m.nq+6].tolist(),
        root_origin_velocity=d.qvel[:3].tolist(), com_velocity=com_vel[3:].tolist(),
        com_correction=correction.tolist()))

mujoco.mj_resetDataKeyframe(m, d, 0)
d.qvel[0] = cfg['reset']['initial_forward_velocity']
d.ctrl[:] = control(np.zeros(4), m, cfg)
mujoco.mj_forward(m, d)
initial_qvel = d.qvel.copy()
initial_sensor = d.sensordata.copy()
if args.engine == 'warp':
    import warp as wp
    wp.config.kernel_cache_dir = str(OUT/'cache/warp')
    import jax
    from mujoco import mjx
    mm = mjx.put_model(m, impl='warp')
    dd = mjx.put_data(m, d, impl='warp', naconmax=128, njmax=256)
    step = jax.jit(lambda state: mjx.step(mm, state))

timing_rows = []
previous_qvel = initial_qvel.copy()
for k in range(4):
    if args.engine == 'warp':
        dd = step(dd)
        jax.block_until_ready(dd)
        qpos, qvel, sensors = map(np.asarray, (dd.qpos, dd.qvel, dd.sensordata))
        actuator_force = np.asarray(dd.actuator_force)
    else:
        mujoco.mj_step(m, d)
        qpos, qvel, sensors = d.qpos.copy(), d.qvel.copy(), d.sensordata.copy()
        actuator_force = d.actuator_force.copy()
    fresh = mujoco.MjData(m)
    fresh.qpos[:] = qpos
    fresh.qvel[:] = qvel
    fresh.ctrl[:] = d.ctrl
    mujoco.mj_forward(m, fresh)
    def s(name):
        sensor = m.sensor(name)
        return sensors[sensor.adr[0]:sensor.adr[0]+sensor.dim[0]].tolist()
    omega_world_end = Rotation.from_quat(qpos[[4, 5, 6, 3]]).apply(qvel[3:6])
    timing_rows.append(dict(step=k+1, time_s=(k+1)*m.opt.timestep,
        qpos_end=qpos.tolist(), qvel_before=previous_qvel.tolist(), qvel_end=qvel.tolist(),
        gyro_after_step=s('gyro_local'), ang_global_after_step=s('ang_global'),
        gyro_refreshed=fresh.sensor('gyro_local').data.tolist(),
        ang_global_end=omega_world_end.tolist(), acc_sensor_after_step=s('acc_local'),
        acc_sensor_refreshed=fresh.sensor('acc_local').data.tolist(),
        qvel_finite_difference=((qvel[:3]-previous_qvel[:3])/m.opt.timestep).tolist(),
        rear_sensor_after_step=s('rearwheel_joint_vel_sensor'),
        rear_qvel_end=float(qvel[m.joint('rearwheel_joint').dofadr[0]]),
        actuator_force_after_step=actuator_force.tolist(),
        actuator_force_refreshed=fresh.actuator_force.tolist(),
        gyro_end_max_abs_error=float(np.max(np.abs(np.asarray(s('gyro_local'))-qvel[3:6]))),
        world_end_max_abs_error=float(np.max(np.abs(np.asarray(s('ang_global'))-omega_world_end)))))
    previous_qvel = qvel.copy()

report = dict(engine=args.engine, python=sys.executable, mujoco=mujoco.__version__,
    python_version=sys.version, mujoco_module=mujoco.__file__,
    source_model_sha256=hashlib.sha256((ROOT/'model/source.xml').read_bytes()).hexdigest(),
    protocol='Four 5ms native step calls; separate CPU mj_forward diagnostic copies never fed back into trajectory',
    physics_dt=m.opt.timestep, integrator=int(m.opt.integrator), simulated_steps=4,
    control_action=[0., 0., 0., 0.], controls=d.ctrl.tolist(),
    body1=m.body(1).name, root_com=m.body_ipos[1].tolist(),
    imu_pos=m.site('imu_site').pos.tolist(), imu_quat=m.site('imu_site').quat.tolist(),
    coordinate_rows=coordinate_rows, timing_rows=timing_rows)
if args.engine == 'warp':
    report['jax_devices'] = [str(device) for device in jax.devices()]
    report['jax_version'] = jax.__version__
    report['warp_version'] = wp.__version__
    report['mjx_module'] = mjx.__file__
    report['warp_module'] = wp.__file__
report['packages'] = {}
for package in ['mujoco', 'mujoco-mjx', 'jax', 'jaxlib', 'warp-lang', 'numpy', 'scipy']:
    try:
        report['packages'][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][package] = None
target = OUT/f'angular_probe_{args.engine}_mujoco_{mujoco.__version__}.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'output':str(target), 'mujoco':mujoco.__version__, 'engine':args.engine,
    'gyro_errors':[row['gyro_end_max_abs_error'] for row in timing_rows],
    'first':timing_rows[0], 'last':timing_rows[-1]}, indent=2))
