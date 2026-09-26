"""Two offline Actor calls; replace only one diagnostic frame's gyro channels."""
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path('/home/qy/ISAAC——SIM')
OUT = ROOT / 'results/migration_audit_20260926'
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from policy_runtime import Actor, Observation, control

target = OUT / 'gyro_action_sensitivity.json'
if target.exists():
    raise FileExistsError(target)

trace_path = OUT / 'angular_probe_warp_mujoco_3.6.0.json'
saved = json.loads(trace_path.read_text())
row = saved['timing_rows'][-1]
cfg = json.loads((ROOT / 'policy/resolved_config.json').read_text())
model = mujoco.MjModel.from_xml_path(str(ROOT / 'model/source.xml'))
actor = Actor(ROOT / 'policy/actor.npz')
qpos, qvel = np.asarray(row['qpos_end']), np.asarray(row['qvel_end'])
gyro = np.asarray(row['gyro_after_step'])
acc = np.asarray(row['acc_sensor_after_step'])
rotation = Rotation.from_quat(qpos[[4, 5, 6, 3]])
names = [model.joint(j).name for j in range(1, model.njnt)]
qa = [model.joint(name).qposadr[0] for name in names]
va = [model.joint(name).dofadr[0] for name in names]
observer = Observation(model)
observer.initial(float(qpos[0]))
raw_obs = observer.advance(
    qpos[:3], qpos[3:7], qpos[qa], qvel[va], names, qvel[:3],
    rotation.apply(gyro), rotation.apply(acc) + [0., 0., -9.81], np.zeros(4))
np.testing.assert_array_equal(raw_obs[:50], np.zeros(50))
np.testing.assert_array_equal(raw_obs[53:56], gyro.astype(np.float32))
np.testing.assert_array_equal(raw_obs[56:59], acc.astype(np.float32))
np.testing.assert_array_equal(raw_obs[67:71], np.zeros(4))
np.testing.assert_array_equal(raw_obs[[24, 49, 74]], [0, 0, 1])

final_obs = raw_obs.copy()
final_obs[53:56] = qvel[3:6]
changed_indices = np.flatnonzero(final_obs != raw_obs).tolist()
assert changed_indices == [53, 54, 55]
cases = {}
for name, obs in [('warp_raw_gyro', raw_obs), ('final_qvel_body', final_obs)]:
    normalized = (np.asarray(obs, dtype=np.float32) - actor.p['mean']) / actor.p['std']
    action = actor(obs)  # Exactly two total Actor inferences.
    controls = control(action, model, cfg)
    assert np.isfinite(np.r_[normalized, action, controls]).all()
    cases[name] = dict(observation=obs.tolist(), normalized_observation=normalized.tolist(),
                       action=action.tolist(), control=controls.tolist())
delta = {}
for field in ['observation', 'normalized_observation', 'action', 'control']:
    difference = np.asarray(cases['final_qvel_body'][field]) - np.asarray(cases['warp_raw_gyro'][field])
    delta[field] = difference.tolist()
    delta[field + '_l2'] = float(np.linalg.norm(difference))
    delta[field + '_max_abs'] = float(np.max(np.abs(difference)))
assert np.flatnonzero(np.asarray(delta['normalized_observation'])).tolist() == changed_indices
paths = [trace_path, ROOT/'policy/actor.npz', ROOT/'policy/resolved_config.json',
         ROOT/'policy_runtime.py', ROOT/'model/source.xml', Path(__file__)]
report = dict(
    role='offline diagnostic only, not actual source observation prefix or closed-loop causal evidence',
    source_trace=str(trace_path), source_trace_engine='MuJoCo/MJX 3.6.0 CUDA Warp',
    source_state_time_s=row['time_s'], python=sys.executable, mujoco=mujoco.__version__,
    actor_inferences=2, new_physics_steps=0, training_transitions=0,
    observation_protocol='76D; first two 25D frames zero; one valid frame; zero last action; geometry/gravity/joints/linear velocity use final state; raw Warp accelerometer identical in both cases; only gyro differs',
    not_source_prefix_reason='Source Warp sensor/kinematic stage can differ from the final state used by the diagnostic Observation helper; no source history replay is claimed',
    valid_history_masks=raw_obs[[24,49,74]].tolist(), changed_observation_indices=changed_indices,
    normalizer_gyro_mean=np.asarray(actor.p['mean'])[53:56].tolist(),
    normalizer_gyro_std=np.asarray(actor.p['std'])[53:56].tolist(),
    control_order=['steering_position_rad','rearwheel_velocity_rad_per_s','hip_position_rad','knee_position_rad'],
    action_order=['steering','rearwheel','hip','knee'],
    cases=cases, delta_final_qvel_minus_warp_raw=delta,
    file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:report[k] for k in ['actor_inferences','new_physics_steps','changed_observation_indices','normalizer_gyro_mean','normalizer_gyro_std']}, indent=2))
for name, case in cases.items():
    print(name, 'normalized gyro',case['normalized_observation'][53:56], 'action',case['action'], 'control',case['control'])
print('delta action',delta['action'],'delta control',delta['control'])
