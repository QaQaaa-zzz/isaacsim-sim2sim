"""Apply declared actuator inputs to real MJX-Warp, saving every physical step.

Use the installed source runtime:
  /home/qy/mujoco_playground/.venv/bin/python probe_mjx_response.py \
      --config declaration.json --output NEW_DIRECTORY

No policy, optimizer, source checkout writes, or CPU trajectory substitute.
The initial CPU forward pass only constructs a complete state for put_data.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from response_protocol import ROOT, SOURCE_MODEL, load_protocol, prepare_model

SOURCE_ENV = Path('/home/qy/mujoco_playground/.venv')


def _write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def _world_angular(qpos, qvel):
    # Quaternion order is wxyz, and free-joint angular qvel is body-relative.
    w, x, y, z = qpos[3:7]
    r = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])
    return r @ qvel[3:6]


def state_record(state, model, meta, ctrl, sample_time):
    """Preserve native derived arrays without calling forward after stepping."""
    qpos = np.asarray(state.qpos).copy()
    qvel = np.asarray(state.qvel).copy()
    sensors = np.asarray(state.sensordata).copy()

    def sensor(name):
        item = model.sensor(name)
        adr, dim = int(item.adr[0]), int(item.dim[0])
        return sensors[adr:adr + dim].copy()

    return {
        'time': np.asarray(sample_time),
        'time_s': np.asarray(sample_time),
        'engine_time': np.asarray(state.time).copy(),
        'qpos': qpos, 'qvel': qvel,
        'joint_q': qpos[meta['qa']], 'joint_qd': qvel[meta['va']],
        'omega_body_end': qvel[3:6].copy(),
        'omega_world_end': _world_angular(qpos, qvel),
        'root_velocity_origin': qvel[:3].copy(),
        'ctrl': np.asarray(ctrl).copy(),
        'actuator_force_native': np.asarray(state.actuator_force).copy(),
        'qfrc_actuator_native': np.asarray(state.qfrc_actuator).copy(),
        'sensordata_native': sensors,
        'gyro_body_native': sensor('gyro_local'),
        'omega_world_native': sensor('ang_global'),
        'acc_body_native': sensor('acc_local'),
    }


def save_trace(path, rows, meta):
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays['joint_interval_velocity'] = np.full_like(arrays['joint_q'], np.nan)
    if len(rows) > 1:
        arrays['joint_interval_velocity'][1:] = np.diff(arrays['joint_q'], axis=0) / np.diff(arrays['time'])[:, None]
    arrays['joint_names'] = np.asarray(meta['joint_names'])
    arrays['actuator_names'] = np.asarray(meta['actuator_names'])
    np.savez_compressed(path, **arrays)
    return arrays


def run(config, output):
    spec, cases = load_protocol(config)
    if Path(sys.prefix).resolve() != SOURCE_ENV.resolve():
        raise RuntimeError(f'run this probe with {SOURCE_ENV}/bin/python; refusing another runtime')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Do not cache compiled outputs in the source checkout or shared home cache.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(output / 'cache/jax')
    os.environ['WARP_CACHE_PATH'] = str(output / 'cache/warp')
    sys.dont_write_bytecode = True
    import mujoco
    import warp as wp
    wp.config.kernel_cache_dir = str(output / 'cache/warp')
    import jax
    from mujoco import mjx

    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('real MJX-Warp CUDA backend required')
    _write_json(output / 'declaration.json', spec)
    runtime = {
        'engine': 'MJX-Warp', 'python': sys.executable, 'python_version': sys.version,
        'python_prefix': sys.prefix, 'mujoco': mujoco.__version__,
        'mujoco_module': mujoco.__file__, 'mjx_module': mjx.__file__,
        'warp_version': wp.__version__, 'warp_module': wp.__file__,
        'jax_devices': [str(device) for device in jax.devices()],
        'model_sha256': hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        'packages': {},
        'input_semantics': 'absolute actuator ctrl, held piecewise constant; set before each native physical step',
        'sample_semantics': 'row 0 is initial state; row k ctrl caused transition from k-1 to k',
        'native_derived_arrays': 'raw MJX-Warp after step; RK4 sensor and actuator force arrays may belong to the last internal stage, not final integrated qpos/qvel',
        'final_angular_state': 'body omega=qvel[3:6]; world omega=R(final qpos quaternion)*body omega',
        'initialization': 'fresh complete MjData/keyframe, declared full qpos/qvel, one CPU mj_forward, then mjx.put_data; no CPU physical steps or later forward refresh',
        'training_interactions': 0,
    }
    for package in ['mujoco', 'mujoco-mjx', 'jax', 'jaxlib', 'warp-lang', 'numpy']:
        runtime['packages'][package] = importlib.metadata.version(package)
    _write_json(output / 'runtime.json', runtime)
    status = {'completed': False, 'engine': 'MJX-Warp', 'planned_physics_steps': sum(c['steps'] for c in cases),
              'attempted_physics_steps': 0, 'completed_physics_steps': 0,
              'initialization_physics_steps': 0, 'training_interactions': 0, 'cases': []}
    _write_json(output / 'status.json', status)
    step = jax.jit(mjx.step)
    started = time.monotonic()
    active_rows = active_meta = active_out = None
    try:
        for case in cases:
            active_out = output / case['name']
            active_out.mkdir()
            serializable = {key: value for key, value in case.items() if key != 'controls'}
            _write_json(active_out / 'case.json', serializable)
            m, d, meta = prepare_model(spec, case)
            _write_json(active_out / 'metadata.json', meta)
            mm = mjx.put_model(m, impl='warp')
            dd = mjx.put_data(m, d, impl='warp', naconmax=128, njmax=256)
            if 'Warp' not in type(dd._impl).__name__:
                raise RuntimeError(f'backend is not real Warp: {type(dd._impl)}')
            runtime['data_implementation'] = type(dd._impl).__name__
            _write_json(output / 'runtime.json', runtime)
            rows = [state_record(dd, m, meta, case['controls'][0], 0.)]
            active_rows, active_meta = rows, meta
            case_started = time.monotonic()
            for k, ctrl in enumerate(case['controls']):
                dd = dd.replace(ctrl=jax.numpy.asarray(ctrl, dtype=dd.ctrl.dtype))
                status['attempted_physics_steps'] += 1
                dd = step(mm, dd)
                jax.block_until_ready(dd)
                status['completed_physics_steps'] += 1
                record = state_record(dd, m, meta, ctrl, (k + 1) * case['physics_dt'])
                rows.append(record)
                if not np.isfinite(record['qpos']).all() or not np.isfinite(record['qvel']).all():
                    raise FloatingPointError(f'{case["name"]}: nonfinite state after step {k + 1}')
            arrays = save_trace(active_out / 'trace.npz', rows, meta)
            metrics = {'name': case['name'], 'physics_steps': len(rows) - 1,
                       'duration_s': float(arrays['time'][-1]),
                       'wall_s': time.monotonic() - case_started,
                       'gyro_vs_end_body_max_abs_rad_s': float(np.max(np.abs(arrays['gyro_body_native'] - arrays['omega_body_end']))),
                       'gyro_vs_end_body_rmse_rad_s': np.sqrt(np.mean((arrays['gyro_body_native'] - arrays['omega_body_end'])**2, axis=0)).tolist(),
                       'native_world_vs_end_world_max_abs_rad_s': float(np.max(np.abs(arrays['omega_world_native'] - arrays['omega_world_end']))),
                       'joint_interval_minus_end_qd_rmse_rad_s': np.sqrt(np.mean((arrays['joint_interval_velocity'][1:] - arrays['joint_qd'][1:])**2, axis=0)).tolist()}
            _write_json(active_out / 'summary.json', metrics)
            status['cases'].append(metrics)
            _write_json(output / 'status.json', status)
            print('CASE_COMPLETE', json.dumps(metrics), flush=True)
            active_rows = active_meta = active_out = None
        status['completed'] = True
    except BaseException:
        status['error'] = traceback.format_exc()
        if active_rows is not None:
            save_trace(active_out / 'trace_partial.npz', active_rows, active_meta)
        raise
    finally:
        status['wall_s'] = time.monotonic() - started
        _write_json(output / 'status.json', status)
    print(json.dumps({'output': str(output), **status}), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == '__main__':
    main()
