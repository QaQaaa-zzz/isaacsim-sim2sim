"""Bounded, recorded, single-environment migration diagnostic; never trains.

Importing this module does not start Isaac. Run each case in a fresh process so
the solver and contact state cannot leak between the declared gain variants.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SEED = 1820701
MAX_PHYSICS_SUBSTEPS = 2000
ACTUATOR_MODE = {
    'rearwheel_joint': 'native_implicit',
    'steering_joint': 'native_implicit',
    'hip_joint': 'explicit_PD_capped',
    'knee_joint': 'explicit_PD_capped',
}


def validate_probe(spec, steps, seed):
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError('steps must be an integer in [1, 100]')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    physics_dt, control_dt = spec['physics_dt'], spec['control_dt']
    for name, value in [('physics_dt', physics_dt), ('control_dt', control_dt)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if steps * control_dt > 2.0 + 1e-12:
        raise ValueError('probe duration must not exceed 2 seconds')
    ratio = control_dt / physics_dt
    substeps = round(ratio)
    if substeps < 1 or not math.isclose(ratio, substeps, rel_tol=0, abs_tol=1e-9):
        raise ValueError('control_dt must be an integer multiple of physics_dt')
    if steps * substeps > MAX_PHYSICS_SUBSTEPS:
        raise ValueError(f'probe exceeds {MAX_PHYSICS_SUBSTEPS} physics substeps')
    if spec.get('actuator_mode') != ACTUATOR_MODE:
        raise ValueError('actuator_mode must exactly match the implemented native rear/steer and capped explicit hip/knee modes')
    return substeps


def validate_runtime_config(spec, frozen):
    """Reject behavioral settings the reused task would silently disregard."""
    for key in ('reward', 'reset', 'action', 'events', 'physical_limits', 'model'):
        if key in spec and spec[key] != frozen.get(key):
            raise ValueError(f'{key} differs from policy/resolved_config.json used by PhysxTask')
    if 'episode_horizon' in spec and spec['episode_horizon'] != frozen['ppo']['episode_horizon']:
        raise ValueError('episode_horizon differs from the frozen task horizon')
    if spec.get('backend_required', 'PhysX') != 'PhysX':
        raise ValueError('backend_required must be PhysX')
    if type(spec.get('source_ppo_port', False)) is not bool:
        raise ValueError('source_ppo_port must be boolean')
    if 'target_drive_gains' in spec:
        gains = spec['target_drive_gains']
        if set(gains) != set(ACTUATOR_MODE):
            raise ValueError('target_drive_gains must declare exactly four actuated joints')
        for joint, gain in gains.items():
            if set(gain) != {'kp', 'kd'} or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in gain.values()):
                raise ValueError(f'invalid target_drive_gains for {joint}')
        if gains['rearwheel_joint']['kp'] != 0:
            raise ValueError('rearwheel_joint is a velocity drive and requires kp=0')
    if 'contact_required' in spec:
        contact = spec['contact_required']
        for key, value in {'static_friction': .5, 'dynamic_friction': .5, 'combine': 'min'}.items():
            if contact.get(key) != value:
                raise ValueError(f'contact_required.{key} would change the frozen .5/min contract')
    # Training metadata may be carried with the config but is explicitly excluded
    # in declaration.json: this entrypoint never invokes a learner or evaluator.


def prepare_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def angular_state(view):
    _, orientation = view.get_world_poses()
    quat = np.asarray(orientation, dtype=float)[0].copy()
    world = np.asarray(view.get_angular_velocities(), dtype=float)[0].copy()
    body = Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply(world)
    return {'quaternion_wxyz': quat, 'world_rad_s': world, 'body_rad_s': body}


def run_rollout(env, actor, *, steps, control_dt, substeps):
    """Keep the true terminal transition, count actual World.step completions.

    On a physics/interface exception, return the completed prefix and error so
    the caller can preserve evidence before exiting unsuccessfully.
    """
    validate_probe({'physics_dt': control_dt / substeps, 'control_dt': control_dt,
                    'actuator_mode': ACTUATOR_MODE}, steps, SEED)
    if env.n != 1:
        raise ValueError('migration probe requires exactly one environment')
    records = {key: [] for key in ('target', 'observations', 'next_observations',
                                  'actions', 'rewards', 'done', 'omega_world',
                                  'omega_body', 'orientation_wxyz', 'step_wall_s')}
    infos = []
    completed = attempted = 0
    original_step = env.world.step
    endpoint_reason = 'engineering_cap'
    error = None
    started = time.monotonic()

    def counted_step(*args, **kwargs):
        nonlocal completed, attempted
        if attempted >= steps * substeps:
            raise RuntimeError('physics substeps budget exhausted')
        attempted += 1
        value = original_step(*args, **kwargs)
        completed += 1
        return value

    env.world.step = counted_step
    try:
        for k in range(steps):
            tick = time.monotonic()
            before = np.asarray(env.obs).copy()
            action = np.asarray(actor(before), dtype=np.float32)
            if before.shape != (1, 76) or not np.isfinite(before).all():
                raise RuntimeError('invalid pre-step actor observation')
            if action.shape != (1, 4) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise RuntimeError('invalid normalized actor action')
            previous_count = completed
            observation, reward, done, info = env.step(action)
            if completed - previous_count != substeps:
                raise RuntimeError('actual physics substeps differ from the validated control period')
            trace = np.asarray(env.trace[0], dtype=float).copy()
            if trace.shape != (31,):
                raise RuntimeError(f'invalid trace shape {trace.shape}')
            trace[0] = (k + 1) * control_dt
            observation = np.asarray(observation).copy()
            reward = np.asarray(reward)
            done = np.asarray(done)
            if observation.shape != (1, 76) or reward.shape != (1,) or done.shape != (1,):
                raise RuntimeError('invalid post-step observation/reward/done shape')
            if not all(np.isfinite(v).all() for v in (trace, observation, reward)):
                raise RuntimeError('nonfinite post-step state')
            if hasattr(env, 'view'):
                omega = angular_state(env.view)
                if not np.allclose(omega['world_rad_s'], trace[21:24], atol=1e-6, rtol=1e-6):
                    raise RuntimeError('trace/runtime world angular velocity mismatch')
            else:
                # Pure recording tests use the same documented trace layout.
                quat = trace[4:8]
                omega = {'quaternion_wxyz': quat, 'world_rad_s': trace[21:24],
                         'body_rad_s': Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply(trace[21:24])}
            if not all(np.isfinite(v).all() for v in omega.values()):
                raise RuntimeError('nonfinite angular state')
            values = (trace, before[0], observation[0], action[0].copy(), float(reward[0]),
                      bool(done[0]), omega['world_rad_s'], omega['body_rad_s'],
                      omega['quaternion_wxyz'], time.monotonic() - tick)
            for key, value in zip(records, values):
                records[key].append(value)
            infos.append(info[0])
            if done[0]:
                endpoint_reason = 'environment_done'
                break
    except Exception:
        endpoint_reason = 'probe_error'
        error = traceback.format_exc()
    finally:
        env.world.step = original_step
    return {'arrays': {k: np.asarray(v) for k, v in records.items()}, 'info': infos,
            'control_steps': len(infos), 'physics_substeps': completed,
            'physics_substeps_attempted': attempted, 'endpoint_reason': endpoint_reason,
            'simulated_duration_s': completed * control_dt / substeps,
            'sampling_wall_s': time.monotonic() - started, 'error': error}


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--actor', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args(argv)
    started = time.monotonic()
    spec = json.loads(args.config.read_text())
    frozen = json.loads((ROOT / 'policy/resolved_config.json').read_text())
    substeps = validate_probe(spec, args.steps, SEED)
    validate_runtime_config(spec, frozen)
    for key in ('target_drive_gains', 'contact_required'):
        if key not in spec:
            raise ValueError(f'missing required runtime configuration: {key}')
    if not args.actor.is_file():
        raise FileNotFoundError(args.actor)
    output = prepare_output(args.output)
    files = [ROOT / name for name in ('probe_physx_migration.py', 'physx_task.py',
             'policy_runtime.py', 'physx_training_math.py', 'source_reset.py',
             'usd_model.py', 'drive_calibration.py', 'collision_geometry.py',
             'policy/resolved_config.json')]
    source_root = Path('/home/qy/DVGC/JIT/src/jit_dvgc')
    files += [source_root / name for name in ('config.py', 'rewards.py', 'semantics.py')]
    files += [p for p in (ROOT / 'model').rglob('*') if p.is_file()]
    declaration = {
        'role': 'bounded engineering migration diagnostic; no training or task qualification',
        'config_path': str(args.config.resolve()), 'config_sha256': sha256(args.config),
        'actor_path': str(args.actor.resolve()), 'actor_sha256': sha256(args.actor),
        'source_and_model_sha256': {str(p): sha256(p) for p in files},
        'spec': spec, 'num_envs': 1, 'seed': SEED, 'ground_only': True,
        'reset': 'new process, hard solver reset, keyframe ground reset with zero joint velocity',
        'initial_forward_velocity': frozen['reset']['initial_forward_velocity'],
        'requested_control_steps': args.steps, 'maximum_duration_s': args.steps * spec['control_dt'],
        'physics_substeps_per_control_step': substeps,
        'rollout_physics_substep_budget': args.steps * substeps,
        'training_transitions': 0, 'stop': 'first true environment done or declared engineering cap',
        'ignored_training_metadata': ['ppo', 'best_selection', 'initial_actor', 'initial_checkpoint',
                                     'initialization', 'requested_training_transitions', 'launch_ready'],
        'trace_layout': {'time_s': [0, 1], 'root_xyz_m': [1, 4], 'root_wxyz': [4, 8],
                         'q_rad': [8, 13], 'qd_rad_s': [13, 18], 'root_velocity_m_s': [18, 21],
                         'world_omega_rad_s': [21, 24], 'acceleration_m_s2': [24, 27],
                         'control_targets': [27, 31]},
        'simulation_app': {'headless': True, 'disable_viewport_updates': True, 'limit_cpu_threads': 4},
    }
    write_json(output / 'declaration.json', declaration)
    write_json(output / 'status.json', {'status': 'initializing', 'training_transitions': 0})
    os.environ.update(JAX_PLATFORMS='cpu', OMNI_KIT_ACCEPT_EULA='YES', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    app = None
    subscription = None
    physics_events = []
    status = {'status': 'error', 'training_transitions': 0}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({**declaration['simulation_app'], 'extra_args': [
            '--/app/settings/loadUserConfig=false', '--/app/settings/persistent=false']})
        from omni.physx import get_physx_interface
        subscription = get_physx_interface().subscribe_physics_step_events(
            lambda dt: physics_events.append(float(dt)))
        from physx_task import PhysxTask
        from policy_runtime import Actor
        actor = Actor(args.actor)
        env = PhysxTask(spec, 1, SEED)
        env.hard_reset()
        env.reset(np.array([0]), seeds=[SEED], ground_only=True)
        if not math.isclose(env.world.get_physics_dt(), spec['physics_dt'], rel_tol=0, abs_tol=1e-12):
            raise RuntimeError('runtime physics_dt does not match the declaration')
        initialization_events = len(physics_events)
        write_json(output / 'runtime_audit.json', {
            **env.audit, 'initial_angular_state': angular_state(env.view),
            'initial_observation': env.obs, 'initialization_physics_substeps': initialization_events,
            'initialization_physics_dts_s': physics_events.copy(),
            'source_config_fields_verified': ['reward', 'reset', 'episode_horizon'],
        })
        initialization_s = time.monotonic() - started
        result = run_rollout(env, actor, steps=args.steps,
                             control_dt=spec['control_dt'], substeps=substeps)
        np.savez_compressed(output / 'traces.npz', **result['arrays'],
                            joint_names=np.asarray(env.names), seed=np.asarray(SEED))
        write_json(output / 'steps.json', result['info'])
        status = {k: v for k, v in result.items() if k not in ('arrays', 'info')}
        status.update(status='complete' if result['error'] is None else 'error',
                      initialization_s=initialization_s, training_transitions=0,
                      initialization_physics_substeps=initialization_events,
                      rollout_physics_events=len(physics_events) - initialization_events,
                      total_physics_substeps=len(physics_events),
                      terminal_info=result['info'][-1] if result['info'] else None,
                      data_role='engineering diagnostic; environment_done may represent physical failure')
        if status['rollout_physics_events'] != status['physics_substeps']:
            status.update(status='error', error='PhysX event count and World.step completion count disagree')
    except Exception:
        status.update(status='error', error=traceback.format_exc(), total_physics_substeps=len(physics_events))
    finally:
        status['total_wall_s_before_close'] = time.monotonic() - started
        write_json(output / 'physics_steps.json', {'dt_s': physics_events})
        write_json(output / 'status.json', status)
        if subscription is not None:
            subscription.unsubscribe()
        if app is not None:
            app.close()
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
