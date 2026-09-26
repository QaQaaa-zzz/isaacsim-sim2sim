"""Independent real-PhysX responses to shared, absolute open-loop inputs.

No Actor, learner, source trajectory replay, or Isaac startup on import. Each
candidate/case has a new USD stage and hard-reset solver. MuJoCo compiles only
the common model and initial state; it never advances the target trajectory.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
MAX_PHYSICS_STEPS = 30000
INITIALIZATION_RESERVE = 16


def validate_candidates(candidates, cases, actuator_names, *, budget=MAX_PHYSICS_STEPS):
    """Validate all target variations before loading Isaac; count rollout steps."""
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('candidates must be a nonempty list')
    if type(budget) is not int or not 1 <= budget <= MAX_PHYSICS_STEPS:
        raise ValueError('invalid physics budget')
    available = {case['name']: case for case in cases}
    seen, result, total = set(), [], 0
    for raw in candidates:
        if not isinstance(raw, dict) or set(raw) - {'name', 'physics_dt', 'drive_mode', 'target_drive_gains', 'case_names', 'phase_adapter'}:
            raise ValueError('unknown candidate fields')
        value = copy.deepcopy(raw)
        name = value.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in seen:
            raise ValueError('candidate names must be unique safe filenames')
        seen.add(name)
        dt = value.get('physics_dt')
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('physics_dt must be finite and positive')
        if value.get('drive_mode') not in ('hybrid', 'implicit', 'explicit', 'phase'):
            raise ValueError('invalid drive_mode')
        if value['drive_mode'] == 'phase':
            phase = value.get('phase_adapter')
            if not isinstance(phase, dict) or set(phase) != {'macro_dt', 'effective_inertias'}:
                raise ValueError('phase_adapter must declare macro_dt and effective_inertias')
            macro = phase['macro_dt']
            if isinstance(macro, bool) or not isinstance(macro, (int, float)) or not math.isfinite(macro) or macro <= 0:
                raise ValueError('phase macro_dt must be finite and positive')
            ratio = macro / dt
            count = round(ratio)
            if count < 2 or count % 2 or not math.isclose(ratio, count, rel_tol=0, abs_tol=1e-10):
                raise ValueError('phase macro_dt requires an even integer number of microsteps')
            inertias = phase['effective_inertias']
            if not isinstance(inertias, dict) or set(inertias) != {'rearwheel_joint', 'steering_joint'} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in inertias.values()
            ):
                raise ValueError('phase effective_inertias must be finite positive rear/steering values')
        elif 'phase_adapter' in value:
            raise ValueError('phase_adapter is unused unless drive_mode is phase')
        gains = value.get('target_drive_gains')
        if not isinstance(gains, dict) or set(gains) != set(actuator_names):
            raise ValueError('gains must identify exactly the four actuated joints')
        for joint, gain in gains.items():
            if not isinstance(gain, dict) or set(gain) != {'kp', 'kd'} or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                for v in gain.values()
            ):
                raise ValueError(f'invalid gains for {joint}')
        if gains['rearwheel_joint']['kp'] != 0:
            raise ValueError('rearwheel_joint gain kp must be zero for velocity targets')
        selected = value.get('case_names', list(available))
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) or any(n not in available for n in selected):
            raise ValueError('case_names must be unique known cases')
        value['case_names'] = selected
        for case_name in selected:
            if value['drive_mode'] == 'phase':
                macro = value['phase_adapter']['macro_dt']
                for segment in available[case_name]['segments']:
                    periods = segment['duration_s'] / macro
                    if not math.isclose(periods, round(periods), rel_tol=0, abs_tol=1e-9):
                        raise ValueError('phase input segments must align to macro periods')
            times, _ = control_timeline(available[case_name], dt)
            total += len(times) - 1
        result.append(value)
    if total > budget:
        raise ValueError(f'rollout physics budget exceeded: {total} > {budget}')
    return result, total


def control_timeline(case, dt):
    from response_protocol import expand_controls
    controls = expand_controls(case['segments'], dt)
    return np.arange(len(controls) + 1, dtype=float) * dt, np.vstack([controls[0], controls])


def initial_root_velocities(quat_wxyz, origin_velocity, body_omega, com_offset):
    rotation = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    world_omega = rotation.apply(body_omega)
    com_velocity = np.asarray(origin_velocity) + np.cross(world_omega, rotation.apply(com_offset))
    return com_velocity, world_omega


def root_velocities_from_physx(quat_wxyz, com_velocity, world_omega, com_offset):
    rotation = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    origin_velocity = np.asarray(com_velocity) - np.cross(world_omega, rotation.apply(com_offset))
    return origin_velocity, rotation.inv().apply(world_omega)


def drive_arrays(names, actuator_names, gains, enabled_joints, mode):
    kp, kd = np.zeros(len(names)), np.zeros(len(names))
    explicit = np.zeros(len(names), dtype=bool)
    for name in actuator_names:
        if name not in enabled_joints:
            continue
        index = names.index(name)
        explicit[index] = mode == 'explicit' or (mode in ('hybrid', 'phase') and name in ('hip_joint', 'knee_joint'))
        if not explicit[index] and mode != 'phase':
            kp[index], kd[index] = gains[name]['kp'], gains[name]['kd']
    return kp, kd, explicit


def explicit_efforts(names, actuator_names, gains, mask, q, qd, ctrl, force_ranges):
    efforts = np.zeros(len(names))
    for actuator, name in enumerate(actuator_names):
        index = names.index(name)
        if not mask[index]:
            continue
        kp, kd = gains[name]['kp'], gains[name]['kd']
        effort = kd * (ctrl[actuator] - qd[index]) if name == 'rearwheel_joint' else kp * (ctrl[actuator] - q[index]) - kd * qd[index]
        efforts[index] = np.clip(effort, *force_ranges[actuator])
    return efforts


def verify_force_caps(names, actuator_names, actual, force_ranges):
    actual = np.asarray(actual)
    for index, name in enumerate(actuator_names):
        if not np.isclose(force_ranges[index, 0], -force_ranges[index, 1], rtol=0, atol=1e-12):
            raise ValueError('source force caps must be symmetric')
        if not np.isclose(actual[names.index(name)], force_ranges[index, 1], rtol=1e-6, atol=1e-6):
            raise ValueError(f'runtime force cap differs from source for {name}')


def make_phase_adapter(names, actuator_names, candidate, enabled_joints, force_ranges):
    from drive_response_adapter import PhaseDriveAdapter
    phase = candidate['phase_adapter']
    limits = {name: float(force_ranges[i, 1]) for i, name in enumerate(actuator_names)}
    active = [name for name in ('rearwheel_joint', 'steering_joint') if name in enabled_joints]
    return PhaseDriveAdapter(names, actuator_names, phase['effective_inertias'], candidate['target_drive_gains'],
        limits, physics_dt=candidate['physics_dt'], macro_dt=phase['macro_dt'], active_joints=active)


def prepare_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=default, indent=2, allow_nan=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set_targets(view, names, actuator_names, ctrl, q_initial):
    positions = np.asarray(q_initial, dtype=np.float32).copy()
    velocities = np.zeros(len(names), dtype=np.float32)
    for index, name in enumerate(actuator_names):
        if name == 'rearwheel_joint':
            velocities[names.index(name)] = ctrl[index]
        else:
            positions[names.index(name)] = ctrl[index]
    view.set_joint_position_targets(positions[None])
    view.set_joint_velocity_targets(velocities[None])


def run_case(spec, case, candidate, output, physics_events, budget):
    """Initialize once, then advance only PhysX; preserve every finite sample."""
    import omni.usd
    from pxr import UsdPhysics, PhysxSchema, Gf
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from response_protocol import prepare_model
    from usd_model import build
    from physx_training_math import continuous_joint_position

    started = time.monotonic()
    dt = candidate['physics_dt']
    times, controls = control_timeline(case, dt)
    case = {**case, 'physics_dt': dt, 'controls': controls[1:], 'steps': len(times) - 1}
    m, initial, metadata = prepare_model(spec, case, physics_dt=dt)
    out = prepare_output(output)
    write_json(out / 'declaration.json', {'protocol': spec, 'case': case, 'candidate': candidate,
        'source_model': metadata, 'training_transitions': 0,
        'state_semantics': 'qpos/qvel in original source order; qvel root angular coordinates body-local; other vector channels explicitly named',
        'control_semantics': 't0 contains first input; row k>0 contains input applied over the just completed physical step',
        'force_semantics': 'explicit_effort_applied is prescribed motor input; dof_actuation_force_readback and projected_joint_effort are native PhysX readbacks, not isolated native-drive motor torque',
        'physics_steps_expected': len(times) - 1, 'initialization_physics_step_reserve': INITIALIZATION_RESERVE})
    write_json(out / 'status.json', {'status': 'initializing', 'training_transitions': 0})
    rows, phase_audits = [], []
    status = {'status': 'error', 'candidate': candidate['name'], 'case': case['name'], 'training_transitions': 0}
    start_events, initialization_events = len(physics_events), None
    completed, attempted = 0, 0
    world = None
    try:
        if len(physics_events) + INITIALIZATION_RESERVE + len(times) - 1 > budget:
            raise RuntimeError('remaining physics budget is insufficient for case plus initialization reserve')
        World.clear_instance()
        omni.usd.get_context().new_stage()
        world = World(physics_dt=dt, rendering_dt=.02, backend='numpy')
        context = world.get_physics_context()
        context.enable_fabric(False)
        stage = omni.usd.get_context().get_stage()
        _, conversion = build(stage, Path(metadata['source_xml']), target_drive_gains=candidate['target_drive_gains'])
        gravity = np.asarray(spec['gravity'], dtype=float)
        gravity_magnitude = float(np.linalg.norm(gravity))
        scene = UsdPhysics.Scene(context.get_current_physics_scene_prim())
        scene.CreateGravityDirectionAttr(Gf.Vec3f(*map(float, gravity / gravity_magnitude if gravity_magnitude else [0, 0, -1])))
        scene.CreateGravityMagnitudeAttr(gravity_magnitude)
        material_audit, collision_audit = [], []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                api.CreateCollisionEnabledAttr(spec['contact_enabled'])
                collision_audit.append({'path': str(prim.GetPath()), 'enabled': bool(api.GetCollisionEnabledAttr().Get())})
            if prim.HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(prim)
                name = prim.GetName()
                geometry = int(name.split('_')[-1]) if name.startswith('material_') else -1
                if geometry >= 0 and 'wheel' in m.geom(geometry).name:
                    api.CreateStaticFrictionAttr(.5)
                    api.CreateDynamicFrictionAttr(.5)
                mode = PhysxSchema.PhysxMaterialAPI.Apply(prim)
                mode.CreateFrictionCombineModeAttr('min')
                material_audit.append({'path': str(prim.GetPath()), 'static_friction': api.GetStaticFrictionAttr().Get(),
                    'dynamic_friction': api.GetDynamicFrictionAttr().Get(), 'combine': mode.GetFrictionCombineModeAttr().Get()})
        view = world.scene.add(Articulation('/World/Bike', name='response_probe'))
        world.reset(soft=False)
        names = list(view.dof_names)
        if set(names) != set(metadata['joint_names']):
            raise RuntimeError('PhysX/source hinge joint names disagree')
        qa = np.array([int(m.joint(n).qposadr[0]) for n in names])
        va = np.array([int(m.joint(n).dofadr[0]) for n in names])
        source_order = np.array([names.index(n) for n in metadata['joint_names']])
        actuator_names = metadata['actuator_joint_names']
        kp, kd, explicit = drive_arrays(names, actuator_names, candidate['target_drive_gains'], case['enabled_joints'], candidate['drive_mode'])
        view.set_gains(kps=kp[None].astype(np.float32), kds=kd[None].astype(np.float32))
        actual_kp, actual_kd = view.get_gains()
        np.testing.assert_allclose(actual_kp[0], kp, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(actual_kd[0], kd, rtol=1e-6, atol=1e-8)
        max_efforts = view.get_max_efforts()
        verify_force_caps(names, actuator_names, max_efforts[0], m.actuator_forcerange)
        phase_adapter = make_phase_adapter(names, actuator_names, candidate, case['enabled_joints'], m.actuator_forcerange) if candidate['drive_mode'] == 'phase' else None
        q_initial = initial.qpos[qa].copy()
        com_velocity, omega_world = initial_root_velocities(initial.qpos[3:7], initial.qvel[:3], initial.qvel[3:6], m.body_ipos[1])
        view.set_world_poses(initial.qpos[None, :3].astype(np.float32), initial.qpos[None, 3:7].astype(np.float32))
        view.set_velocities(np.r_[com_velocity, omega_world][None].astype(np.float32))
        view.set_joint_positions(q_initial[None].astype(np.float32))
        view.set_joint_velocities(initial.qvel[None, va].astype(np.float32))
        _set_targets(view, names, actuator_names, controls[0], q_initial)
        view.set_joint_efforts(np.zeros((1, len(names)), dtype=np.float32))
        world.physics_sim_view.update_articulations_kinematic()
        engine_time_start = float(world.current_time)
        initialization_events = len(physics_events) - start_events
        if initialization_events > INITIALIZATION_RESERVE:
            raise RuntimeError('initialization exceeded its reserved physics step budget')
        if not math.isclose(world.get_physics_dt(), dt, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError('runtime physics_dt differs from the declaration')
        write_json(out / 'runtime_audit.json', {'joint_names_physx': names, 'joint_names_source': metadata['joint_names'],
            'gains_native': [actual_kp, actual_kd], 'explicit_joint_mask': explicit, 'max_efforts': max_efforts,
            'source_force_caps_verified': True, 'engine_time_origin_s': engine_time_start,
            'gravity_direction': list(scene.GetGravityDirectionAttr().Get()), 'gravity_magnitude': scene.GetGravityMagnitudeAttr().Get(),
            'colliders': collision_audit, 'materials': material_audit, 'conversion_before_contact_override': conversion,
            'initialization_physics_steps': initialization_events, 'physics_dt': world.get_physics_dt(),
            'reset': 'new stage, new World, hard reset, full initial pose and velocity assignment; no subsequent state injection'})
        wheels = np.array([names.index(n) for n in ('frontwheel_joint', 'rearwheel_joint')])
        previous_wheels, continuous_wheels = q_initial[wheels].copy(), q_initial[wheels].copy()
        last_joint_q = None
        last_effort = np.zeros(len(names))
        for k, (sample_time, ctrl) in enumerate(zip(times, controls)):
            if k:
                if len(physics_events) >= budget:
                    raise RuntimeError('physics budget exhausted before next step')
                q = view.get_joint_positions()[0].copy()
                qd = view.get_joint_velocities()[0].copy()
                _set_targets(view, names, actuator_names, ctrl, q_initial)
                last_effort = explicit_efforts(names, actuator_names, candidate['target_drive_gains'], explicit,
                    q, qd, ctrl, m.actuator_forcerange)
                if phase_adapter is not None:
                    phase_effort, phase_audit = phase_adapter.compute_efforts(k - 1, q, qd, ctrl)
                    last_effort += phase_effort
                    phase_audits.append({'step_index': k - 1, 'time_before_s': times[k - 1],
                        'q_before_physx_order': q.copy(), 'qd_before_physx_order': qd.copy(),
                        'ctrl': ctrl.copy(), 'effort_applied_physx_order': last_effort.copy(), **phase_audit})
                view.set_joint_efforts(last_effort[None].astype(np.float32))
                attempted += 1
                event_before = len(physics_events)
                world.step(render=False)
                completed += 1
                if len(physics_events) - event_before != 1:
                    raise RuntimeError('World.step did not produce exactly one physics event')
                if not math.isclose(physics_events[-1], dt, rel_tol=1e-6, abs_tol=1e-10):
                    raise RuntimeError('actual physics event timestep differs from declaration')
            positions, quaternions = view.get_world_poses()
            pos, quat = positions[0].copy(), quaternions[0].copy()
            q, qd = view.get_joint_positions()[0].copy(), view.get_joint_velocities()[0].copy()
            world_omega = view.get_angular_velocities()[0].copy()
            origin_velocity, body_omega = root_velocities_from_physx(quat, view.get_linear_velocities()[0], world_omega, m.body_ipos[1])
            raw = np.r_[pos, quat, q, qd, world_omega, origin_velocity]
            if not np.isfinite(raw).all():
                np.savez_compressed(out / 'nonfinite_state.npz', time=sample_time, raw_state=raw)
                raise RuntimeError('nonfinite PhysX state; stopped at first observed physical step')
            continuous_wheels = continuous_joint_position(previous_wheels, q[wheels], continuous_wheels)
            previous_wheels = q[wheels].copy()
            q[wheels] = continuous_wheels
            qpos, qvel = np.empty(m.nq), np.empty(m.nv)
            qpos[:3], qpos[3:7], qpos[qa] = pos, quat, q
            qvel[:3], qvel[3:6], qvel[va] = origin_velocity, body_omega, qd
            joint_q, joint_qd = q[source_order], qd[source_order]
            interval_velocity = np.full(len(names), np.nan) if last_joint_q is None else (joint_q - last_joint_q) / dt
            last_joint_q = joint_q.copy()
            projected = view.get_measured_joint_efforts()[0].copy()
            actuation = view._physics_view.get_dof_actuation_forces()[0].copy()
            if not np.isfinite(np.r_[projected, actuation, last_effort]).all():
                raise RuntimeError('nonfinite effort readback')
            engine_time = float(world.current_time) - engine_time_start
            if not math.isclose(engine_time, sample_time, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError('engine time differs from recorded input timeline')
            rows.append({'time': sample_time, 'engine_time': engine_time, 'qpos': qpos, 'qvel': qvel, 'joint_q': joint_q, 'joint_qd': joint_qd,
                'joint_interval_velocity': interval_velocity, 'omega_world_end': world_omega, 'omega_body_end': body_omega,
                'root_velocity_origin': origin_velocity, 'ctrl': ctrl.copy(), 'explicit_effort_applied': last_effort[source_order].copy(),
                'dof_actuation_force_readback': actuation[source_order], 'projected_joint_effort': projected[source_order]})
        status.update(status='complete', endpoint='declared_duration')
    except Exception:
        status.update(status='error', endpoint='probe_error', error=traceback.format_exc())
    finally:
        case_events = physics_events[start_events:]
        if initialization_events is None:
            initialization_events = len(case_events)
        if rows:
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            arrays['time_s'] = arrays['time'].copy()
            arrays['joint_names'] = np.asarray(metadata['joint_names'])
            np.savez_compressed(out / 'traces.npz', **arrays)
        status.update(recorded_samples=len(rows), recorded_transition_rows=max(0, len(rows) - 1),
            physics_steps_completed=completed, physics_steps_attempted=attempted,
            initialization_physics_steps=initialization_events, rollout_physics_events=len(case_events) - initialization_events,
            total_physics_steps=len(case_events), duration_s=float(rows[-1]['time']) if rows else 0., wall_s=time.monotonic() - started)
        write_json(out / 'physics_steps.json', {'dt_s': case_events})
        if candidate['drive_mode'] == 'phase':
            write_json(out / 'phase_adapter_steps.json', phase_audits)
        write_json(out / 'status.json', status)
        if world is not None:
            world.stop()
            world.clear()
        World.clear_instance()
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-physics-steps', type=int, default=MAX_PHYSICS_STEPS)
    args = parser.parse_args(argv)
    from response_protocol import load_protocol, prepare_model, ACTUATED_JOINTS
    spec, cases = load_protocol(args.config)
    candidates, rollout_total = validate_candidates(json.loads(args.candidates.read_text()), cases, ACTUATED_JOINTS, budget=args.max_physics_steps)
    case_count = sum(len(c['case_names']) for c in candidates)
    if rollout_total + INITIALIZATION_RESERVE * case_count > args.max_physics_steps:
        raise ValueError('physics budget must also cover each case initialization reserve')
    by_name = {case['name']: case for case in cases}
    out = prepare_output(args.output)
    files = [ROOT / name for name in ('probe_physx_response.py', 'response_protocol.py', 'usd_model.py',
        'drive_calibration.py', 'collision_geometry.py', 'physx_training_math.py', 'model/source.xml')]
    if any(candidate['drive_mode'] == 'phase' for candidate in candidates):
        files.append(ROOT / 'drive_response_adapter.py')
    versions = {}
    for name in ('isaacsim', 'mujoco', 'numpy', 'scipy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(out / 'declaration.json', {'backend': 'Isaac Sim real PhysX CPU articulation', 'versions': versions,
        'python': sys.executable, 'config_path': str(args.config.resolve()), 'config_sha256': sha256(args.config),
        'candidates_path': str(args.candidates.resolve()), 'candidates_sha256': sha256(args.candidates),
        'source_hashes': {str(p): sha256(p) for p in files}, 'protocol': spec, 'candidates': candidates,
        'rollout_steps_requested': rollout_total, 'initialization_reserve': INITIALIZATION_RESERVE * case_count,
        'max_physics_steps': args.max_physics_steps, 'training_transitions': 0})
    write_json(out / 'status.json', {'status': 'initializing', 'training_transitions': 0})
    os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1')
    sys.dont_write_bytecode = True
    app, subscription = None, None
    physics_events, results = [], []
    status = {'status': 'error', 'training_transitions': 0}
    try:
        import faulthandler
        faulthandler.dump_traceback_later(60, repeat=True)
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'disable_viewport_updates': True, 'limit_cpu_threads': 4,
            'extra_args': ['--/app/settings/loadUserConfig=false', '--/app/settings/persistent=false']})
        faulthandler.cancel_dump_traceback_later()
        # MuJoCo is loaded only after Kit startup, as in the existing working
        # PhysX entrypoints. These checks still perform zero physics steps.
        for candidate in candidates:
            for name in candidate['case_names']:
                _, controls = control_timeline(by_name[name], candidate['physics_dt'])
                prepare_model(spec, {**by_name[name], 'controls': controls[1:]}, physics_dt=candidate['physics_dt'])
        from omni.physx import get_physx_interface
        subscription = get_physx_interface().subscribe_physics_step_events(lambda dt: physics_events.append(float(dt)))
        for candidate in candidates:
            for name in candidate['case_names']:
                result = run_case(spec, by_name[name], candidate, out / candidate['name'] / name, physics_events, args.max_physics_steps)
                results.append(result)
                write_json(out / 'summary.json', results)
                write_json(out / 'status.json', {'status': 'running', 'cases_completed': sum(r['status'] == 'complete' for r in results),
                    'total_physics_steps': len(physics_events), 'training_transitions': 0})
                print('CASE_COMPLETE', candidate['name'], name, json.dumps(result), flush=True)
                if result['status'] != 'complete':
                    raise RuntimeError(f"case failed: {candidate['name']}/{name}")
        status.update(status='complete')
    except Exception:
        status.update(status='error', error=traceback.format_exc())
    finally:
        status.update(cases_requested=case_count, cases_completed=sum(r['status'] == 'complete' for r in results),
            rollout_physics_steps=sum(r['physics_steps_completed'] for r in results),
            initialization_physics_steps=sum(r['initialization_physics_steps'] for r in results),
            total_physics_steps=len(physics_events))
        write_json(out / 'status.json', status)
        if subscription is not None:
            subscription.unsubscribe()
        if app is not None:
            app.close()
    print(json.dumps(status), flush=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
