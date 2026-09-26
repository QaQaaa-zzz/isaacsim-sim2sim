"""Shared declarations for real MJX-Warp and PhysX open-loop responses.

No engine is started on import.  The isolated source copy is the only permitted
model; qpos, qvel, input units, and the exact piecewise-constant input timeline
are shared.  Contact-free fixtures are explicit diagnostic cases, not training.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_MODEL = ROOT / 'model/source.xml'
MAX_PHYSICS_STEPS = 2000
MAX_CASE_DURATION_S = .2
ACTUATED_JOINTS = ['steering_joint', 'rearwheel_joint', 'hip_joint', 'knee_joint']
ACTUATOR_NAMES = ['cmd_steering_v', 'cmd_rearwheel_f', 'cmd_hip_f', 'cmd_knee_f']


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f'{name} must contain {length} numbers')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f'{name} must contain only finite numbers')
    return [float(v) for v in value]


def _initial(initial):
    if not isinstance(initial, dict) or set(initial) - {'qpos', 'qvel', 'keyframe'}:
        raise ValueError('initial must specify qpos/qvel, with optional keyframe')
    qpos = _vector(initial.get('qpos'), 12, 'initial.qpos')
    qvel = _vector(initial.get('qvel'), 11, 'initial.qvel')
    if not math.isclose(float(np.linalg.norm(qpos[3:7])), 1., rel_tol=0, abs_tol=1e-7):
        raise ValueError('initial root quaternion wxyz must have unit norm')
    keyframe = initial.get('keyframe', 0)
    if type(keyframe) is not int or keyframe != 0:
        raise ValueError('only source keyframe 0 is supported')
    return {'keyframe': keyframe, 'qpos': qpos, 'qvel': qvel}


def expand_controls(segments, dt):
    """One ctrl per physical step, applied before integrating that step."""
    dt = _positive(dt, 'physics_dt')
    if not isinstance(segments, list) or not segments:
        raise ValueError('segments must be a nonempty list')
    controls = []
    duration = 0.
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {'duration_s', 'ctrl'}:
            raise ValueError('each segment must contain exactly duration_s and ctrl')
        segment_duration = _positive(segment['duration_s'], 'duration_s')
        steps = round(segment_duration / dt)
        if steps < 1 or not math.isclose(steps * dt, segment_duration, rel_tol=0, abs_tol=1e-10):
            raise ValueError('each duration_s must be an integer multiple of physics_dt')
        control = _vector(segment['ctrl'], 4, 'ctrl')
        duration += segment_duration
        if duration > MAX_CASE_DURATION_S + 1e-10 or len(controls) + steps > MAX_PHYSICS_STEPS:
            raise ValueError(f'case exceeds {MAX_CASE_DURATION_S} seconds or {MAX_PHYSICS_STEPS} physics steps')
        controls.extend([control] * steps)
    return np.asarray(controls, dtype=float)


def validate_protocol(spec, *, physics_dt=None):
    """Validate without importing MuJoCo; return independent normalized cases."""
    if not isinstance(spec, dict) or spec.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')
    source = Path(spec.get('model_path', SOURCE_MODEL)).expanduser().resolve()
    if source != SOURCE_MODEL.resolve():
        raise ValueError('model_path must be the isolated model/source.xml copy')
    declared_dt = _positive(spec.get('physics_dt'), 'physics_dt')
    dt = declared_dt if physics_dt is None else _positive(physics_dt, 'physics_dt override')
    if spec.get('integrator') not in {'RK4', 'EULER', 'IMPLICITFAST'}:
        raise ValueError('integrator must be RK4, EULER, or IMPLICITFAST')
    _vector(spec.get('gravity'), 3, 'gravity')
    if type(spec.get('contact_enabled')) is not bool:
        raise ValueError('contact_enabled must be boolean')
    top_initial = _initial(spec['initial']) if 'initial' in spec else None
    if not isinstance(spec.get('cases'), list) or not spec['cases']:
        raise ValueError('cases must be a nonempty list')
    cases, names, total_steps = [], set(), 0
    for raw in spec['cases']:
        if not isinstance(raw, dict):
            raise ValueError('each case must be an object')
        name = raw.get('name')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in names:
            raise ValueError('case names must be unique safe filenames')
        names.add(name)
        initial = _initial(raw['initial']) if 'initial' in raw else copy.deepcopy(top_initial)
        if initial is None:
            raise ValueError('each case needs full initial qpos/qvel, directly or at top level')
        enabled = raw.get('enabled_joints', ACTUATED_JOINTS)
        if not isinstance(enabled, list) or any(not isinstance(j, str) for j in enabled) or len(enabled) != len(set(enabled)) or any(j not in ACTUATED_JOINTS for j in enabled):
            raise ValueError('enabled_joints must be unique actuated source joint names')
        controls = expand_controls(raw.get('segments'), dt)
        total_steps += len(controls)
        if total_steps > MAX_PHYSICS_STEPS:
            raise ValueError(f'protocol exceeds {MAX_PHYSICS_STEPS} total physics steps')
        cases.append({**copy.deepcopy(raw), 'initial': initial,
                      'enabled_joints': list(enabled), 'physics_dt': dt,
                      'steps': len(controls), 'controls': controls})
    return cases


def load_protocol(path, *, physics_dt=None):
    spec = json.loads(Path(path).read_text())
    return spec, validate_protocol(spec, physics_dt=physics_dt)


def prepare_model(spec, case, *, physics_dt=None):
    """Compile a fresh model and reset full state; performs zero physics steps.

    CPU mj_forward initializes derived fields once before transfer to Warp.  It
    never advances or replaces any subsequently measured Warp trajectory.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    dt = case.get('physics_dt', spec['physics_dt']) if physics_dt is None else physics_dt
    m.opt.timestep = _positive(dt, 'physics_dt')
    m.opt.integrator = getattr(mujoco.mjtIntegrator, 'mjINT_' + spec['integrator'])
    m.opt.gravity[:] = spec['gravity']
    if not spec['contact_enabled']:
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    joint_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    joint_names = [m.joint(j).name for j in joint_ids]
    qa = [int(m.jnt_qposadr[j]) for j in joint_ids]
    va = [int(m.jnt_dofadr[j]) for j in joint_ids]
    actuator_names = [m.actuator(i).name for i in range(m.nu)]
    actuator_joint_names = [m.joint(int(j)).name for j in m.actuator_trnid[:, 0]]
    if actuator_names != ACTUATOR_NAMES or actuator_joint_names != ACTUATED_JOINTS or (m.nq, m.nv) != (12, 11):
        raise ValueError('isolated source model topology differs from declared protocol')
    for i, name in enumerate(actuator_joint_names):
        if name not in case['enabled_joints']:
            m.actuator_gainprm[i] = 0.
            m.actuator_biasprm[i] = 0.
    controls = np.asarray(case['controls'])
    limited = np.asarray(m.actuator_ctrllimited, dtype=bool)
    if np.any(controls[:, limited] < m.actuator_ctrlrange[limited, 0]) or np.any(controls[:, limited] > m.actuator_ctrlrange[limited, 1]):
        raise ValueError('ctrl values exceed the original actuator ctrlrange')
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, case['initial'].get('keyframe', 0))
    d.qpos[:] = case['initial']['qpos']
    d.qvel[:] = case['initial']['qvel']
    d.ctrl[:] = controls[0]
    mujoco.mj_forward(m, d)
    metadata = {
        'source_xml': str(SOURCE_MODEL),
        'model_sha256': hashlib.sha256(SOURCE_MODEL.read_bytes()).hexdigest(),
        'joint_names': joint_names, 'qa': qa, 'va': va,
        'actuator_names': actuator_names, 'actuator_joint_names': actuator_joint_names,
        'sensor_names': [m.sensor(i).name for i in range(m.nsensor)],
        'sensor_adr': m.sensor_adr.tolist(), 'sensor_dim': m.sensor_dim.tolist(),
        'ctrl_units': ['rad', 'rad/s', 'rad', 'rad'],
        'root_qpos_order': 'x,y,z,qw,qx,qy,qz',
        'root_qvel_order': 'world linear xyz, body angular xyz',
        'physics_dt': float(m.opt.timestep), 'integrator': spec['integrator'],
        'contact_enabled': spec['contact_enabled'], 'gravity': list(spec['gravity']),
        'enabled_joints': list(case['enabled_joints']),
        'actuator_gainprm': m.actuator_gainprm.tolist(),
        'actuator_biasprm': m.actuator_biasprm.tolist(),
        'actuator_forcerange': m.actuator_forcerange.tolist(),
    }
    return m, d, metadata
