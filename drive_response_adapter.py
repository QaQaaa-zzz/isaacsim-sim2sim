"""Bounded real-torque candidate for matching a coarse scalar RK4 servo.

This module never advances an engine or writes generalized state.  It predicts
one scalar actuator response per active joint, then returns two constant torque
phases to apply in PhysX.  Exact endpoint matching only holds for the declared
constant-inertia, semi-implicit scalar surrogate before force clipping.  Coupled
rigid-body dynamics and contact require independent measured validation.
"""
import copy
import math

import numpy as np


def _finite(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def _phase_steps(macro_dt, physics_dt):
    macro_dt = _positive(macro_dt, 'macro_dt')
    physics_dt = _positive(physics_dt, 'physics_dt')
    steps = round(macro_dt / physics_dt)
    if steps < 2 or steps % 2 or not math.isclose(steps * physics_dt, macro_dt, rel_tol=0, abs_tol=1e-12):
        raise ValueError('macro_dt requires an even integer number of physics microsteps')
    return steps // 2


def rk4_servo_target(q, v, target, kp, kd, inertia, force_limit, macro_dt, *, velocity_mode=False):
    """Classical RK4 of q'=v and the force-capped scalar servo acceleration."""
    q, v, target = [_finite(value, name) for value, name in [(q, 'q'), (v, 'v'), (target, 'target')]]
    kp, kd = _finite(kp, 'kp'), _finite(kd, 'kd')
    if kp < 0 or kd < 0 or (velocity_mode and kp != 0):
        raise ValueError('gains must be nonnegative, with kp=0 for a velocity servo')
    inertia, force_limit, dt = _positive(inertia, 'inertia'), _positive(force_limit, 'force_limit'), _positive(macro_dt, 'macro_dt')
    stages = []

    def evaluate(state):
        sq, sv = state
        raw = kd * (target - sv) if velocity_mode else kp * (target - sq) - kd * sv
        torque = float(np.clip(raw, -force_limit, force_limit))
        stages.append({'q': float(sq), 'v': float(sv), 'uncapped_torque': float(raw), 'torque': torque})
        return np.asarray([sv, torque / inertia])

    state = np.asarray([q, v])
    k1 = evaluate(state)
    k2 = evaluate(state + .5 * dt * k1)
    k3 = evaluate(state + .5 * dt * k2)
    k4 = evaluate(state + dt * k3)
    final = state + (dt / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return {'q_end': float(final[0]), 'v_end': float(final[1]),
            'stage_torques': [s['torque'] for s in stages], 'stages': stages,
            'inertia': inertia, 'macro_dt': dt}


def two_phase_torques(q0, v0, q1, v1, inertia, macro_dt, physics_dt, force_limit):
    """Synthesize two equal-length phases, then cap to original motor authority.

    With n semi-implicit microsteps of size h in each phase:
      v1-v0 = n*h*(a1+a2)
      q1-q0-2*n*h*v0 = h²*n*(3*n+1)/2*a1 + h²*n*(n+1)/2*a2.
    These equations determine the uncapped torques uniquely.  The actual
    candidate always applies the capped values, including when matching fails.
    """
    q0, v0, q1, v1 = [_finite(value, name) for value, name in [(q0, 'q0'), (v0, 'v0'), (q1, 'q1'), (v1, 'v1')]]
    inertia, force_limit = _positive(inertia, 'inertia'), _positive(force_limit, 'force_limit')
    n = _phase_steps(macro_dt, physics_dt)
    h, duration = float(physics_dt), float(macro_dt)
    a = n * h
    b1 = h*h*n*(3*n + 1)/2.
    b2 = h*h*n*(n + 1)/2.
    velocity_sum = (v1 - v0) / a
    accel1 = (q1 - q0 - duration*v0 - b2*velocity_sum)/(b1-b2)
    accel2 = velocity_sum - accel1
    requested = inertia * np.asarray([accel1, accel2])
    capped = np.clip(requested, -force_limit, force_limit)
    capped_accel = capped / inertia
    predicted_q = q0 + duration*v0 + b1*capped_accel[0] + b2*capped_accel[1]
    predicted_v = v0 + a*np.sum(capped_accel)
    return {'requested_torques': requested.tolist(), 'capped_torques': capped.tolist(),
            'clipped_phases': (np.abs(requested) > force_limit).tolist(),
            'any_clipped': bool(np.any(np.abs(requested) > force_limit)),
            'force_limit': force_limit, 'microsteps_per_phase': n,
            'desired_q_end': q1, 'desired_v_end': v1,
            'predicted_q_end_after_clip': float(predicted_q),
            'predicted_v_end_after_clip': float(predicted_v),
            'terminal_error_after_clip': [float(predicted_q-q1), float(predicted_v-v1)]}


class PhaseDriveAdapter:
    """Stateful scheduler returning efforts; caller alone advances PhysX.

    Native drives for active joints must be disabled by the caller.  Inactive
    joints get zero contribution from this adapter so the caller can preserve
    their existing actuator implementation.  Reset between every case/episode.
    """

    def __init__(self, joint_names, actuator_joint_names, effective_inertias,
                 gains, force_limits, *, physics_dt=.0005, macro_dt=.005,
                 active_joints=('rearwheel_joint', 'steering_joint')):
        self.joint_names = list(joint_names)
        self.actuator_joint_names = list(actuator_joint_names)
        self.active_joints = list(active_joints)
        if len(set(self.joint_names)) != len(self.joint_names) or len(set(self.actuator_joint_names)) != len(self.actuator_joint_names):
            raise ValueError('joint names must be unique')
        if len(set(self.active_joints)) != len(self.active_joints) or any(j not in self.joint_names or j not in self.actuator_joint_names for j in self.active_joints):
            raise ValueError('active joints must be unique declared actuated joints')
        self.n = _phase_steps(macro_dt, physics_dt)
        self.physics_dt, self.macro_dt = float(physics_dt), float(macro_dt)
        self.inertias, self.gains, self.limits = {}, {}, {}
        for joint in self.active_joints:
            self.inertias[joint] = _positive(effective_inertias[joint], f'inertia.{joint}')
            self.limits[joint] = _positive(force_limits[joint], f'force_limit.{joint}')
            self.gains[joint] = {key: _finite(gains[joint][key], f'{key}.{joint}') for key in ('kp', 'kd')}
            if min(self.gains[joint].values()) < 0 or (joint == 'rearwheel_joint' and self.gains[joint]['kp'] != 0):
                raise ValueError('gains must be nonnegative, rear kp must be zero')
        self.reset()

    def reset(self):
        self._next_step = 0
        self._plans = {}
        self._ctrl = None

    def compute_efforts(self, step_index, q, qd, ctrl):
        if type(step_index) is not int or step_index != self._next_step:
            raise ValueError('phase adapter requires consecutive steps starting at zero')
        q, qd, ctrl = map(lambda x: np.asarray(x, dtype=float), (q, qd, ctrl))
        if q.shape != (len(self.joint_names),) or qd.shape != q.shape or ctrl.shape != (len(self.actuator_joint_names),):
            raise ValueError('q/qd/ctrl must match declared joint and actuator order')
        if not all(np.isfinite(value).all() for value in (q, qd, ctrl)):
            raise ValueError('q/qd/ctrl must remain finite')
        microstep = step_index % (2*self.n)
        if microstep == 0:
            self._ctrl = ctrl.copy()
            self._plans = {}
            for joint in self.active_joints:
                ji, ai = self.joint_names.index(joint), self.actuator_joint_names.index(joint)
                gain = self.gains[joint]
                prediction = rk4_servo_target(q[ji], qd[ji], ctrl[ai], gain['kp'], gain['kd'], self.inertias[joint], self.limits[joint], self.macro_dt, velocity_mode=joint == 'rearwheel_joint')
                plan = two_phase_torques(q[ji], qd[ji], prediction['q_end'], prediction['v_end'], self.inertias[joint], self.macro_dt, self.physics_dt, self.limits[joint])
                self._plans[joint] = {**plan, 'initial_q': float(q[ji]), 'initial_v': float(qd[ji]),
                                      'ctrl': float(ctrl[ai]), 'rk4_prediction': prediction}
        elif not np.array_equal(ctrl, self._ctrl):
            raise ValueError('ctrl must remain constant throughout each macro period')
        phase = int(microstep >= self.n)
        efforts = np.zeros(len(self.joint_names))
        for joint, plan in self._plans.items():
            efforts[self.joint_names.index(joint)] = plan['capped_torques'][phase]
        self._next_step += 1
        return efforts, {'macro_index': step_index // (2*self.n), 'microstep_index': microstep,
                         'phase_index': phase, 'plans': copy.deepcopy(self._plans)}


def identify_effective_inertias(model, data, actuator_joint_names):
    """Static free-articulation Schur complement; performs zero physics steps.

    data.qM must already be populated by a forward calculation at the declared
    initial qpos.  Contacts and actuator forces are not included in this mass
    matrix.  This is an approximation when subsequently held fixed.
    """
    import mujoco

    matrix = np.empty((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, matrix, data.qM)
    except TypeError:
        mujoco.mj_fullM(model, data, matrix)
    inverse = np.linalg.inv(matrix)
    dofs = [int(model.joint(name).dofadr[0]) for name in actuator_joint_names]
    effective = {name: float(1./inverse[dof, dof]) for name, dof in zip(actuator_joint_names, dofs)}
    if any(not math.isfinite(v) or v <= 0 for v in effective.values()):
        raise ValueError('effective inertia must be finite and positive')
    sub = inverse[np.ix_(dofs, dofs)]
    normalized = sub / np.sqrt(np.outer(np.diag(sub), np.diag(sub)))
    return {'method': 'I_eff[j] = 1 / inverse(M(q0))[dof_j,dof_j]; all other generalized accelerations are unconstrained',
            'effective_inertias': effective,
            'actuator_joint_names': list(actuator_joint_names), 'actuator_dof_indices': dofs,
            'mass_matrix': matrix.tolist(), 'inverse_mass_matrix': inverse.tolist(),
            'normalized_inverse_actuator_coupling': normalized.tolist(),
            'root_body_angular_acceleration_per_joint_torque': inverse[np.ix_([3, 4, 5], dofs)].tolist(),
            'physical_steps': 0}
