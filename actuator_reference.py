"""Continuous-time reference, independent of either simulator's integrator."""
import numpy as np


def velocity_step(t, *, inertia, damping, limit, target):
    """Rest-start fixed-axis rotor: I v'=clip(k*(target-v), +/-limit)."""
    t = np.asarray(t, dtype=float)
    if min(inertia, damping, limit) <= 0 or np.any(t < 0):
        raise ValueError('Positive physical parameters and nonnegative time required')
    speed = abs(target)
    release_v = max(0., speed-limit/damping)
    release_t = release_v*inertia/limit
    ramp_t = np.minimum(t, release_t)
    tail_t = np.maximum(0., t-release_t)
    rate = damping/inertia
    v = limit/inertia*ramp_t + (speed-release_v)*(-np.expm1(-rate*tail_t))
    q = .5*limit/inertia*ramp_t**2 + speed*tail_t + (speed-release_v)/rate*np.expm1(-rate*tail_t)
    return np.sign(target)*q, np.sign(target)*v
