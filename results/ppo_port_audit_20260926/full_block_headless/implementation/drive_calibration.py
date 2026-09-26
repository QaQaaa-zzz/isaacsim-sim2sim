"""Explicit target-only drive calibration, keyed by joint rather than DOF order."""
import numpy as np


def configured_gains(names, kp, kd, values):
    """Override explicit SI gains without changing source-model actuator mapping."""
    kp,kd=np.array(kp,dtype=float,copy=True),np.array(kd,dtype=float,copy=True)
    for joint,value in values.items():
        if joint not in names:
            raise ValueError(f'Unknown calibration joint: {joint}')
        gains=np.array([value['kp'],value['kd']],dtype=float)
        if not np.isfinite(gains).all() or np.any(gains<0):
            raise ValueError(f'Invalid drive gains for {joint}: {value}')
        idx=names.index(joint);kp[...,idx],kd[...,idx]=gains
    return kp,kd


def scaled_gains(names, kp, kd, scales):
    kp, kd = np.array(kp, copy=True), np.array(kd, copy=True)
    for joint, value in scales.items():
        if joint not in names:
            raise ValueError(f'Unknown calibration joint: {joint}')
        factors = np.array([value['kp'], value['kd']], dtype=float)
        if not np.isfinite(factors).all() or np.any(factors < 0):
            raise ValueError(f'Invalid drive scales for {joint}: {value}')
        idx = names.index(joint)
        kp[..., idx] *= factors[0]; kd[..., idx] *= factors[1]
    return kp, kd
