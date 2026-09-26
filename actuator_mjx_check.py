"""Confirm selected probes in installed training MJX-Warp, without editing it."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/home/qy/ISAAC——SIM/cache/jax'
import jax
import numpy as np
import mujoco
from mujoco import mjx
from actuator_audit import prepare, ROOT

p = argparse.ArgumentParser()
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--cases', nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
summary = []
for name in a.cases:
    case = json.loads((a.audit/name/'case.json').read_text())
    pair = np.load(a.audit/name/'traces.npz')
    names = pair['joint_names'].tolist()
    m = mujoco.MjModel.from_xml_path(str(a.audit/'rotor.xml' if case['model'] == 'rotor' else ROOT/'model/source.xml'))
    d, qa, va, ai = prepare(m, case, names)
    mm = mjx.put_model(m, impl='warp')
    dd = mjx.put_data(m, d, impl='warp', naconmax=128, njmax=256)
    step = jax.jit(lambda state: mjx.step(mm, state))
    rows = [np.r_[0., d.qpos[qa], d.qvel[va]]]
    for k in range(round(case['duration']/case['dt'])):
        dd = step(dd); jax.block_until_ready(dd)
        rows.append(np.r_[(k+1)*case['dt'], np.asarray(dd.qpos)[qa], np.asarray(dd.qvel)[va]])
    rows = np.asarray(rows)
    out = a.output/name; out.mkdir()
    np.savez_compressed(out/'trace.npz', mjx=rows, joint_names=names)
    err = rows[:, 1+len(names):]-pair['RK4'][:, 1+len(names):]
    metrics = dict(case=case, engine='MJX-Warp in original training virtual environment',
                   max_abs_qd_difference_from_cpu=float(np.max(np.abs(err))),
                   qd_rmse_from_cpu=np.sqrt(np.mean(err**2, axis=0)).tolist(),
                   mjx_final_qd=rows[-1, 1+len(names):].tolist())
    (out/'result.json').write_text(json.dumps(metrics, indent=2))
    summary.append(metrics); (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('MJX_DONE', json.dumps(metrics), flush=True)
(a.output/'status.json').write_text(json.dumps(dict(completed=True, cases=len(summary), training_interactions=0)))
