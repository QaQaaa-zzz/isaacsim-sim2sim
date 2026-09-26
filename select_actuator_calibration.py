"""Freeze selection before running predeclared independent validation inputs."""
import argparse
import json
from pathlib import Path
import numpy as np


def loss(out):
    case = json.loads((out/'case.json').read_text())
    d = np.load(out/'traces.npz'); names = d['joint_names'].tolist()
    j = names.index(case['joint']); n = len(names)
    scales = [.6, 12.] if case['joint'] == 'rearwheel_joint' else [.1, 2.]
    indices = [1+j, 1+n+j]
    errors = (d['physx'][1:, indices]-d['RK4'][1:, indices])/scales
    raw = d['physx'][1:, indices]-d['RK4'][1:, indices]
    return dict(loss=float(np.mean(np.sum(errors**2, axis=1))),
                q_rmse_rad=float(np.sqrt(np.mean(raw[:, 0]**2))),
                qd_rmse_rad_s=float(np.sqrt(np.mean(raw[:, 1]**2))))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--validation-config', type=Path, required=True)
    a = p.parse_args()
    declaration = json.loads((a.input/'declaration.json').read_text())
    status = json.loads((a.input/'status.json').read_text())
    assert status['completed'] and status['cases'] == len(declaration['cases'])
    grouped = {}
    for case in declaration['cases']:
        row = dict(case=case, **loss(a.input/case['name']))
        grouped.setdefault(case['joint'], []).append(row)
    selection = {}
    for joint, rows in grouped.items():
        best = min(rows, key=lambda r: r['loss'])
        baseline = next(r for r in rows if r['case']['target_drive_scales'] == dict(kp=1., kd=1.))
        selection[joint] = dict(best=best, baseline=baseline, candidates=rows, adopted=False)
    cases = []
    for i, val in enumerate(declaration['validation']):
        for label in ['baseline', 'best']:
            selected = selection[val['joint']][label]['case']
            cases.append(dict(selected, name=f'validation_{i}_{label}', input=val['input'],
                              initial_joint_velocities={val['joint']: val['initial_velocity']},
                              validation_index=i, candidate_role=label))
    (a.input/'selection.json').write_text(json.dumps(selection, indent=2))
    a.validation_config.write_text(json.dumps(dict(declaration, cases=cases,
        purpose='Independent declared inputs, selected target parameters frozen; no refitting'), indent=2))
    print(json.dumps({j: {k: v[k] for k in ['best','baseline','adopted']} for j, v in selection.items()}, indent=2))


if __name__ == '__main__':
    main()
