"""Compare recorded real-engine responses on common, non-interpolated times."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

JOINTS = ['rearwheel_joint', 'steering_joint', 'frontwheel_joint', 'hip_joint', 'knee_joint']
Q_SCALE = np.array([.72, .1, .1, .1, .1])
QD_SCALE = np.array([12., 2., 2., 2., 2.])


def validate_trace(trace, protocol, case):
    from response_protocol import expand_controls
    required = ('time_s', 'qpos', 'qvel', 'joint_names', 'joint_q', 'joint_qd', 'ctrl', 'omega_body_end')
    for key in required:
        if key not in trace:
            raise ValueError(f'missing required trace field {key}')
    t = trace['time_s']
    if len(t) < 2:
        raise ValueError('trace needs initial and integrated state')
    # Recorded Warp times use float32. Recover the declared substep at ns
    # precision, then verify every timestamp instead of accumulating roundoff.
    dt = round(float(t[1] - t[0]), 9)
    controls = expand_controls(case['segments'], dt)
    expected_t = np.arange(len(controls) + 1) * dt
    expected_ctrl = np.vstack([controls[0], controls])
    if len(t) != len(expected_t) or not np.allclose(t, expected_t, atol=1e-7, rtol=0):
        raise ValueError('trace does not contain complete uniform physical times')
    if 'engine_time' in trace and not np.allclose(trace['engine_time'], expected_t, atol=2e-7, rtol=0):
        raise ValueError('actual engine times differ from the protocol')
    if trace['ctrl'].shape != expected_ctrl.shape or not np.allclose(trace['ctrl'], expected_ctrl, atol=2e-6, rtol=0):
        raise ValueError('input differs from the protocol at a physical substep')
    initial = case.get('initial', protocol.get('initial'))
    for key, width in [('qpos', 12), ('qvel', 11), ('joint_q', 5), ('joint_qd', 5), ('omega_body_end', 3)]:
        if trace[key].shape != (len(t), width) or not np.isfinite(trace[key]).all():
            raise ValueError(f'invalid response field {key}')
        if key in ('qpos', 'qvel') and not np.allclose(trace[key][0], initial[key], atol=2e-6, rtol=0):
            raise ValueError(f'initial {key} differs from protocol')


def paired_samples(source_time, target_time, sample_dt=.01):
    """Use identical actual timestamps; never interpolate measured states."""
    a, b = np.asarray(source_time), np.asarray(target_time)
    if (len(a) < 2 or len(b) < 2 or not np.isfinite(a).all()
            or not np.isfinite(b).all() or np.any(np.diff(a) <= 0)
            or np.any(np.diff(b) <= 0) or abs(a[0]) > 1e-8 or abs(b[0]) > 1e-8):
        raise ValueError('invalid time arrays')
    if not np.isclose(a[-1], b[-1], atol=1e-7, rtol=0):
        raise ValueError('response durations differ')
    required = np.arange(1, round(a[-1] / sample_dt) + 1) * sample_dt
    indices = []
    for time in (a, b):
        idx = np.array([int(np.argmin(np.abs(time - t))) for t in required])
        if not np.allclose(time[idx], required, atol=1e-7, rtol=0):
            raise ValueError('missing common recorded timestamp; interpolation forbidden')
        indices.append(idx)
    return tuple(indices)


def response_metrics(source, target):
    if (source['joint_names'].tolist() != JOINTS
            or target['joint_names'].tolist() != JOINTS):
        raise ValueError('joint order must match the declared source order')
    for key in ('qpos', 'qvel'):
        if key not in source or key not in target:
            raise ValueError(f'missing required {key}')
        if not np.allclose(source[key][0], target[key][0], atol=2e-6, rtol=0):
            raise ValueError(f'initial {key} differs')
    i, j = paired_samples(source['time_s'], target['time_s'])
    if not np.allclose(source['ctrl'][i], target['ctrl'][j], atol=2e-6, rtol=0):
        raise ValueError('applied input differs at common samples')
    errors = [target[key][j] - source[key][i] for key in ('joint_q', 'joint_qd', 'omega_body_end')]
    if not all(np.isfinite(e).all() for e in errors):
        raise ValueError('nonfinite response cannot be scored')
    eq, ev, ew = errors
    rmse = lambda value: float(np.sqrt(np.mean(value ** 2)))
    all_i, all_j = np.r_[0, i], np.r_[0, j]
    interval_source = np.diff(source['joint_q'][all_i], axis=0) / np.diff(source['time_s'][all_i])[:, None]
    interval_target = np.diff(target['joint_q'][all_j], axis=0) / np.diff(target['time_s'][all_j])[:, None]
    return dict(loss=float(np.mean((eq / Q_SCALE)**2 + (ev / QD_SCALE)**2) + np.mean(ew**2)),
                q_rmse_rad=rmse(eq), qd_rmse_rad_s=rmse(ev), omega_rmse_rad_s=rmse(ew),
                per_joint_q_rmse_rad=np.sqrt(np.mean(eq**2, axis=0)).tolist(),
                per_joint_qd_rmse_rad_s=np.sqrt(np.mean(ev**2, axis=0)).tolist(),
                per_axis_omega_rmse_rad_s=np.sqrt(np.mean(ew**2, axis=0)).tolist(),
                interval_velocity_rmse_rad_s=rmse(interval_target - interval_source),
                source_qd_vs_interval_rmse_rad_s=rmse(source['joint_qd'][i] - interval_source),
                target_qd_vs_interval_rmse_rad_s=rmse(target['joint_qd'][j] - interval_target),
                sample_count=len(i), sample_dt_s=.01)


def admissible(candidate, baseline):
    return all(candidate[key] <= 1.1 * baseline[key] + floor for key, floor in
               [('q_rmse_rad', 1e-4), ('qd_rmse_rad_s', 1e-3), ('omega_rmse_rad_s', 1e-3)])


def read_trace(folder):
    folder = Path(folder)
    paths = [folder / n for n in ('trace.npz', 'traces.npz') if (folder / n).is_file()]
    if len(paths) != 1:
        raise ValueError(f'expected one trace in {folder}; got {paths}')
    with np.load(paths[0], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def make_plots(protocol, source, targets, metrics, output, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    colors = ['#CC6677', '#4477AA', '#228833']
    names = ['steering_joint', 'rearwheel_joint', 'hip_joint', 'knee_joint']
    for case in protocol['cases']:
        name = case['name']; src = source[name]
        active = case.get('enabled_joints', names)[0]
        k = JOINTS.index(active); ci = names.index(active)
        fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
        shown = [('MJX-Warp RK4 5 ms', src, '#111111')]
        shown += [(candidate, targets[candidate][name], color) for candidate, color in zip(selected, colors)]
        for label, trace, color in shown:
            t = trace['time_s']; q = trace['joint_q'][:, k]
            axes[0, 0].step(t, trace['ctrl'][:, ci], where='pre', label=label, color=color)
            axes[0, 1].plot(t, q, label=label, color=color)
            axes[1, 0].plot(t, trace['joint_qd'][:, k], label=label, color=color)
            axes[1, 1].plot(t[1:], np.diff(q) / np.diff(t), label=label, color=color)
            axes[2, 0].plot(t, trace['omega_body_end'][:, 1], label=label, color=color)
            axes[2, 1].plot(t, np.linalg.norm(trace['omega_body_end'], axis=1), label=label, color=color)
        titles = [('Applied target', 'rad/s' if active == 'rearwheel_joint' else 'rad'),
                  ('Joint angle', 'rad'), ('End-state joint velocity', 'rad/s'),
                  ('Measured delta angle / physics dt', 'rad/s'),
                  ('End-state body omega Y', 'rad/s'), ('End-state body omega norm', 'rad/s')]
        for axis, (title, unit) in zip(axes.flat, titles):
            axis.set(title=title, xlabel='Time (s)', ylabel=unit); axis.grid(alpha=.2)
            boundary = 0.
            for segment in case['segments'][:-1]:
                boundary += segment['duration_s']; axis.axvline(boundary, color='.65', lw=.6, ls=':')
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f'{name} | {active} | identical direct inputs and initial state\n'
                     f'contact={protocol["contact_enabled"]}, gravity={protocol["gravity"]}; engineering response, not task success')
        fig.savefig(output / f'{name}.png', dpi=160); fig.savefig(output / f'{name}.pdf'); plt.close(fig)
        if len(case.get('enabled_joints', names)) > 1:
            fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True, constrained_layout=True)
            for row, joint in enumerate(names):
                column = JOINTS.index(joint)
                for label, trace, color in shown:
                    t = trace['time_s']
                    axes[row, 0].step(t, trace['ctrl'][:, row], where='pre', color=color, label=label)
                    axes[row, 1].plot(t, trace['joint_q'][:, column], color=color, label=label)
                    axes[row, 2].plot(t, trace['joint_qd'][:, column], color=color, label=label)
                for col, title in enumerate(['Target', 'Joint angle (rad)', 'Joint velocity (rad/s)']):
                    axes[row, col].set_title(f'{joint}: {title}'); axes[row, col].grid(alpha=.2)
                    axes[row, col].set_xlabel('Time (s)')
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(f'{name} | all four active joints | target units rad, rear rad/s')
            fig.savefig(output / f'{name}_all_joints.png', dpi=160)
            fig.savefig(output / f'{name}_all_joints.pdf'); plt.close(fig)
        if 'gyro_body_native' in src:
            fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
            for k, axis in enumerate(axes):
                axis.plot(src['time_s'], src['gyro_body_native'][:, k], label='native gyro after RK4', color='#CC6677')
                axis.plot(src['time_s'], src['omega_body_end'][:, k], label='final-state body omega', color='#111111')
                axis.set(ylabel=f'{"XYZ"[k]} (rad/s)'); axis.grid(alpha=.2)
            axes[0].legend(); axes[2].set_xlabel('Time (s)'); fig.suptitle(f'{name} | actual MJX-Warp sensor vs final state')
            fig.savefig(output / f'{name}_gyro.png', dpi=160); fig.savefig(output / f'{name}_gyro.pdf'); plt.close(fig)


def analyze(source_root, target_root, protocol_path, output, baseline='current_hybrid_1ms'):
    for root in (source_root, target_root):
        status_path = Path(root) / 'status.json'
        status = json.loads(status_path.read_text())
        if not (status.get('completed') is True or status.get('status') == 'complete'):
            raise ValueError(f'run is not complete: {root}')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(Path(protocol_path).read_text()); names = [c['name'] for c in protocol['cases']]
    source = {name: read_trace(Path(source_root) / name) for name in names}
    target_dirs = sorted(p for p in Path(target_root).iterdir() if p.is_dir() and all((p / n).is_dir() for n in names))
    targets = {p.name: {name: read_trace(p / name) for name in names} for p in target_dirs}
    if baseline not in targets:
        raise ValueError(f'baseline {baseline} missing complete cases')
    for case in protocol['cases']:
        validate_trace(source[case['name']], protocol, case)
        for candidate in targets.values():
            validate_trace(candidate[case['name']], protocol, case)
    metrics = {cand: {name: response_metrics(source[name], target[name]) for name in names} for cand, target in targets.items()}
    ranking = []
    for cand, by_case in metrics.items():
        ranking.append(dict(candidate=cand, mean_loss=float(np.mean([v['loss'] for v in by_case.values()])),
                            admissible_all_cases=all(admissible(by_case[n], metrics[baseline][n]) for n in names)))
    ranking.sort(key=lambda value: (value['mean_loss'], value['candidate'] != baseline))
    admissible_rank = [r for r in ranking if r['admissible_all_cases']]
    best = admissible_rank[0]['candidate']
    summary = dict(baseline=baseline, ranking=ranking, best_admissible=best, metrics=metrics,
                   selected_is_improvement=best != baseline and next(r['mean_loss'] for r in ranking if r['candidate'] == best)
                       < next(r['mean_loss'] for r in ranking if r['candidate'] == baseline) - 1e-12,
                   sample_rule='real common 10 ms samples; excludes initial state')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'metrics.csv').open('w') as file:
        writer = csv.writer(file); writer.writerow(['candidate', 'case', 'loss', 'q_rmse_rad', 'qd_rmse_rad_s', 'omega_rmse_rad_s', 'admissible'])
        for cand, by_case in metrics.items():
            for name, values in by_case.items():
                writer.writerow([cand, name] + [values[key] for key in ['loss', 'q_rmse_rad', 'qd_rmse_rad_s', 'omega_rmse_rad_s']] + [admissible(values, metrics[baseline][name])])
                src, tgt = source[name], targets[cand][name]; i, j = paired_samples(src['time_s'], tgt['time_s'])
                cols = np.column_stack([src['time_s'][i], src['ctrl'][i], src['joint_q'][i], tgt['joint_q'][j], src['joint_qd'][i], tgt['joint_qd'][j], src['omega_body_end'][i], tgt['omega_body_end'][j]])
                header = ['time_s'] + [f'ctrl_{k}' for k in range(4)]
                header += [f'{engine}_{kind}_{joint}' for kind in ['q', 'qd'] for engine in ['mjx', 'physx'] for joint in JOINTS]
                header += [f'{engine}_omega_body_{axis}' for engine in ['mjx', 'physx'] for axis in 'xyz']
                np.savetxt(output / f'{cand}__{name}.csv', cols, delimiter=',', header=','.join(header), comments='')
    selected = list(dict.fromkeys([baseline, ranking[0]['candidate'], best]))
    make_plots(protocol, source, targets, metrics, output, selected)
    print(json.dumps(dict(ranking=ranking, best_admissible=best), indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True); p.add_argument('--target', required=True)
    p.add_argument('--protocol', required=True); p.add_argument('--output', required=True)
    p.add_argument('--baseline', default='current_hybrid_1ms')
    a = p.parse_args(); analyze(a.source, a.target, a.protocol, a.output, a.baseline)
