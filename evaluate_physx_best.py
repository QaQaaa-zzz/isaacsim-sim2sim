"""Re-evaluate the declared best using the unchanged training environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.update(OMNI_KIT_ACCEPT_EULA='YES', JAX_PLATFORMS='cpu', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = json.loads((args.run / 'declaration.json').read_text())
    best = json.loads((args.run / 'best_model.json').read_text())
    actor_path = Path(best['checkpoint']) / 'actor.npz'
    identity = json.loads(actor_path.with_name('identity.json').read_text())
    assert hashlib.sha256(actor_path.read_bytes()).hexdigest() == identity['actor_sha256']
    hashes=json.loads((args.run / 'implementation_hashes.json').read_text())
    # Presentation code may be repaired without changing the frozen physics contract.
    presentation={'render_traces.py','scene_visuals.py','report_physx_training.py','run_physx_pipeline.py'}
    for name, expected in hashes.items():
        if name not in presentation:
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2))
    save('declaration.json', dict(best=best, actor_sha256=identity['actor_sha256'],
         training_configuration=str((args.run / 'declaration.json').resolve()),
         role='repeat of declared test panel; no training or parameter changes'))
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'limit_cpu_threads': 4, 'extra_args': [
        '--/app/settings/loadUserConfig=false', '--/app/settings/persistent=false']})
    try:
        import numpy as np
        from physx_task import PhysxTask
        from policy_runtime import Actor
        env = PhysxTask(spec, spec['ppo']['num_envs'], spec['ppo']['seed'])
        actor = Actor(actor_path)
        seeds = spec['best_selection']['test_seeds']
        env.hard_reset()
        env.reset(np.arange(len(seeds)), seeds=seeds)
        save('runtime_audit.json', env.audit)
        active = np.ones(len(seeds), bool)
        traces, steps = [[] for _ in seeds], [[] for _ in seeds]
        for k in range(spec['episode_horizon']):
            _, _, done, info = env.step(actor(env.obs))
            for i in range(len(seeds)):
                if active[i]:
                    row = env.trace[i].copy()
                    row[0] = (k + 1) * spec['control_dt']
                    traces[i].append(row)
                    steps[i].append(info[i])
                    if done[i]:
                        active[i] = False
            if not active.any():
                break
            env.reset(np.flatnonzero(done))
        summary, repeatability = [], []
        for i, seed in enumerate(seeds):
            name = f'seed_{seed}'
            folder = args.output / name
            folder.mkdir()
            z = np.asarray(traces[i])
            np.savez_compressed(folder / 'traces.npz', target=z, joint_names=env.names)
            for filename, value in [('steps.json', steps[i]), ('endpoint.json', steps[i][-1]),
                ('case.json', dict(seed=seed, recorded_engine='PhysX', friction=.5,
                  duration=float(z[-1, 0]), checkpoint=str(actor_path.parent),
                  contact_label='best @ 5.251M | wheel mu=0.5 | calibrated PD'))]:
                (folder / filename).write_text(json.dumps(value, indent=2))
            summary.append(dict(seed=seed, **steps[i][-1]))
            old = np.load(args.run / 'evaluation' / 'best_test' / name / 'traces.npz')['target']
            repeatability.append(dict(seed=seed, old_steps=len(old), new_steps=len(z),
                identical=bool(np.array_equal(z, old)),
                max_abs_difference=float(np.max(np.abs(z-old))) if z.shape == old.shape else None))
        save('summary.json', summary)
        save('repeatability.json', repeatability)
        save('status.json', dict(completed=True, recorded_transitions=sum(map(len, traces)),
             physics_transitions=(k+1)*env.n, all_finite=all(np.isfinite(x).all() for x in map(np.asarray, traces))))
        print('EVALUATION_COMPLETE', json.dumps(repeatability), flush=True)
    finally:
        app.close()

if __name__ == '__main__':
    main()
