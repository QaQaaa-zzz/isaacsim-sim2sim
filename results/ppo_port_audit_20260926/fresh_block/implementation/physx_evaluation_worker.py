"""Persistent, independent eight-environment PhysX evaluation process.

Run with ``--config CONFIG --queue NEW_QUEUE_DIRECTORY``. After ready.json,
the serial client atomically replaces request.json with an object containing
``id``, ``actor_path``, ``seeds`` (eight distinct integers), and ``output``
(a new panel directory); ``checkpoint_label`` is optional. Paths are resolved
against the worker's working directory, so clients should use absolute paths.
The worker claims requests by renaming them to processing.json. A matching
response.json is published only after all panel artifacts have been closed.
Clients must consume each response before sending another request. Creating
queue/stop requests shutdown, including cancellation at the next control step.

Neither importing this module nor --help starts Isaac Sim. The environment
always has eight lanes, independently of the training vector width in CONFIG.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import faulthandler
import signal
import logging

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
sys.dont_write_bytecode = True

NUM_ENVS = 8
MAX_HORIZON = 400
POLL_SECONDS = 0.2


def write_json(path, value):
    """Publish complete JSON atomically, rejecting nonfinite numerical values."""
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError('Request must be a JSON object')
    request_id = request.get('id')
    if (not isinstance(request_id, (str, int)) or isinstance(request_id, bool)
            or (isinstance(request_id, str) and not request_id)):
        raise ValueError('Request id must be a nonempty string or integer')
    seeds = request.get('seeds')
    if (not isinstance(seeds, list) or len(seeds) != NUM_ENVS
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != NUM_ENVS):
        raise ValueError('Exactly eight distinct nonnegative integer seeds are required')
    for field in ('actor_path', 'output'):
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f'{field} must be a nonempty path string')
    actor_path = Path(request['actor_path']).expanduser().resolve()
    if not actor_path.is_file():
        raise FileNotFoundError(f'Actor export does not exist: {actor_path}')
    output = Path(request['output']).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Panel output must be new: {output}')
    label = request.get('checkpoint_label', actor_path.parent.name)
    if not isinstance(label, str) or not label:
        raise ValueError('checkpoint_label must be a nonempty string')
    return dict(request, actor_path=str(actor_path), output=str(output),
                checkpoint_label=label)


def validate_spec(spec):
    if spec.get('source_ppo_port') is not True:
        raise ValueError('Evaluation requires source_ppo_port=true')
    horizon = spec.get('episode_horizon')
    if type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError('episode_horizon must be an integer in [1, 400]')
    for field in ('physics_dt', 'control_dt'):
        value = spec.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{field} must be finite and positive')
    contact = spec.get('contact_required', {})
    if (contact.get('static_friction') != 0.5
            or contact.get('dynamic_friction') != 0.5
            or contact.get('combine') != 'min'):
        raise ValueError('This worker records the declared wheel 0.5/min contact contract')


def require_finite(value, label):
    import numpy as np

    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f'{label}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite(item, f'{label}[{index}]')
    elif isinstance(value, (float, int, np.ndarray, np.number)):
        if not np.isfinite(value).all():
            raise RuntimeError(f'Nonfinite value in {label}')


def load_actor(path):
    import numpy as np
    from policy_runtime import Actor

    before = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    actor = Actor(path)
    after = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError('Actor export changed while it was loaded')
    require_finite(actor.p, 'actor_export')
    if (np.shape(actor.p['mean']) != (76,)
            or np.shape(actor.p['std']) != (76,)
            or not (actor.p['std'] > 0).all()):
        raise ValueError('Actor export requires finite 76D mean and positive 76D std')
    # Keep the exported running statistics fixed throughout this panel.
    for value in actor.p.values():
        value.setflags(write=False)
    return actor, before


def evaluate_panel(env, spec, request, actor, actor_sha256, counters=None,
                   stop_path=None):
    """Record only the first episode of each lane, including its terminal step."""
    import numpy as np

    if env.n != NUM_ENVS:
        raise ValueError('Scored evaluation must use exactly eight PhysX environments')
    validate_spec(spec)
    request = validate_request(request)
    if counters is None:
        counters = {'physics_transitions': 0, 'recorded_transitions': 0}
    started = time.monotonic()
    folder = Path(request['output'])
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'request.json', request)
    env.hard_reset()
    env.reset(np.arange(NUM_ENVS), seeds=request['seeds'])
    require_finite(env.obs, 'reset_observation')
    active = np.ones(NUM_ENVS, dtype=bool)
    traces = [[] for _ in range(NUM_ENVS)]
    rows = [[] for _ in range(NUM_ENVS)]
    endpoints = [None] * NUM_ENVS
    for step in range(spec['episode_horizon']):
        if stop_path is not None and Path(stop_path).exists():
            raise InterruptedError('Evaluation cancelled by queue/stop')
        actions = np.asarray(actor(env.obs), dtype=np.float32)
        require_finite(actions, 'deterministic_actions')
        if actions.shape != (NUM_ENVS, 4) or (np.abs(actions) > 1).any():
            raise ValueError('Actor must produce eight bounded four-dimensional actions')
        observation, reward, done, info = env.step(actions)
        counters['physics_transitions'] += NUM_ENVS
        require_finite(observation, 'observation')
        require_finite(reward, 'reward')
        require_finite(env.trace, 'target_trace')
        require_finite(info, 'step_info')
        if (np.shape(observation) != (NUM_ENVS, 76)
                or np.shape(reward) != (NUM_ENVS,)
                or np.shape(done) != (NUM_ENVS,)
                or np.shape(env.trace) != (NUM_ENVS, 31)
                or len(info) != NUM_ENVS):
            raise ValueError('Unexpected PhysxTask observation, reward, terminal, or trace shape')
        for index in np.flatnonzero(active):
            row = env.trace[index].copy()
            row[0] = (step + 1) * spec['control_dt']
            traces[index].append(row)
            rows[index].append(info[index])
            counters['recorded_transitions'] += 1
            if done[index]:
                endpoints[index] = info[index]
                active[index] = False
        if not active.any():
            break
        # Finished lanes still participate in physics; their subsequent episodes
        # are explicitly excluded from the score and from recorded traces.
        env.reset(np.flatnonzero(done))
        require_finite(env.obs, 'post_reset_observation')

    summary = []
    for index, seed in enumerate(request['seeds']):
        endpoint = endpoints[index] or rows[index][-1]
        endpoints[index] = endpoint
        path = folder / f'seed_{seed}'
        path.mkdir()
        np.savez_compressed(path / 'traces.npz',
                            target=np.asarray(traces[index]), joint_names=env.names)
        write_json(path / 'steps.json', rows[index])
        write_json(path / 'endpoint.json', endpoint)
        write_json(path / 'case.json', {
            'seed': seed, 'request_id': request['id'], 'recorded_engine': 'PhysX',
            'contact_label': 'Wheel friction 0.5/min; calibrated PD; actual endpoint',
            'friction': 0.5, 'source_anisotropic_contact_preserved': False,
            'checkpoint': str(Path(request['actor_path']).parent),
            'actor_path': request['actor_path'],
            'checkpoint_label': request['checkpoint_label'],
            'actor_sha256': actor_sha256, 'deterministic_policy': True,
            'normalizer': 'frozen exported mean and std',
            'reset_sampling': 'original NumPy seeded reset; source training stream not used',
            'airborne_reset': endpoint['airborne_reset'],
            'duration': len(traces[index]) * spec['control_dt'],
            'physics_dt': spec['physics_dt'], 'control_dt': spec['control_dt'],
            'num_evaluation_envs': NUM_ENVS, 'full_solver_reset_before_panel': True,
            'recorded_steps': len(traces[index]),
            'endpoint_reason': ('environment_done' if not active[index]
                                else 'evaluation_horizon'),
        })
        summary.append(dict(seed=seed, **endpoint,
                            horizon_reached=len(traces[index]) == spec['episode_horizon']))
    mean_return = float(np.mean([endpoint['return'] for endpoint in endpoints]))
    require_finite(mean_return, 'mean_return')
    write_json(folder / 'summary.json', summary)
    response = dict(id=request['id'], complete=True, mean_return=mean_return,
                    summary=summary, endpoints=endpoints, error=None,
                    output=str(folder), actor_path=request['actor_path'],
                    actor_sha256=actor_sha256, **counters,
                    wall_s=time.monotonic() - started)
    write_json(folder / 'result.json', response)
    return response


def serve(env, spec, queue):
    request_path = queue / 'request.json'
    processing_path = queue / 'processing.json'
    completed = 0
    while not (queue / 'stop').exists():
        if not request_path.exists():
            time.sleep(POLL_SECONDS)
            continue
        request_path.rename(processing_path)
        request = None
        counters = {'physics_transitions': 0, 'recorded_transitions': 0}
        started = time.monotonic()
        try:
            request = json.loads(processing_path.read_text())
            request = validate_request(request)
            write_json(queue / 'state.json', dict(stage='evaluating', pid=os.getpid(),
                       request_id=request['id'], completed_requests=completed))
            actor, actor_sha256 = load_actor(request['actor_path'])
            response = evaluate_panel(env, spec, request, actor, actor_sha256,
                                      counters, queue / 'stop')
            completed += 1
            write_json(queue / 'state.json', dict(stage='ready', pid=os.getpid(),
                       last_request_id=request['id'], completed_requests=completed))
            # Remove the claim before publishing the response: a client may
            # immediately enqueue the next request once it observes completion.
            processing_path.unlink()
            write_json(queue / 'response.json', response)
        except BaseException as error:
            request_id = request.get('id') if isinstance(request, dict) else None
            response = dict(id=request_id, complete=False, mean_return=None,
                            summary=[], endpoints=[], **counters,
                            wall_s=time.monotonic() - started,
                            error=f'{type(error).__name__}: {error}',
                            traceback=traceback.format_exc(), pid=os.getpid())
            write_json(queue / 'error.json', response)
            write_json(queue / 'state.json', dict(stage='failed', pid=os.getpid(),
                       request_id=request_id, completed_requests=completed,
                       error=response['error']))
            write_json(queue / 'response.json', response)
            # A failed panel may have left the solver or disk incomplete. Keep
            # its claim and partial artifacts for inspection; never auto-retry.
            raise
    write_json(queue / 'state.json', dict(stage='stopped', pid=os.getpid(),
               completed_requests=completed))


def main(args):
    queue = args.queue.expanduser().resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for filename in ('ready.json', 'request.json', 'processing.json',
                         'response.json', 'error.json', 'state.json', 'stop'):
            if (queue / filename).exists():
                raise FileExistsError(f'Use a fresh queue; found {queue / filename}')
        app = None
        try:
            spec = json.loads(args.config.read_text())
            validate_spec(spec)
            write_json(queue / 'state.json', dict(stage='initializing', pid=os.getpid()))
            from isaacsim import SimulationApp

            app = SimulationApp({
                'headless': True, 'disable_viewport_updates': True, 'limit_cpu_threads': 4,
                'extra_args': ['--/app/settings/loadUserConfig=false',
                               '--/app/settings/persistent=false'],
            })
            faulthandler.enable()
            faulthandler.register(signal.SIGUSR1,all_threads=False)
            logging.getLogger('jax').setLevel(logging.WARNING)
            from physx_task import PhysxTask

            env = PhysxTask(spec, NUM_ENVS, spec['ppo']['seed'])
            ready = dict(pid=os.getpid(), num_envs=NUM_ENVS,
                         config=str(args.config.resolve()),
                         runtime_audit=env.audit)
            write_json(queue / 'runtime_audit.json', env.audit)
            write_json(queue / 'state.json', dict(stage='ready', pid=os.getpid(),
                       completed_requests=0))
            write_json(queue / 'ready.json', ready)
            serve(env, spec, queue)
        except BaseException as error:
            # Request failures have already retained their id and exact error.
            if not (queue / 'error.json').exists():
                failure = dict(id=None, complete=False, mean_return=None,
                               summary=[], endpoints=[], physics_transitions=0,
                               recorded_transitions=0, wall_s=0,
                               pid=os.getpid(), error=f'{type(error).__name__}: {error}',
                               traceback=traceback.format_exc())
                write_json(queue / 'error.json', failure)
                write_json(queue / 'state.json', dict(stage='failed', pid=os.getpid(),
                           error=failure['error']))
                write_json(queue / 'response.json', failure)
            raise
        finally:
            if app is not None:
                app.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    main(parser.parse_args())
