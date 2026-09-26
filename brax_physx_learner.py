"""Original Brax PPO learner, with host-collected PhysX Transition[B, T] data.

Source contracts (read-only, /home/qy/DVGC/JIT/src/jit_dvgc):
  ppo.py:79-89: asymmetric 76/106-input, three 256-unit Swish networks.
  formal_training.py:731-768: source PPO arguments and constant learning rate.
Installed Brax 0.14.2 training/agents/ppo/train.py:493-627 supplies the
sequence shuffle, original loss, optimizer, and pre-SGD statistics timing.
No dynamics or simulator is imported here. Critic/GAE are recomputed by the
original loss at every minibatch, rather than cached from rollout sampling.

The source full-reset wrapper clears both `truncation` and `time_out` at done
(Playground wrapper.py:193-201; JIT ppo.py:108-166 only preserves episode
metrics). Hence this adapter requires those collected fields to be zero and
uses discount=1-done, including source timeouts. This preserves effective
source behavior; it does not silently repair the nominal timeout flag.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses, networks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(normalizer) -> int:
    count = normalizer.count
    if isinstance(count, types.UInt64):
        return int(count.hi) * 2**32 + int(count.lo)
    return int(count)


def _source_payload(checkpoint: Path):
    # Import only the small identity/serialization module, never source env/ppo.
    source = Path('/home/qy/DVGC/JIT/src')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from jit_dvgc.checkpoint import CheckpointIdentity, CheckpointPayload
    from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
    identity = json.loads((checkpoint / 'identity.json').read_text())
    expected = CheckpointIdentity(
        identity['config_sha256'], identity['xml_sha256'], ACTOR_FRAME_FIELDS,
        ACTOR_TASK_FIELDS, ACTION_ORDER)
    for field, value in expected.__dict__.items():
        declared = identity.get(field)
        if isinstance(value, tuple) and isinstance(declared, list):
            declared = tuple(declared)
        if declared != value:
            raise ValueError(f'Checkpoint {field} mismatch')
    if _sha256(checkpoint / 'payload.pkl') != identity['payload_sha256']:
        raise ValueError('Checkpoint payload hash mismatch')

    class NumpyCompatibleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Source NumPy 2 pickles use numpy._core; Isaac requires NumPy 1.
            if np.lib.NumpyVersion(np.__version__) < '2.0.0' and module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core', 1)
            return super().find_class(module, name)

    with (checkpoint / 'payload.pkl').open('rb') as stream:
        payload = NumpyCompatibleUnpickler(stream).load()
    if not isinstance(payload, CheckpointPayload) or payload.identity != expected:
        raise ValueError('Checkpoint payload identity mismatch')
    if payload.training_transitions != identity['training_transitions']:
        raise ValueError('Checkpoint transition count mismatch')
    return payload, identity


class Learner:
    """CPU JAX PPO with exact Brax loss/sequence-update semantics.

    Call begin_rollout(), then sample() for each tick, then update() with
    complete sequences. Explicit sample/update keys are available for parity
    tests and do not advance the default sampling stream.
    """

    def __init__(self, checkpoint: Path | None, ppo: dict[str, Any]):
        self._configure(ppo)
        global_key, local = jax.random.split(jax.random.PRNGKey(self.ppo['seed']))
        if checkpoint is None:
            # Brax train.py:383-390,705-721: original fresh-start initialization.
            policy_key, value_key = jax.random.split(global_key)
            self.params = losses.PPONetworkParams(
                policy=self.network.policy_network.init(policy_key),
                value=self.network.value_network.init(value_key))
            self.normalizer = running_statistics.init_state({
                'state': jax.ShapeDtypeStruct((76,), jnp.float32),
                'privileged_state': jax.ShapeDtypeStruct((106,), jnp.float32)})
            self.source_checkpoint = self.source_identity = None
            self.initialization = 'fresh'
        else:
            checkpoint = Path(checkpoint).resolve()
            payload, identity = _source_payload(checkpoint)
            self.source_checkpoint = str(checkpoint)
            self.source_identity = identity
            self.params = losses.PPONetworkParams(
                policy=payload.actor_params, value=payload.critic_params)
            self.normalizer = payload.observation_normalizer
            self.initialization = 'source_checkpoint'
        self.optimizer_state = self.optimizer.init(self.params)
        self.update_count = 0
        self.optimizer_steps = 0
        self.training_transitions = 0
        self.metrics = {}
        local = jax.random.fold_in(local, 0)
        self.local_key, self.environment_key, self.evaluation_key = jax.random.split(local, 3)
        self._sampling_key = None
        self._sgd_key = None
        self._rollout_ticks = 0
        self.runtime_identity = {
            'initialization': self.initialization,
            'initial_normalizer_count': _count(self.normalizer),
            'pretrained_weights_loaded': checkpoint is not None,
            'versions': {name: importlib.metadata.version(name)
                         for name in ['jax', 'jaxlib', 'brax', 'flax', 'optax']},
            'original_brax_sha256': {
                'losses.py': _sha256(Path(losses.__file__)),
                'ppo_networks.py': _sha256(Path(networks.__file__)),
                'running_statistics.py': _sha256(Path(running_statistics.__file__))},
        }

    def _configure(self, ppo):
        self.ppo = json.loads(json.dumps(ppo))
        for name in ['batch_size', 'num_minibatches', 'num_updates_per_batch',
                     'unroll_length', 'num_parallel_envs']:
            value = self.ppo[name]
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        sequences = self.ppo['batch_size'] * self.ppo['num_minibatches']
        if sequences != self.ppo['num_parallel_envs']:
            raise ValueError('This source adapter requires one unroll per environment per block')
        self.network = networks.make_ppo_networks(
            {'state': 76, 'privileged_state': 106}, 4,
            policy_hidden_layer_sizes=(256, 256, 256),
            value_hidden_layer_sizes=(256, 256, 256),
            policy_obs_key='state', value_obs_key='privileged_state',
            distribution_type='tanh_normal',
            preprocess_observations_fn=running_statistics.normalize)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.ppo['max_grad_norm']),
            optax.adam(learning_rate=self.ppo['learning_rate']))
        loss_fn = functools.partial(
            losses.compute_ppo_loss, ppo_network=self.network,
            entropy_cost=self.ppo['entropy_cost'],
            discounting=self.ppo['discounting'],
            reward_scaling=self.ppo['reward_scaling'],
            gae_lambda=self.ppo['gae_lambda'],
            clipping_epsilon=self.ppo['clipping_epsilon'],
            normalize_advantage=True, vf_coefficient=.5,
            clipping_epsilon_value=None, use_distributional_critic=False)
        loss_and_grad = jax.value_and_grad(loss_fn, has_aux=True)
        make_policy = networks.make_inference_fn(self.network, compute_value=True)

        def infer(params, normalizer, obs, key):
            return make_policy((normalizer, params.policy, params.value))(obs, key)

        def deterministic(params, normalizer, obs):
            return make_policy((normalizer, params.policy, params.value),
                               deterministic=True)(obs, jax.random.PRNGKey(0))[0]

        def update(params, optimizer_state, normalizer, data, key):
            # Source train.py:611-619: sampling uses old stats; SGD uses new stats.
            normalizer = running_statistics.update(normalizer, data.observation)

            def minibatch(carry, samples):
                state, parameters, rng = carry
                rng, loss_key = jax.random.split(rng)
                (_, metrics), grads = loss_and_grad(parameters, normalizer, samples, loss_key)
                metrics = {**metrics, 'grad_norm': optax.global_norm(grads),
                           'learning_rate': jnp.asarray(self.ppo['learning_rate'])}
                updates, state = self.optimizer.update(grads, state)
                return (state, optax.apply_updates(parameters, updates), rng), metrics

            def epoch(carry, _):
                state, parameters, rng = carry
                rng, permutation_key, gradient_key = jax.random.split(rng, 3)
                def shuffle(x):
                    x = jax.random.permutation(permutation_key, x)
                    return x.reshape((self.ppo['num_minibatches'], -1) + x.shape[1:])
                shuffled = jax.tree.map(shuffle, data)
                (state, parameters, _), metrics = jax.lax.scan(
                    minibatch, (state, parameters, gradient_key), shuffled,
                    length=self.ppo['num_minibatches'])
                return (state, parameters, rng), metrics

            (optimizer_state, params, _), metrics = jax.lax.scan(
                epoch, (optimizer_state, params, key), (),
                length=self.ppo['num_updates_per_batch'])
            finite_state = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((params, optimizer_state, normalizer))]))
            return params, optimizer_state, normalizer, jax.tree.map(jnp.mean, metrics), finite_state

        self._infer = jax.jit(infer)
        self._deterministic = jax.jit(deterministic)
        self._update = jax.jit(update)

    def initial_env_keys(self, n: int | None = None):
        return jax.random.split(self.environment_key, n or self.ppo['num_parallel_envs'])

    def begin_rollout(self):
        """Source single-device epoch -> training_step -> generate_unroll splits."""
        if self._sgd_key is not None:
            raise ValueError('An active rollout must be updated before beginning another')
        epoch_key, self.local_key = jax.random.split(self.local_key)
        epoch_key = jax.random.split(epoch_key, 1)[0]
        self._sgd_key, generate_key, _ = jax.random.split(epoch_key, 3)
        self._sampling_key, _ = jax.random.split(generate_key)
        self._rollout_ticks = 0

    def sample(self, obs: dict[str, Any], key=None):
        if key is None:
            if self._sampling_key is None:
                self.begin_rollout()
            key, self._sampling_key = jax.random.split(self._sampling_key)
            self._rollout_ticks += 1
        action, extras = self._infer(self.params, self.normalizer,
                                    jax.tree.map(jnp.asarray, obs), key)
        return jax.tree.map(lambda x: np.asarray(jax.device_get(x)), (action, extras))

    def deterministic(self, obs: dict[str, Any]) -> np.ndarray:
        return np.asarray(jax.device_get(self._deterministic(
            self.params, self.normalizer, jax.tree.map(jnp.asarray, obs))))

    def _validate_data(self, data):
        if not isinstance(data, types.Transition):
            raise ValueError('Expected a Brax Transition with [sequence, time] leading axes')
        shape = (self.ppo['batch_size'] * self.ppo['num_minibatches'],
                 self.ppo['unroll_length'])
        for name in ['reward', 'discount']:
            if np.shape(getattr(data, name)) != shape:
                raise ValueError(f'{name} must preserve full sequences with shape {shape}')
        for observations in [data.observation, data.next_observation]:
            for name, width in [('state', 76), ('privileged_state', 106)]:
                if np.shape(observations[name]) != shape + (width,):
                    raise ValueError(f'{name} has incorrect observation dimensions')
        for leaf in jax.tree.leaves(data):
            if not np.isfinite(np.asarray(leaf)).all():
                raise ValueError('Nonfinite transition supplied to PPO')
        for name in ['truncation', 'time_out']:
            if np.any(data.extras['state_extras'][name]):
                raise ValueError(f'{name} must reflect the source full-reset wrapper (zero)')

    def update(self, data: types.Transition, key=None) -> dict[str, float]:
        self._validate_data(data)
        if key is None:
            if self._sgd_key is None:
                self.begin_rollout()
            key = self._sgd_key
        result = self._update(self.params, self.optimizer_state, self.normalizer,
                              jax.tree.map(jnp.asarray, data), key)
        params, optimizer_state, normalizer, metrics, finite_state = result
        metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if not bool(finite_state) or not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Nonfinite original Brax PPO update; learner state was not accepted')
        self.params, self.optimizer_state, self.normalizer = params, optimizer_state, normalizer
        self.update_count += 1
        steps = self.ppo['num_minibatches'] * self.ppo['num_updates_per_batch']
        self.optimizer_steps += steps
        self.training_transitions += self.ppo['num_parallel_envs'] * self.ppo['unroll_length']
        self._sampling_key = self._sgd_key = None
        self._rollout_ticks = 0
        self.metrics = {**metrics, 'update': self.update_count,
                        'optimizer_steps_this_update': steps,
                        'optimizer_steps': self.optimizer_steps,
                        'normalizer_count': _count(normalizer)}
        return self.metrics.copy()

    def save(self, path: Path, training_transitions: int):
        """Save full learner state and a policy_runtime.Actor-compatible export.

        Simulator state is deliberately absent; this is a learner checkpoint,
        not an exact PhysX environment continuation checkpoint.
        """
        if training_transitions < 0:
            raise ValueError('training_transitions must be nonnegative')
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {'mean': np.asarray(self.normalizer.mean['state']),
                  'std': np.asarray(self.normalizer.std['state'])}
        for name, layer in self.params.policy['params'].items():
            for field, value in layer.items():
                arrays[f'{name}_{field}'] = np.asarray(value)
        np.savez(path / 'actor.npz', **arrays)
        state = {key: getattr(self, key) for key in [
            'ppo', 'source_checkpoint', 'source_identity', 'runtime_identity', 'initialization',
            'params', 'normalizer', 'optimizer_state', 'local_key',
            'environment_key', 'evaluation_key', '_sampling_key', '_sgd_key',
            '_rollout_ticks', 'update_count', 'optimizer_steps', 'metrics']}
        state['training_transitions'] = int(training_transitions)
        state['schema'] = 'physx_original_brax_learner_v1'
        temporary = path / 'learner_state.pkl.tmp'
        with temporary.open('wb') as stream:
            pickle.dump(jax.device_get(state), stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path / 'learner_state.pkl')
        identity = {
            'schema': state['schema'], 'training_transitions': int(training_transitions),
            'parent': self.source_identity, 'source_checkpoint': self.source_checkpoint,
            'optimizer_initialization': ('fresh_optax_adam' if self.initialization == 'fresh'
                                         else 'fresh_optax_adam_source_checkpoint_has_no_optimizer'),
            'exact_environment_resume': False,
            'learner_state_sha256': _sha256(path / 'learner_state.pkl'),
            'actor_sha256': _sha256(path / 'actor.npz'),
            **self.runtime_identity}
        (path / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')

    @classmethod
    def restore(cls, path: Path, ppo: dict[str, Any] | None = None):
        path = Path(path)
        identity = json.loads((path / 'identity.json').read_text())
        if _sha256(path / 'learner_state.pkl') != identity['learner_state_sha256']:
            raise ValueError('Learner checkpoint hash mismatch')
        with (path / 'learner_state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        if state.pop('schema') != 'physx_original_brax_learner_v1':
            raise ValueError('Unsupported learner checkpoint schema')
        if ppo is not None and ppo != state['ppo']:
            raise ValueError('Resume PPO configuration differs from the saved learner')
        learner = cls.__new__(cls)
        learner._configure(state['ppo'])
        array_fields = {'params', 'normalizer', 'optimizer_state', 'local_key',
                        'environment_key', 'evaluation_key', '_sampling_key', '_sgd_key'}
        for key, value in state.items():
            if key in array_fields:
                value = jax.tree.map(jnp.asarray, value)
            setattr(learner, key, value)
        if not hasattr(learner, 'initialization'):
            learner.initialization = 'source_checkpoint'
        return learner
