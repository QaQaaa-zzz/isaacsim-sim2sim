"""CPU-only contracts for the original Brax learner at the PhysX boundary."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import importlib.util
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses, networks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = Path('/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826')


def learner(config=None):
    assert importlib.util.find_spec('brax_physx_learner') is not None, 'Original Brax learner adapter is missing'
    from brax_physx_learner import Learner
    cfg = json.loads((SOURCE / 'resolved_config.json').read_text())['ppo']
    cfg.update(num_parallel_envs=4, batch_size=2, num_minibatches=2,
               num_updates_per_batch=2, unroll_length=4)
    cfg.update(config or {})
    return Learner(SOURCE / 'checkpoints/transition_4988928', cfg)


def observations(net, shape):
    rng = np.random.default_rng(418)
    return {k: np.asarray(net.normalizer.mean[k]) + np.asarray(net.normalizer.std[k]) *
            rng.normal(0, .1, shape + (width,)).astype(np.float32)
            for k, width in [('state', 76), ('privileged_state', 106)]}


def batch(net):
    b = net.ppo['batch_size'] * net.ppo['num_minibatches']
    t = net.ppo['unroll_length']
    obs = observations(net, (b, t))
    action, extras = net.sample(obs, key=jax.random.PRNGKey(919))
    discount = np.ones((b, t), np.float32)
    discount[0, 1] = 0  # A real episode boundary inside an intact sequence.
    return types.Transition(
        observation=obs, action=action,
        reward=np.linspace(-2, 3, b*t, dtype=np.float32).reshape(b, t),
        discount=discount, next_observation=observations(net, (b, t)),
        extras={'policy_extras': extras, 'state_extras': {
            'truncation': np.zeros((b, t), np.float32),
            'time_out': np.zeros((b, t), np.float32)}})


def assert_tree_close(left, right, atol=2e-6):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, atol=atol, rtol=2e-6)


def test_original_checkpoint_actor_critic_and_sample_parity():
    net = learner()
    obs = observations(net, (5,))
    key = jax.random.PRNGKey(42)
    original = networks.make_inference_fn(net.network, compute_value=True)(
        (net.normalizer, net.params.policy, net.params.value))
    expected, extra = original(obs, key)
    actual, actual_extra = net.sample(obs, key=key)
    assert_tree_close(actual, expected)
    assert_tree_close(actual_extra, extra)
    assert actual_extra['value'].shape == (5,)
    assert net.params.value['params']['hidden_0']['kernel'].shape == (106, 256)
    assert int(net.normalizer.count.lo) == 4988928


def test_one_update_matches_original_brax_loss_with_new_normalizer():
    net = learner(dict(num_parallel_envs=2, num_minibatches=1, num_updates_per_batch=1))
    data = batch(net)
    key = jax.random.PRNGKey(23)
    norm = running_statistics.update(net.normalizer, jax.tree.map(jnp.asarray, data.observation))
    _, permutation_key, gradient_key = jax.random.split(key, 3)
    shuffled = jax.tree.map(lambda x: jax.random.permutation(permutation_key, jnp.asarray(x)), data)
    _, loss_key = jax.random.split(gradient_key)
    def original_loss(params):
        return losses.compute_ppo_loss(
            params, norm, shuffled, loss_key, net.network,
            entropy_cost=.01, discounting=.99, reward_scaling=.1,
            gae_lambda=.95, clipping_epsilon=.2,
            normalize_advantage=True, vf_coefficient=.5,
            clipping_epsilon_value=None, use_distributional_critic=False)
    (expected_loss, _), grads = jax.value_and_grad(original_loss, has_aux=True)(net.params)
    updates, expected_opt = net.optimizer.update(grads, net.optimizer_state)
    expected_params = optax.apply_updates(net.params, updates)
    metrics = net.update(data, key=key)
    assert_tree_close(net.params, expected_params)
    assert_tree_close(net.optimizer_state, expected_opt)
    assert_tree_close(net.normalizer, norm)
    np.testing.assert_allclose(metrics['total_loss'], expected_loss, rtol=2e-5)
    assert metrics['optimizer_steps_this_update'] == 1
    assert metrics['normalizer_count'] == 4988936


def test_sequence_epochs_and_full_state_roundtrip(tmp_path):
    net = learner()
    data = batch(net)
    before = jax.tree.map(np.array, net.params)
    net.begin_rollout()
    net.sample(observations(net, (4,)))
    metrics = net.update(data)
    assert metrics['optimizer_steps_this_update'] == 4
    assert metrics['normalizer_count'] == 4988944
    assert all(np.isfinite(x) for x in metrics.values())
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(net.params)))
    target = tmp_path / 'saved'
    net.save(target, training_transitions=16)
    from brax_physx_learner import Learner
    restored = Learner.restore(target)
    assert restored.training_transitions == 16
    assert_tree_close(net.params, restored.params, atol=0)
    assert_tree_close(net.normalizer, restored.normalizer, atol=0)
    assert_tree_close(net.optimizer_state, restored.optimizer_state, atol=0)
    obs = observations(net, (4,))
    from policy_runtime import Actor
    np.testing.assert_allclose(Actor(target/'actor.npz')(obs['state']), net.deterministic(obs), atol=3e-6)
    net.begin_rollout(); restored.begin_rollout()
    assert_tree_close(net.sample(obs), restored.sample(obs), atol=0)
    assert_tree_close(net.update(data), restored.update(data), atol=0)
    assert_tree_close(net.params, restored.params, atol=0)


def test_source_nested_prng_keys_and_initial_environment_keys():
    net = learner()
    _, local = jax.random.split(jax.random.PRNGKey(820701))
    local = jax.random.fold_in(local, 0)
    local, env_key, _ = jax.random.split(local, 3)
    assert_tree_close(net.initial_env_keys(4), jax.random.split(env_key, 4), atol=0)
    epoch, _ = jax.random.split(local)
    epoch = jax.random.split(epoch, 1)[0]
    _, generate, _ = jax.random.split(epoch, 3)
    generate, _ = jax.random.split(generate)  # One source generate_unroll block.
    step_key, _ = jax.random.split(generate)
    obs = observations(net, (4,))
    expected = net.sample(obs, key=step_key)
    net.begin_rollout()
    assert_tree_close(net.sample(obs), expected, atol=0)


def test_fresh_initialization_uses_brax_random_weights_and_empty_statistics():
    # Catches accidental checkpoint inheritance when the user requested fresh PPO.
    from brax_physx_learner import Learner
    cfg = json.loads((SOURCE / 'resolved_config.json').read_text())['ppo']
    cfg.update(num_parallel_envs=4, batch_size=2, num_minibatches=2,
               num_updates_per_batch=2, unroll_length=4)
    net = Learner(None, cfg)
    original = networks.make_ppo_networks(
        {'state': 76, 'privileged_state': 106}, 4,
        policy_hidden_layer_sizes=(256, 256, 256),
        value_hidden_layer_sizes=(256, 256, 256),
        policy_obs_key='state', value_obs_key='privileged_state',
        distribution_type='tanh_normal',
        preprocess_observations_fn=running_statistics.normalize)
    global_key, _ = jax.random.split(jax.random.PRNGKey(820701))
    policy_key, value_key = jax.random.split(global_key)
    assert_tree_close(net.params.policy, original.policy_network.init(policy_key), atol=0)
    assert_tree_close(net.params.value, original.value_network.init(value_key), atol=0)
    assert int(net.normalizer.count.lo) == int(net.normalizer.count.hi) == 0
    for key in ('state', 'privileged_state'):
        np.testing.assert_array_equal(net.normalizer.mean[key], 0)
        np.testing.assert_array_equal(net.normalizer.std[key], 1)
    assert all(np.count_nonzero(np.asarray(x)) == 0 for x in jax.tree.leaves(net.optimizer_state))
    assert net.source_checkpoint is None and net.source_identity is None
    assert net.training_transitions == net.update_count == net.optimizer_steps == 0
    other = Learner(None, cfg | {'seed': 820702})
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(net.params), jax.tree.leaves(other.params)))


def test_fresh_training_counts_only_new_samples_and_saves_no_pretrained_parent(tmp_path):
    from brax_physx_learner import Learner
    cfg = json.loads((SOURCE / 'resolved_config.json').read_text())['ppo']
    cfg.update(num_parallel_envs=4, batch_size=2, num_minibatches=2,
               num_updates_per_batch=2, unroll_length=4)
    net = Learner(None, cfg)
    metrics = net.update(batch(net), key=jax.random.PRNGKey(23))
    assert metrics['normalizer_count'] == 16
    assert metrics['optimizer_steps'] == 4
    folder = tmp_path / 'fresh'
    net.save(folder, 16)
    identity = json.loads((folder / 'identity.json').read_text())
    assert identity['parent'] is None and identity['source_checkpoint'] is None
    assert identity['initialization'] == 'fresh'
    restored = Learner.restore(folder)
    assert restored.initialization == 'fresh'
    assert_tree_close(net.params, restored.params, atol=0)
    assert_tree_close(net.normalizer, restored.normalizer, atol=0)


@pytest.mark.parametrize('bad_kind', ['flat', 'nonfinite', 'truncation'])
def test_invalid_transition_rejected_before_learning(bad_kind):
    net = learner()
    data = batch(net)
    before = jax.tree.map(np.array, net.params)
    if bad_kind == 'flat':
        data = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), data)
    elif bad_kind == 'nonfinite':
        data.reward[0, 0] = np.nan
    else:
        data.extras['state_extras']['truncation'][0, 1] = 1
    with pytest.raises(ValueError):
        net.update(data)
    assert_tree_close(net.params, before, atol=0)
    assert int(net.normalizer.count.lo) == 4988928
