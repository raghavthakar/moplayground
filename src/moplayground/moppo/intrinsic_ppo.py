"""IntrinsicPPO: single-objective PPO trained purely on RND novelty.

Training never uses the environment's multi-objective (extrinsic) reward.
A separate evaluator rolls the same policy out on the *ungated* MO env and
reports true objective episodic returns + unlock rates vs thresholds.

This is Stage 1 of the exploration stack (breakthrough detection).
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional, Sequence, Tuple

from absl import logging
from brax import base
from brax import envs
from brax.training import acting as brax_acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from moplayground.learning.wrappers import MultiObjectiveEpisodeWrapper
from moplayground.learning.wrappers import MultiObjectiveEvalWrapper
from moplayground.moppo import rnd as rnd_lib
from brax.envs.wrappers.training import VmapWrapper
from mujoco_playground import wrapper


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    rnd_state: rnd_lib.RNDState
    rnd_target_params: Params  # frozen random network; never trained
    env_steps: types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


def wrap_extrinsic_eval_env(
    env,
    episode_length: int,
    action_repeat: int = 1,
    randomization_fn=None,
):
    """Wrap for extrinsic breakthrough eval — no episodic threshold gating."""
    del randomization_fn
    env = VmapWrapper(env)
    env = MultiObjectiveEpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env)
    return env


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified in intrinsic_ppo.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    wrap_for_training = wrap_env_fn or envs.training.wrap
    return wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )


class ExtrinsicBreakthroughEvaluator:
    """Roll out a directive-free policy; report true MO returns + unlocks.

    Uses an ungated eval env so ``episode_metrics['reward']`` is the dense
    extrinsic objective vector. Compares each eval episode's return against
    ``unlock_thresholds`` when provided.
    """

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
        num_objs: int,
        unlock_thresholds: Optional[Sequence[float]] = None,
    ):
        self._key = key
        self._eval_walltime = 0.0
        self.num_eval_envs = num_eval_envs
        self.num_objs = num_objs
        self.unlock_thresholds = (
            None
            if unlock_thresholds is None
            else jnp.asarray(unlock_thresholds, dtype=jnp.float32)
        )

        eval_env = MultiObjectiveEvalWrapper(eval_env)

        def generate_eval_unroll(policy_params, key: PRNGKey):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return brax_acting.generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params),
                key,
                unroll_length=episode_length // action_repeat,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics: Metrics,
    ) -> Metrics:
        self._key, unroll_key = jax.random.split(self._key)
        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        self._eval_walltime += epoch_eval_time

        rewards = np.asarray(eval_metrics.episode_metrics['reward'])
        # Dummy directives so plot_mo_progress still works.
        directives = np.ones_like(rewards) / max(self.num_objs, 1)

        metrics = {
            'reward': rewards,
            'directive': directives,
            'eval/avg_episode_length': float(np.mean(eval_metrics.episode_steps)),
            'eval/epoch_eval_time': epoch_eval_time,
            'eval/sps': self._steps_per_unroll / max(epoch_eval_time, 1e-8),
            'eval/walltime': self._eval_walltime,
        }

        if self.unlock_thresholds is not None:
            thr = np.asarray(self.unlock_thresholds, dtype=float)
            # rewards: [num_eval_envs, num_objs]
            unlocked = rewards >= thr[None, :]
            for i in range(self.num_objs):
                metrics[f'eval/unlock/obj_{i}'] = float(np.mean(unlocked[:, i]))
            metrics['eval/unlock/any'] = float(np.mean(np.any(unlocked, axis=1)))
            metrics['eval/unlock/all'] = float(np.mean(np.all(unlocked, axis=1)))
            for i in range(self.num_objs):
                metrics[f'eval/return/obj_{i}/mean'] = float(np.mean(rewards[:, i]))
                metrics[f'eval/return/obj_{i}/max'] = float(np.max(rewards[:, i]))

        metrics.update(training_metrics)
        return metrics


def _intrinsic_actor_step(
    env,
    env_state,
    policy,
    key,
    rnd_module,
    rnd_target_params,
    rnd_state,
    extra_fields=(),
):
    """Env step with RND novelty as the scalar training reward."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    intrinsic = rnd_lib.novelty_reward(
        rnd_module, rnd_target_params, rnd_state, env_state.obs, actions
    )
    # Env may return a vector reward; keep it only for diagnostics.
    extrinsic = nstate.reward
    state_extras = {x: nstate.info[x] for x in extra_fields}
    state_extras['extrinsic_reward'] = extrinsic
    return nstate, types.Transition(
        observation=env_state.obs,
        action=actions,
        reward=intrinsic,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )


def _generate_intrinsic_unroll(
    env,
    env_state,
    policy,
    key,
    unroll_length,
    rnd_module,
    rnd_target_params,
    rnd_state,
    extra_fields=(),
):
    def f(carry, _):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = _intrinsic_actor_step(
            env,
            state,
            policy,
            current_key,
            rnd_module,
            rnd_target_params,
            rnd_state,
            extra_fields=extra_fields,
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length
    )
    return final_state, data


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    wrap_env: bool = True,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    save_checkpoint_path: Optional[str] = None,
    run_evals: bool = True,
    # RND / breakthrough
    rnd_hidden_layer_sizes: Sequence[int] = (256, 256),
    rnd_output_size: int = 64,
    rnd_learning_rate: float = 1e-4,
    unlock_thresholds: Optional[Sequence[float]] = None,
):
    """Train IntrinsicPPO; evaluate breakthroughs on ungated extrinsic returns."""
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * process_count
    assert num_envs % device_count == 0

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = int(
        np.ceil(
            num_timesteps
            / (
                num_evals_after_init
                * env_step_per_training_step
                * max(num_resets_per_eval, 1)
            )
        )
    )

    key = jax.random.PRNGKey(seed)
    key = jax.random.fold_in(key, process_id)
    key, key_env, eval_key, key_policy, key_value, key_rnd = jax.random.split(key, 6)

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )

    # Extrinsic eval: never apply episodic threshold gating.
    if eval_env is None:
        eval_env = environment
    eval_env = wrap_extrinsic_eval_env(
        eval_env, episode_length=episode_length, action_repeat=action_repeat
    )

    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )
    env_state = reset_fn(key_envs)

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    num_objectives = int(env_state.reward.shape[-1]) if env_state.reward.ndim > 1 else 1
    # Leading dims: (local_devices, envs_per_device, ...)
    sample_obs = jax.tree_util.tree_map(lambda x: x[0, 0], env_state.obs)
    obs_dim = int(rnd_lib.flatten_obs(sample_obs).shape[-1])

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    ppo_net = network_factory(
        observation_size=obs_shape,
        action_size=env.action_size,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_net)

    rnd_module, rnd_target_params, rnd_state, rnd_optimizer = rnd_lib.init_rnd(
        key=key_rnd,
        obs_dim=obs_dim,
        action_size=env.action_size,
        hidden_layer_sizes=rnd_hidden_layer_sizes,
        output_size=rnd_output_size,
        learning_rate=rnd_learning_rate,
    )

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_net,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )
    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    def minibatch_step(carry, data: types.Transition, normalizer_params):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state,
        )
        return (optimizer_state, params, key), metrics

    def sgd_step(carry, unused_t, data: types.Transition, normalizer_params):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(carry, unused_t):
        training_state, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))

        def scan_unroll(carry_u, _):
            current_state, current_key = carry_u
            current_key, next_key = jax.random.split(current_key)
            next_state, data = _generate_intrinsic_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                rnd_module,
                training_state.rnd_target_params,
                training_state.rnd_state,
                extra_fields=('truncation', 'episode_metrics', 'episode_done'),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            scan_unroll,
            (state, key_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        # Update RND predictor on the collected (s, a) batch.
        # data.observation / action: [batch, unroll, ...]
        obs_flat = jax.tree_util.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]), data.observation
        )
        act_flat = data.action.reshape((-1,) + data.action.shape[2:])
        new_rnd_state, rnd_loss = rnd_lib.update_predictor(
            rnd_module,
            training_state.rnd_target_params,
            training_state.rnd_state,
            rnd_optimizer,
            obs_flat,
            act_flat,
        )
        metrics = {
            **metrics,
            'rnd_loss': rnd_loss,
            'intrinsic_reward_mean': jnp.mean(data.reward),
        }

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            rnd_state=new_rnd_state,
            rnd_target_params=training_state.rnd_target_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(training_state, state, key):
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    training_walltime = 0.0

    def training_epoch_with_timing(training_state, env_state, key):
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_net.policy_network.init(key_policy),
        value=ppo_net.value_network.init(key_value),
    )
    obs_shape_specs = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape_specs),
        rnd_state=rnd_state,
        rnd_target_params=rnd_target_params,
        env_steps=types.UInt64(hi=0, lo=0),
    )
    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    evaluator = ExtrinsicBreakthroughEvaluator(
        eval_env=eval_env,
        eval_policy_fn=functools.partial(
            make_policy, deterministic=deterministic_eval
        ),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        num_objs=num_objectives,
        unlock_thresholds=unlock_thresholds,
    )

    current_step = 0
    metrics = {}
    if process_id == 0 and num_evals > 1 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    policy_params_fn(current_step, make_policy, params)

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)
        print(f'starting iteration {it} {time.time() - xt}')

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, key = jax.random.split(key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            training_state, env_state, training_metrics = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        params = _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))
        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = training_metrics
            if run_evals:
                metrics = evaluator.run_evaluation(params, training_metrics)
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`={num_timesteps}.'
        )

    pmap.assert_is_replicated(training_state)
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return make_policy, params, metrics
