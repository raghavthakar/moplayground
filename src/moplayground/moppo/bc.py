"""Behavioral cloning of teacher demos into a MORLAX hypernetwork.

This is the migration bridge: given an offline buffer of (observation,
raw_action, preference) tuples from frozen exploration teachers, train a cold
MORLAX hypernetwork by pure supervised BC so that at each teacher's labeled
preference it emits that teacher's actions. The resulting hypernetwork params
are used verbatim as MORLAX's policy init (see morlax.train's
``init_hypernetwork_params``); MORLAX training itself is untouched.

The BC loss vmaps ``policy_network.apply`` over the hypernetwork's per-preference
batch of policy params -- the hypernet emits one policy per preference in a
batch, so the policy network must be mapped over that leading axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.acme import running_statistics, specs

from moplayground.moppo import acting, factory
from moplayground.moppo.teacher_demos import sample_bc_batch
from moplayground.utils.pareto import compute_pareto_statistics


def build_normalizer(buffer: dict) -> running_statistics.RunningStatisticsState:
    """Frozen observation normalizer estimated from the demo buffer."""
    obs_dim = int(buffer['observation']['state'].shape[-1])
    normalizer = running_statistics.init_state(
        {'state': specs.Array((obs_dim,), jnp.float32)}
    )
    return running_statistics.update(normalizer, buffer['observation'])


def standard_eval_preferences(num_objectives: int) -> list[np.ndarray]:
    """Fixed preference anchors for comparable BC head-start measurement."""
    if num_objectives == 2:
        return [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.5, 0.5], dtype=np.float32),
        ]
    return [np.ones(num_objectives, dtype=np.float32) / num_objectives]


def _unlock_fraction(rewards: np.ndarray, thresholds: list[float] | None) -> dict:
    """Fraction of evaluated preferences that cross each threshold."""
    if thresholds is None or len(thresholds) == 0:
        return {}
    thr = np.asarray(thresholds, dtype=float)
    # rewards must be (num_preferences, num_objectives); bail out otherwise so a
    # shape surprise degrades to "no unlock stats" instead of crashing the run.
    if rewards.ndim != 2 or rewards.shape[1] < len(thr):
        return {}
    out = {}
    for i in range(len(thr)):
        out[f'unlock_obj_{i}'] = float((rewards[:, i] >= thr[i]).mean())
    out['unlock_both'] = float((rewards[:, : len(thr)] >= thr).all(axis=1).mean())
    return out


def evaluate_hypernetwork(
    networks: factory.MORLAXNetworks,
    normalizer: running_statistics.RunningStatisticsState,
    hypernet_params,
    env,
    preferences: list,
    *,
    episode_length: int,
    action_repeat: int,
    seed: int = 0,
    thresholds: list[float] | None = None,
    teacher_objectives: list[np.ndarray] | None = None,
    num_eval_envs: int = 64,
) -> dict:
    """Roll out a hypernetwork at fixed preferences; return summary metrics.

    Each preference is rolled out in ``num_eval_envs`` parallel envs and the
    per-objective episodic returns are averaged, giving one ``(num_objectives,)``
    return vector per preference. Uses the same gated training env as MORLAX so
    BC numbers are comparable to finetune step-0 ``eval/hypervolume``.
    """
    from moplayground.learning.wrappers import MultiObjectiveEvalWrapper

    inference_fn = factory.make_hypernetwork_inference_fn(networks)
    params = (normalizer, hypernet_params)
    prefs = [np.asarray(p, dtype=np.float32) for p in preferences]
    eval_env = MultiObjectiveEvalWrapper(env)
    unroll_len = episode_length // action_repeat

    @jax.jit
    def rollout(pref, key):
        directive = jnp.asarray(pref, dtype=jnp.float32)
        policy = inference_fn(params, directive, deterministic=True)
        state = eval_env.reset(jax.random.split(key, num_eval_envs))
        final_state = acting.generate_unroll(
            eval_env, state, policy, directive, key, unroll_length=unroll_len
        )[0]
        # episode_metrics['reward'] is (num_eval_envs, num_objectives); average
        # the parallel rollouts into one return vector for this preference.
        episodic_return = final_state.info['eval_metrics'].episode_metrics['reward']
        return jnp.mean(episodic_return, axis=0)

    key = jax.random.PRNGKey(seed)
    rewards = []
    for pref in prefs:
        key, sub = jax.random.split(key)
        rewards.append(np.asarray(rollout(pref, sub), dtype=float))
    rewards = np.stack(rewards, axis=0).astype(float)  # (num_preferences, num_objectives)
    directives = np.stack(prefs, axis=0).astype(float)

    try:
        stats = compute_pareto_statistics(rewards)
        hv = float(stats.hypervolume)
        sparsity = float(stats.sparsity)
    except Exception:
        hv, sparsity = float('nan'), float('nan')

    report = {
        'hypervolume': hv,
        'sparsity': sparsity,
        'num_points': int(len(prefs)),
        'forward_max': float(rewards[:, 0].max()) if rewards.shape[1] >= 1 else float('nan'),
        'jump_max': float(rewards[:, 1].max()) if rewards.shape[1] >= 2 else float('nan'),
        'rewards': rewards.tolist(),
        'directives': directives.tolist(),
        **_unlock_fraction(rewards, thresholds),
    }

    if teacher_objectives is not None:
        retention = []
        for i, (pref, t_obj) in enumerate(zip(prefs, teacher_objectives)):
            student = rewards[i]
            teacher = np.asarray(t_obj, dtype=float)
            retention.append({
                'preference': pref.tolist(),
                'teacher_objectives': teacher.tolist(),
                'student_returns': student.tolist(),
                'ratio': (student / np.maximum(teacher, 1e-6)).tolist(),
            })
        report['teacher_retention'] = retention
    return report


def _log_eval_report(run, prefix: str, report: dict, step: int = 0):
    """Log a compact scalar view of ``evaluate_hypernetwork`` to W&B."""
    if run is None:
        return
    metrics = {
        f'{prefix}/hypervolume': report['hypervolume'],
        f'{prefix}/sparsity': report.get('sparsity', float('nan')),
        f'{prefix}/forward_max': report['forward_max'],
        f'{prefix}/jump_max': report['jump_max'],
        f'{prefix}/num_points': report['num_points'],
    }
    for key, val in report.items():
        if key.startswith('unlock_'):
            metrics[f'{prefix}/{key}'] = val
    run.log(metrics, step=step)


def pretrain_hypernetwork(
    networks: factory.MORLAXNetworks,
    normalizer: running_statistics.RunningStatisticsState,
    buffer: dict,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int = 0,
    log_every: int = 1000,
    run=None,
):
    """Supervised-BC a MORLAX hypernetwork on ``buffer``; return its params.

    ``buffer`` is the jax demo buffer from ``teacher_demos.demo_buffer_to_jax``:
    ``{'observation': {'state': (N, obs)}, 'raw_action': (N, act),
    'directive': (N, num_obj)}``.
    """
    hyper = networks.hypernetwork
    policy_apply = jax.vmap(networks.policy_network.apply, in_axes=(None, 0, 0))
    parametric_action_distribution = networks.parametric_action_distribution

    def bc_loss(params, batch):
        policy_params, _ = hyper.apply(params, batch['directive'])
        logits = policy_apply(normalizer, policy_params, batch['observation'])
        log_prob = parametric_action_distribution.log_prob(
            logits, batch['raw_action']
        )
        return -jnp.mean(log_prob)

    params = hyper.init(jax.random.PRNGKey(seed))
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, key):
        batch = sample_bc_batch(key, buffer, batch_size)
        loss, grads = jax.value_and_grad(bc_loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    key = jax.random.PRNGKey(seed)
    for step in range(1, steps + 1):
        key, sub = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, sub)
        if step == 1 or step % log_every == 0 or step == steps:
            loss_f = float(loss)
            print(f'  [bc {step}/{steps}] loss={loss_f:.4f}')
            if run is not None:
                run.log({'bc/loss': loss_f}, step=step)
    return params
