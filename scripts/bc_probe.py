#!/usr/bin/env python3
"""Standalone probe: does behavior cloning transfer teacher skills into a
MORLAX hypernetwork?

This deliberately isolates the transfer question from PPO. It trains a COLD
MORLAX hypernetwork on the offline teacher demo buffer with a pure supervised
BC loss -- no PPO, no pmap, no environment interaction during training. Every
demo transition carries its teacher's fixed preference label, so the hypernet
learns to emit teacher-like actions at each anchor preference.

It then evaluates the cloned hypernetwork on the (ungated) env at each teacher's
labeled preference and prints its per-objective return next to that FROZEN
teacher's own return on the same env and seed (the reference). If BC transfer
works, the student should approach its teacher at each anchor:

    w=[1,0]  -> reproduces the run teacher's Forward Distance
    w=[0,1]  -> reproduces the jump teacher's Jump Height
    w=[.5,.5]-> reproduces the balanced teacher

which is something warm-start (one policy baked into every preference) cannot do.
All evals share a single fixed seed so the numbers are comparable across steps.

Usage:
    python -m scripts.bc_probe config/morlax/mohopper_sparse_bc.yaml \
        --steps 20000 --batch 512 --lr 1e-3 --eval-every 2000
"""

import argparse
import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training.acme import running_statistics, specs

import moplayground as mop
import minimal_mjx as mm
from moplayground.moppo import factory
from moplayground.moppo.teacher_demos import (
    demo_buffer_to_jax,
    load_demo_buffer,
    load_ppo_teacher_policy,
    sample_bc_batch,
)

# Fixed evaluation seed: every eval (teacher references and every student
# checkpoint) starts from the same initial states, so the reported numbers are
# comparable step-to-step and the curve reflects learning rather than luck.
EVAL_SEED = 12345


def _anchors(distill):
    """Return [(name, preference, checkpoint), ...] for each teacher."""
    anchors = []
    for t in distill.get('teachers', []):
        anchors.append(
            (
                t.get('name', 'teacher'),
                [float(x) for x in t['preference']],
                t['checkpoint'],
            )
        )
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=str, help='Training YAML with distill_params')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-envs', type=int, default=128)
    args = ap.parse_args()

    cfg = mop.utils.read_config(args.config)
    distill = cfg.learning_params.morlax_params.distill_params
    anchors = _anchors(distill)
    if not anchors:
        raise ValueError('distill_params.teachers is empty.')

    buffer = demo_buffer_to_jax(load_demo_buffer(distill.demo_buffer))
    n_demos = int(buffer['raw_action'].shape[0])
    print(f'Loaded {n_demos} demo transitions.')
    for name, pref, _ in anchors:
        print(f'  anchor {name}: w={pref}')

    mm.utils.setupGPU.run_setup()

    # Ungated eval env: measure the true skill, not the gated reward.
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.env_config.reward.episodic_threshold.enabled = False
    env, _ = mop.envs.create_environment(eval_cfg, for_training=True)
    episode_length = int(cfg.learning_params.base_ppo_params.episode_length)
    action_repeat = int(cfg.learning_params.base_ppo_params.action_repeat)
    env = mop.learning.training.mo_wrapper(
        env, episode_length=episode_length, action_repeat=action_repeat
    )

    labels = list(cfg.env_config.reward.optimization.get('labels', ['obj0', 'obj1']))
    num_obj = len(anchors[0][1])
    obs_dim = int(buffer['observation']['state'].shape[-1])

    # Frozen normalizer from demo observations (BC and eval share it).
    normalizer = running_statistics.init_state(
        {'state': specs.Array((obs_dim,), jnp.float32)}
    )
    normalizer = running_statistics.update(normalizer, buffer['observation'])

    network_params = dict(cfg.learning_params.morlax_params.network_params)
    networks = factory.make_morlax_networks(
        observation_size={'state': (obs_dim,)},
        action_size=env.action_size,
        num_objectives=num_obj,
        key=jax.random.PRNGKey(0),
        preprocess_observations_fn=running_statistics.normalize,
        **network_params,
    )
    hyper = networks.hypernetwork
    policy_net = networks.policy_network
    dist = networks.parametric_action_distribution
    inference_fn = factory.make_hypernetwork_inference_fn(networks)

    params = hyper.init(jax.random.PRNGKey(0))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    # BC loss: the hypernet emits one policy per (batched) preference, so
    # policy_net.apply must be vmapped over that batch axis.
    policy_apply = jax.vmap(policy_net.apply, in_axes=(None, 0, 0))

    def bc_loss(params, batch):
        policy_params, _ = hyper.apply(params, batch['directive'])
        logits = policy_apply(normalizer, policy_params, batch['observation'])
        log_prob = dist.log_prob(logits, batch['raw_action'])
        return -jnp.mean(log_prob)

    @jax.jit
    def train_step(params, opt_state, key):
        batch = sample_bc_batch(key, buffer, args.batch)
        loss, grads = jax.value_and_grad(bc_loss)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    # Shared episode rollout: run a policy(obs, key) -> (action, extras) from the
    # same fixed initial states and return the mean per-objective episode return.
    # Reward is accumulated only while the first episode is still active so that
    # BraxAutoReset restarts do not double-count.
    eval_key = jax.random.PRNGKey(EVAL_SEED)

    def rollout_return(policy, num_envs):
        def step(carry, _):
            state, ret, active = carry
            action, _ = policy(state.obs, eval_key)
            nstate = env.step(state, action)
            ret = ret + nstate.reward * active[:, None]
            active = active * (1.0 - nstate.done)
            return (nstate, ret, active), None

        state = env.reset(jax.random.split(eval_key, num_envs))
        init = (state, jnp.zeros((num_envs, num_obj)), jnp.ones((num_envs,)))
        (_, ret, _), _ = jax.lax.scan(step, init, None, length=episode_length)
        return ret.mean(axis=0)

    @partial(jax.jit, static_argnums=(2,))
    def eval_student(params, pref, num_envs):
        policy = inference_fn(
            params=(normalizer, params), directive=pref, deterministic=True
        )
        return rollout_return(policy, num_envs)

    def make_teacher_eval(teacher_policy):
        @partial(jax.jit, static_argnums=(0,))
        def _run(num_envs):
            return rollout_return(teacher_policy, num_envs)

        return _run

    # One-time reference: score each FROZEN teacher on the same env / seed, so
    # every student row below can be read against the behavior it is cloning.
    print('Frozen-teacher reference returns (same env, same eval seed):')
    teacher_refs = {}
    for name, pref, checkpoint in anchors:
        teacher_policy = load_ppo_teacher_policy(checkpoint, deterministic=True)
        ref = np.asarray(make_teacher_eval(teacher_policy)(args.eval_envs))
        teacher_refs[name] = ref
        desc = ', '.join(f'{l}={v:.1f}' for l, v in zip(labels, ref))
        print(f'  w={pref} [{name}]: {desc}')

    def run_eval(params):
        for name, pref, _ in anchors:
            r = np.asarray(
                eval_student(params, jnp.asarray(pref, jnp.float32), args.eval_envs)
            )
            ref = teacher_refs[name]
            desc = ', '.join(
                f'{l}={s:.1f} (tch {t:.1f})' for l, s, t in zip(labels, r, ref)
            )
            print(f'  w={pref} [{name}]: {desc}')

    print('Evaluating cold hypernetwork (sanity baseline):')
    run_eval(params)

    key = jax.random.PRNGKey(0)
    for step in range(1, args.steps + 1):
        key, tk = jax.random.split(key)
        params, opt_state, loss = train_step(params, opt_state, tk)
        if step == 1 or step % args.eval_every == 0:
            print(f'[step {step}] bc_loss={float(loss):.4f}')
            run_eval(params)

    print('BC probe complete.')


if __name__ == '__main__':
    main()
