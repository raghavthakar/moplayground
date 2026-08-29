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
import optax
from brax.training.acme import running_statistics, specs

from moplayground.moppo import factory
from moplayground.moppo.teacher_demos import sample_bc_batch


def build_normalizer(buffer: dict) -> running_statistics.RunningStatisticsState:
    """Frozen observation normalizer estimated from the demo buffer."""
    obs_dim = int(buffer['observation']['state'].shape[-1])
    normalizer = running_statistics.init_state(
        {'state': specs.Array((obs_dim,), jnp.float32)}
    )
    return running_statistics.update(normalizer, buffer['observation'])


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
