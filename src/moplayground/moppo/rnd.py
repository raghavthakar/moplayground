"""Random Network Distillation (RND) for intrinsic novelty rewards.

Fixed random target network and trainable predictor. Novelty at ``(s, a)`` is
prediction error, normalized by a running standard deviation for PPO.
"""

from typing import Mapping, Sequence, Tuple

from brax.training.types import Params
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax


def flatten_obs(obs) -> jnp.ndarray:
    """Extract a flat observation vector (handles dict or array obs)."""
    if isinstance(obs, Mapping):
        return obs['state']
    return obs


class RNDMLP(nn.Module):
    """Small MLP used for both the fixed target and the trainable predictor."""

    hidden_layer_sizes: Sequence[int] = (256, 256)
    output_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for width in self.hidden_layer_sizes:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.output_size)(x)


@flax.struct.dataclass
class RNDState:
    """Predictor params + optimizer state + running novelty stats."""

    predictor_params: Params
    optimizer_state: optax.OptState
    novelty_mean: jnp.ndarray
    novelty_var: jnp.ndarray
    novelty_count: jnp.ndarray


def init_rnd(
    key: jax.Array,
    obs_dim: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    output_size: int = 64,
    learning_rate: float = 1e-4,
) -> Tuple[RNDMLP, Params, RNDState, optax.GradientTransformation]:
    """Create target (fixed) + predictor (trainable) and optimizer."""
    key_target, key_pred = jax.random.split(key)
    module = RNDMLP(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        output_size=output_size,
    )
    dummy = jnp.zeros((1, obs_dim + action_size))
    target_params = module.init(key_target, dummy)
    predictor_params = module.init(key_pred, dummy)
    optimizer = optax.adam(learning_rate)
    state = RNDState(
        predictor_params=predictor_params,
        optimizer_state=optimizer.init(predictor_params),
        novelty_mean=jnp.zeros(()),
        novelty_var=jnp.ones(()),
        novelty_count=jnp.array(1e-4),
    )
    return module, target_params, state, optimizer


def _sa_features(obs, action: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([flatten_obs(obs), action], axis=-1)


def novelty_reward(
    module: RNDMLP,
    target_params: Params,
    rnd_state: RNDState,
    obs,
    action: jnp.ndarray,
    eps: float = 1e-8,
    scale: float = 1.0,
) -> jnp.ndarray:
    """Normalized novelty used as the intrinsic PPO reward (no grad).

    ``scale`` multiplies the std-normalized prediction error (the usual RND
    coefficient). Default 1.0 leaves the running-std unit scale unchanged.
    """
    x = _sa_features(obs, action)
    target = module.apply(target_params, x)
    pred = module.apply(rnd_state.predictor_params, x)
    raw = jnp.mean(jnp.square(target - pred), axis=-1)
    std = jnp.sqrt(rnd_state.novelty_var + eps)
    return jax.lax.stop_gradient(scale * raw / std)


def update_novelty_stats(rnd_state: RNDState, raw: jnp.ndarray) -> RNDState:
    """Welford-style update of running mean/var from a batch of raw novelties."""
    flat = raw.reshape(-1)
    batch_count = flat.shape[0]
    batch_mean = jnp.mean(flat)
    batch_var = jnp.var(flat)
    count = rnd_state.novelty_count
    new_count = count + batch_count
    delta = batch_mean - rnd_state.novelty_mean
    new_mean = rnd_state.novelty_mean + delta * batch_count / new_count
    m_a = rnd_state.novelty_var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / new_count
    new_var = m2 / new_count
    return rnd_state.replace(
        novelty_mean=new_mean,
        novelty_var=new_var,
        novelty_count=new_count,
    )


def update_predictor(
    module: RNDMLP,
    target_params: Params,
    rnd_state: RNDState,
    optimizer: optax.GradientTransformation,
    obs,
    action: jnp.ndarray,
) -> Tuple[RNDState, jnp.ndarray]:
    """One Adam step on the predictor; refreshes novelty running stats."""

    def loss_fn(predictor_params):
        x = _sa_features(obs, action)
        target = jax.lax.stop_gradient(module.apply(target_params, x))
        pred = module.apply(predictor_params, x)
        return jnp.mean(jnp.square(target - pred))

    loss, grads = jax.value_and_grad(loss_fn)(rnd_state.predictor_params)
    updates, opt_state = optimizer.update(grads, rnd_state.optimizer_state)
    predictor_params = optax.apply_updates(rnd_state.predictor_params, updates)

    x = _sa_features(obs, action)
    target = module.apply(target_params, x)
    pred = module.apply(predictor_params, x)
    raw = jnp.mean(jnp.square(target - pred), axis=-1)
    rnd_state = rnd_state.replace(
        predictor_params=predictor_params,
        optimizer_state=opt_state,
    )
    rnd_state = update_novelty_stats(rnd_state, jax.lax.stop_gradient(raw))
    return rnd_state, loss
