"""Random Network Distillation (RND) for intrinsic novelty rewards.

Fixed random target network and trainable predictor. Novelty is prediction
error on normalized root/CoM state features (excluding actions and high-DoF
joint configurations), normalized by a running standard deviation for PPO.
"""

from typing import Mapping, Optional, Sequence, Tuple

from brax.training.types import Params
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

# MOHumanoid flat observation layout (58 dims total):
#   [0:21]  joint_angles (21)
#   [21:22] head_height (1)
#   [22:25] extremities (3)
#   [25:28] torso_vertical_orientation (3)
#   [28:31] com_vel (3)
#   [31:37] root free-joint qvel (6)
#   [37:58] actuator joint qvel (21)
_HUMANOID_OBS_DIM = 58


def flatten_obs(obs) -> jnp.ndarray:
    """Extract a flat observation vector (handles dict or array obs)."""
    if isinstance(obs, Mapping):
        return obs['state']
    return obs


def _root_com_slice(state: jnp.ndarray) -> jnp.ndarray:
    """Restrict state vector to root/CoM posture and velocity features."""
    if state.shape[-1] == _HUMANOID_OBS_DIM:
        return jnp.concatenate(
            [state[..., 21:22], state[..., 25:37]],
            axis=-1,
        )
    return state


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


def _sa_features(
    obs,
    action: Optional[jnp.ndarray] = None,
    normalizer_params=None,
    max_abs_obs: float = 5.0,
) -> jnp.ndarray:
    del action  # Actions are excluded so RND only scores state novelty.
    state = flatten_obs(obs)
    if normalizer_params is not None:
        mean = flatten_obs(normalizer_params.mean)
        std = flatten_obs(normalizer_params.std)
        state = (state - mean) / std
        if max_abs_obs is not None:
            state = jnp.clip(state, -max_abs_obs, max_abs_obs)
    return _root_com_slice(state)


def init_rnd(
    key: jax.Array,
    obs_dim: int,
    action_size: int = 0,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    output_size: int = 64,
    learning_rate: float = 1e-4,
) -> Tuple[RNDMLP, Params, RNDState, optax.GradientTransformation]:
    """Create target (fixed) + predictor (trainable) and optimizer."""
    del action_size
    key_target, key_pred = jax.random.split(key)
    module = RNDMLP(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        output_size=output_size,
    )
    dummy = _sa_features(jnp.zeros((1, obs_dim)))
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


def novelty_reward(
    module: RNDMLP,
    target_params: Params,
    rnd_state: RNDState,
    obs,
    action: jnp.ndarray,
    eps: float = 1e-8,
    scale: float = 1.0,
    normalizer_params=None,
) -> jnp.ndarray:
    """Normalized novelty used as the intrinsic PPO reward (no grad).

    ``scale`` multiplies the std-normalized prediction error (the usual RND
    coefficient). Default 1.0 leaves the running-std unit scale unchanged.
    When ``normalizer_params`` is provided, observations are standardized and
    clipped to ``[-5, 5]`` before feeding the RND networks (Burda et al., 2019).
    """
    x = _sa_features(obs, action, normalizer_params=normalizer_params)
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
    normalizer_params=None,
) -> Tuple[RNDState, jnp.ndarray]:
    """One Adam step on the predictor; refreshes novelty running stats."""
    x = _sa_features(obs, action, normalizer_params=normalizer_params)
    target = jax.lax.stop_gradient(module.apply(target_params, x))

    def loss_fn(predictor_params):
        pred = module.apply(predictor_params, x)
        return jnp.mean(jnp.square(target - pred))

    loss, grads = jax.value_and_grad(loss_fn)(rnd_state.predictor_params)
    updates, opt_state = optimizer.update(grads, rnd_state.optimizer_state)
    predictor_params = optax.apply_updates(rnd_state.predictor_params, updates)

    pred = module.apply(predictor_params, x)
    raw = jnp.mean(jnp.square(target - pred), axis=-1)
    rnd_state = rnd_state.replace(
        predictor_params=predictor_params,
        optimizer_state=opt_state,
    )
    rnd_state = update_novelty_stats(rnd_state, jax.lax.stop_gradient(raw))
    return rnd_state, loss
