"""Offline demonstration buffers for behavioral cloning into MORLAX.

Frozen exploration policies (e.g. IntrinsicPPO checkpoints) are rolled out on
the training environment; transitions are tagged with a fixed preference label
``w_label`` that registers where on the MORLAX simplex that behavior should
live.  The buffer is consumed during MORLAX training as an auxiliary BC loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import PRNGKey


@dataclass(frozen=True)
class TeacherSpec:
    """One frozen teacher and the MORLAX preference it anchors."""

    name: str
    checkpoint: str
    preference: Sequence[float]


def load_ppo_teacher_policy(
    checkpoint_path: str,
    policy_hidden_layer_sizes: Sequence[int] = (64, 64),
    value_hidden_layer_sizes: Sequence[int] = (256, 256),
    deterministic: bool = True,
):
    """Load a standard brax PPO teacher and return ``policy(obs, key)``."""

    def network_factory(**kwargs):
        return ppo_networks.make_ppo_networks(
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            **kwargs,
        )

    return ppo_checkpoint.load_policy(
        checkpoint_path,
        network_factory=network_factory,
        deterministic=deterministic,
    )


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1 and x.shape[0] == 1:
        return x[0]
    return x


def collect_teacher_demos(
    env,
    teacher_policy,
    preference: Sequence[float],
    *,
    num_episodes: int,
    episode_length: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Roll a frozen teacher and return numpy arrays for the demo buffer."""
    pref = np.asarray(preference, dtype=np.float32)
    obs_states: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    directives: list[np.ndarray] = []

    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    # episode_length is enforced by the training wrapper; unroll until auto-reset.
    for ep in range(num_episodes):
        for _ in range(episode_length):
            key, act_key = jax.random.split(key)
            action, extras = teacher_policy(state.obs, act_key)
            obs_state = _squeeze_batch(np.asarray(state.obs['state']))
            obs_states.append(obs_state)
            raw_actions.append(_squeeze_batch(np.asarray(extras['raw_action'])))
            directives.append(pref.copy())
            state = env.step(state, action)
            if bool(jnp.all(state.done)):
                break

    return {
        'observation_state': np.stack(obs_states, axis=0).astype(np.float32),
        'raw_action': np.stack(raw_actions, axis=0).astype(np.float32),
        'directive': np.stack(directives, axis=0).astype(np.float32),
    }


def merge_demo_buffers(buffers: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not buffers:
        raise ValueError('No demo buffers to merge.')
    return {
        key: np.concatenate([buf[key] for buf in buffers], axis=0)
        for key in buffers[0]
    }


def save_demo_buffer(path: str, buffer: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **buffer)


def load_demo_buffer(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def demo_buffer_to_jax(buffer: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
    return {
        'observation': {'state': jnp.asarray(buffer['observation_state'])},
        'raw_action': jnp.asarray(buffer['raw_action']),
        'directive': jnp.asarray(buffer['directive']),
    }


def sample_bc_batch(
    key: PRNGKey,
    buffer: dict[str, jnp.ndarray],
    batch_size: int,
) -> dict[str, jnp.ndarray]:
    """Uniformly sample a BC minibatch from a loaded demo buffer."""
    n = buffer['raw_action'].shape[0]
    idx = jax.random.randint(key, (batch_size,), 0, n)
    return {
        'observation': {'state': buffer['observation']['state'][idx]},
        'raw_action': buffer['raw_action'][idx],
        'directive': buffer['directive'][idx],
    }


def collect_all_teachers(
    env,
    teachers: Sequence[TeacherSpec],
    *,
    num_episodes: int,
    episode_length: int,
    policy_hidden_layer_sizes: Sequence[int],
    value_hidden_layer_sizes: Sequence[int],
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Collect and merge demos from every teacher in ``teachers``."""
    buffers = []
    for i, teacher in enumerate(teachers):
        print(
            f'Collecting demos for teacher {teacher.name!r} '
            f'at w={list(teacher.preference)} from {teacher.checkpoint}'
        )
        policy = load_ppo_teacher_policy(
            teacher.checkpoint,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            value_hidden_layer_sizes=value_hidden_layer_sizes,
            deterministic=True,
        )
        policy = jax.jit(policy)
        buf = collect_teacher_demos(
            env,
            policy,
            teacher.preference,
            num_episodes=num_episodes,
            episode_length=episode_length,
            seed=seed + i,
        )
        print(f'  -> {buf["observation_state"].shape[0]} transitions')
        buffers.append(buf)
    merged = merge_demo_buffers(buffers)
    print(f'Merged demo buffer: {merged["observation_state"].shape[0]} transitions')
    return merged
