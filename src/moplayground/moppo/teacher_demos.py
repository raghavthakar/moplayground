"""Offline demonstration buffers for behavioral cloning into MORLAX.

Frozen exploration policies (IntrinsicPPO checkpoints) are rolled out on the
training environment; transitions are tagged with a fixed preference label
``w_label`` that registers where on the MORLAX simplex that behavior should
live.  The buffer is consumed during MORLAX training as an auxiliary BC loss.

Checkpoint loading goes through ``brax.training.agents.ppo.checkpoint`` so the
network shape is read from the saved ``ppo_network_config.json`` rather than
guessed here -- there is exactly one source of truth for a checkpoint's shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.types import PRNGKey


@dataclass(frozen=True)
class TeacherSpec:
    """One frozen teacher and the MORLAX preference it anchors."""

    name: str
    checkpoint: str
    preference: Sequence[float]


def load_ppo_teacher_policy(checkpoint_path: str, deterministic: bool = False):
    """Load a brax PPO teacher as ``policy(obs, key) -> (action, extras)``.

    The network shape comes from the checkpoint's saved config. Collection uses
    the stochastic policy (``deterministic=False``) because BC needs the
    pre-tanh ``raw_action`` that only the stochastic branch emits.
    """
    return ppo_checkpoint.load_policy(checkpoint_path, deterministic=deterministic)


def _squeeze_batch(x: np.ndarray) -> np.ndarray:
    return x[0] if x.ndim > 1 and x.shape[0] == 1 else x


def collect_teacher_demos(
    env,
    teacher_policy,
    preference: Sequence[float],
    *,
    num_steps: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Roll a frozen teacher for ``num_steps`` and return demo arrays.

    ``env`` is expected to auto-reset on episode end (BraxAutoResetWrapper), so
    a single continuous unroll spans many episodes with no manual bookkeeping.
    """
    pref = np.asarray(preference, dtype=np.float32)
    obs_states: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []

    key = jax.random.PRNGKey(seed)
    state = env.reset(key)
    for _ in range(num_steps):
        key, act_key = jax.random.split(key)
        action, extras = teacher_policy(state.obs, act_key)
        if 'raw_action' not in extras:
            raise ValueError(
                'Teacher policy did not emit raw_action; load it with '
                'deterministic=False so BC targets are available.'
            )
        obs_states.append(_squeeze_batch(np.asarray(state.obs['state'])))
        raw_actions.append(_squeeze_batch(np.asarray(extras['raw_action'])))
        state = env.step(state, action)

    obs_arr = np.stack(obs_states, axis=0).astype(np.float32)
    return {
        'observation_state': obs_arr,
        'raw_action': np.stack(raw_actions, axis=0).astype(np.float32),
        'directive': np.tile(pref, (obs_arr.shape[0], 1)),
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
    num_steps: int,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Collect and merge stochastic demos from every teacher."""
    buffers = []
    for i, teacher in enumerate(teachers):
        print(
            f'Collecting demos for teacher {teacher.name!r} '
            f'at w={list(teacher.preference)} from {teacher.checkpoint}'
        )
        policy = jax.jit(load_ppo_teacher_policy(teacher.checkpoint))
        buf = collect_teacher_demos(
            env,
            policy,
            teacher.preference,
            num_steps=num_steps,
            seed=seed + i,
        )
        print(f'  -> {buf["observation_state"].shape[0]} transitions')
        buffers.append(buf)
    merged = merge_demo_buffers(buffers)
    print(f'Merged demo buffer: {merged["observation_state"].shape[0]} transitions')
    return merged
