"""Offline demonstration buffers for behavioral cloning into MORLAX.

Frozen exploration policies (IntrinsicPPO checkpoints) are rolled out on the
training environment; transitions are tagged with a fixed preference label
``w_label`` that registers where on the MORLAX simplex that behavior should
live.  The buffer seeds BC-based transfer into the MORLAX hypernetwork.

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
    demo_noise_scale: float = 0.1,
    max_abs_obs: float = 1e4,
) -> dict[str, np.ndarray]:
    """Roll a frozen teacher for ``num_steps`` and return actor demo arrays.

    ``env`` is a single-env ``mo_wrapper`` stack (VmapWrapper batches one env,
    so reset takes a batched key of shape ``(1, 2)``). It auto-resets on episode
    end, so one continuous unroll spans many episodes with no manual bookkeeping.

    States are labeled with the teacher's full action distribution parameters
    ``distribution_params = (loc, scale_raw)`` (and noiseless pre-tanh mean
    ``loc``) while the env is stepped with damped exploration noise
    (``demo_noise_scale``) so trajectories stay in the threshold-unlocking
    basin yet cover a tube of states around the gait. Any transition with a
    diverged/non-finite observation or pre-NaN MJX velocity spike
    (``|obs| > max_abs_obs``) is dropped so it cannot corrupt the BC
    observation normalizer.
    """
    pref = np.asarray(preference, dtype=np.float32)
    obs_states: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    dist_params_list: list[np.ndarray] = []

    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    state = env.reset(jax.random.split(reset_key, 1))
    for _ in range(num_steps):
        key, act_key, noise_key = jax.random.split(key, 3)
        action, extras = teacher_policy(state.obs, act_key)
        dist_p = None
        if 'distribution_params' in extras:
            dist_p = extras['distribution_params']
            loc, scale_raw = jnp.split(dist_p, 2, axis=-1)
            scale = jax.nn.softplus(scale_raw) + 1e-3
            u_exec = loc + float(demo_noise_scale) * scale * jax.random.normal(
                noise_key, shape=loc.shape
            )
            action = jnp.tanh(u_exec)
            target_raw = loc
        elif 'raw_action' in extras:
            target_raw = extras['raw_action']
        else:
            raise ValueError(
                'Teacher policy did not emit distribution_params or raw_action; '
                'load it with deterministic=False so BC targets are available.'
            )
        obs_np = _squeeze_batch(np.asarray(state.obs['state'], dtype=np.float32))
        raw_np = _squeeze_batch(np.asarray(target_raw, dtype=np.float32))
        state = step_fn(state, action)
        if (
            np.all(np.isfinite(obs_np))
            and np.all(np.isfinite(raw_np))
            and np.max(np.abs(obs_np)) <= float(max_abs_obs)
        ):
            obs_states.append(obs_np)
            raw_actions.append(raw_np)
            if dist_p is not None:
                dist_params_list.append(
                    _squeeze_batch(np.asarray(dist_p, dtype=np.float32))
                )

    obs_arr = np.stack(obs_states, axis=0).astype(np.float32)
    out = {
        'observation_state': obs_arr,
        'raw_action': np.stack(raw_actions, axis=0).astype(np.float32),
        'directive': np.tile(pref, (obs_arr.shape[0], 1)),
    }
    if len(dist_params_list) == len(obs_states):
        out['distribution_params'] = np.stack(
            dist_params_list, axis=0
        ).astype(np.float32)
    return out


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
    out = {
        'observation': {'state': jnp.asarray(buffer['observation_state'])},
        'raw_action': jnp.asarray(buffer['raw_action']),
        'directive': jnp.asarray(buffer['directive']),
    }
    if 'distribution_params' in buffer:
        out['distribution_params'] = jnp.asarray(buffer['distribution_params'])
    return out


def sample_bc_batch(
    key: PRNGKey,
    buffer: dict[str, jnp.ndarray],
    batch_size: int,
) -> dict[str, jnp.ndarray]:
    """Uniformly sample a BC minibatch from a loaded demo buffer."""
    n = buffer['raw_action'].shape[0]
    idx = jax.random.randint(key, (batch_size,), 0, n)
    batch = {
        'observation': {'state': buffer['observation']['state'][idx]},
        'raw_action': buffer['raw_action'][idx],
        'directive': buffer['directive'][idx],
    }
    if 'distribution_params' in buffer:
        batch['distribution_params'] = buffer['distribution_params'][idx]
    return batch


def collect_all_teachers(
    env,
    teachers: Sequence[TeacherSpec],
    *,
    num_steps: int,
    seed: int = 0,
    demo_noise_scale: float = 0.1,
) -> dict[str, np.ndarray]:
    """Collect and merge demos from every teacher."""
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
            demo_noise_scale=demo_noise_scale,
        )
        print(f'  -> {buf["observation_state"].shape[0]} transitions')
        buffers.append(buf)
    merged = merge_demo_buffers(buffers)
    print(f'Merged demo buffer: {merged["observation_state"].shape[0]} transitions')
    return merged
