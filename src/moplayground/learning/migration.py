"""End-to-end explore -> behavioral-clone -> finetune migration, one run.

A single sample budget is split between two phases of one MOHopper run:

  1. Exploration: IntrinsicPPO maximizes RND novelty (reward is replaced by
     novelty) and records an unlock archive of extrinsic (run, jump) return
     vectors with their checkpoints.
  2. Migration: a subset of the exploration Pareto archive is distilled into a
     cold MORLAX hypernetwork by behavioral cloning. Each teacher's preference
     label is read from its POSITION on the Pareto front (its objective vector
     normalized to the simplex) -- no hand assignment.
  3. Finetuning: standard MORLAX trains for the rest of the budget, its policy
     initialized from the BC hypernetwork. MORLAX itself is unchanged; the only
     migration hook is ``morlax.train(init_hypernetwork_params=...)``.

Both phases reuse ``train_policy`` verbatim (phase 2 via a ``handle_params``
override that injects the BC init), so nothing about either trainer is forked.
"""

from __future__ import annotations

import copy
import functools
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from brax.training.acme import running_statistics
from minimal_mjx.utils.config import create_config_dict

from moplayground.moppo import bc, factory, teacher_demos
from moplayground.learning.training import mo_wrapper, setup_morlax, train_policy


def pref_from_objectives(objectives) -> np.ndarray:
    """Preference = the policy's position on the Pareto front (simplex-normalized).

    Negative objective returns are clipped to 0 before normalizing; a degenerate
    all-zero vector falls back to a uniform preference.
    """
    o = np.clip(np.asarray(objectives, dtype=float), 0.0, None)
    total = o.sum()
    if total <= 0.0:
        return np.ones_like(o) / len(o)
    return o / total


def select_archive_teachers(explore_dir, labels, selection='nondominated', max_teachers=0):
    """Read the exploration archive and pick teachers to migrate.

    Returns a list of ``(ckpt_step, ckpt_path, objective_vector, preference)``.
    Members are grouped by checkpoint (one policy per eval step); a policy's
    objective vector is the mean of its archived return rows.
    """
    explore_dir = Path(explore_dir)
    csv_path = explore_dir / 'archive.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f'No archive at {csv_path}: exploration produced no threshold unlocks. '
            'Lower the unlock thresholds or extend explore_steps.'
        )
    df = pd.read_csv(csv_path)
    if selection == 'nondominated':
        subset = df[df['nondominated'] == 1]
        if len(subset) == 0:
            subset = df
    elif selection == 'all':
        subset = df
    else:
        raise ValueError(f"Unknown selection '{selection}'. Use 'nondominated' or 'all'.")

    labels = list(labels)
    teachers = []
    for ckpt_step, group in subset.groupby('ckpt_step'):
        ckpt = explore_dir / 'archive' / 'ckpts' / f'{int(ckpt_step):012d}'
        if not ckpt.is_dir():
            continue
        objectives = group[labels].mean().to_numpy(dtype=float)
        teachers.append(
            (int(ckpt_step), str(ckpt), objectives, pref_from_objectives(objectives))
        )
    if not teachers:
        raise RuntimeError(f'No archived checkpoints found under {explore_dir}/archive/ckpts.')

    teachers.sort(key=lambda t: t[3][0])  # ascending by first-objective share
    if max_teachers and len(teachers) > max_teachers:
        idx = np.linspace(0, len(teachers) - 1, max_teachers).round().astype(int)
        teachers = [teachers[i] for i in sorted(set(idx.tolist()))]
    return teachers


def build_bc_init(config, env, explore_dir):
    """Distill selected exploration teachers into a MORLAX hypernetwork (BC).

    Returns the BC-trained hypernetwork params, ready to seed ``morlax.train``.
    """
    mp = config.migration_params
    labels = list(config.env_config.reward.optimization.labels)
    seed = int(config.learning_params.base_ppo_params.seed)

    teachers_info = select_archive_teachers(
        explore_dir,
        labels,
        selection=mp.get('selection', 'nondominated'),
        max_teachers=int(mp.get('max_teachers', 0)),
    )
    print(f'Migrating {len(teachers_info)} teacher(s) from {explore_dir}:')
    for step, _ckpt, objectives, pref in teachers_info:
        objs = ', '.join(f'{l}={v:.1f}' for l, v in zip(labels, objectives))
        w = [round(float(x), 3) for x in pref]
        print(f'  step {step}: {objs} -> w={w}')

    teacher_specs = [
        teacher_demos.TeacherSpec(
            name=f'm@{step}', checkpoint=ckpt, preference=pref.tolist()
        )
        for (step, ckpt, _objectives, pref) in teachers_info
    ]

    episode_length = int(config.learning_params.base_ppo_params.episode_length)
    action_repeat = int(config.learning_params.base_ppo_params.action_repeat)
    demo_env = mo_wrapper(env, episode_length=episode_length, action_repeat=action_repeat)
    num_episodes = int(mp.demos.num_episodes)

    buffer = teacher_demos.demo_buffer_to_jax(
        teacher_demos.collect_all_teachers(
            demo_env,
            teacher_specs,
            num_steps=num_episodes * episode_length,
            seed=seed,
        )
    )

    obs_dim = int(buffer['observation']['state'].shape[-1])
    num_objectives = len(labels)
    network_params = dict(config.learning_params.morlax_params.network_params)
    networks = factory.make_morlax_networks(
        observation_size={'state': (obs_dim,)},
        action_size=env.action_size,
        num_objectives=num_objectives,
        key=jax.random.PRNGKey(seed),
        preprocess_observations_fn=running_statistics.normalize,
        **network_params,
    )
    normalizer = bc.build_normalizer(buffer)
    print(f'Behavioral cloning {int(buffer["raw_action"].shape[0])} transitions '
          f'into the MORLAX hypernetwork...')
    return bc.pretrain_hypernetwork(
        networks,
        normalizer,
        buffer,
        steps=int(mp.bc.steps),
        batch_size=int(mp.bc.batch_size),
        lr=float(mp.bc.lr),
        seed=seed,
    )


def train_migration(config, env, eval_env, run=None):
    """Run explore -> BC -> finetune as a single budget-split pipeline."""
    config = create_config_dict(config)
    mp = config.migration_params
    base_name = config.name

    # --- Phase 1/2: exploration (IntrinsicPPO), builds the unlock archive. ---
    explore_cfg = copy.deepcopy(config)
    explore_cfg.algorithm = 'intrinsic_ppo'
    explore_cfg.name = f'{base_name}/explore'
    explore_cfg.learning_params.base_ppo_params.num_timesteps = int(mp.explore_steps)
    print(f'=== Migration phase 1/2: exploration for {int(mp.explore_steps)} steps ===')
    train_policy(explore_cfg, env, eval_env, run=run)
    explore_dir = Path(config.save_dir) / f'{base_name}/explore'

    # --- Migration: BC-distill archive teachers into the MORLAX policy init. ---
    print('=== Migration: behavioral cloning archive teachers into MORLAX init ===')
    init_hypernetwork_params = build_bc_init(config, env, explore_dir)

    # --- Phase 2/2: MORLAX finetuning from the BC init (MORLAX unchanged). ---
    finetune_cfg = copy.deepcopy(config)
    finetune_cfg.algorithm = 'morlax'
    finetune_cfg.name = f'{base_name}/finetune'
    finetune_cfg.learning_params.base_ppo_params.num_timesteps = int(mp.finetune_steps)

    def handle_params(cfg):
        train_fn, network_factory = setup_morlax(cfg)
        train_fn = functools.partial(
            train_fn, init_hypernetwork_params=init_hypernetwork_params
        )
        return train_fn, network_factory

    print(f'=== Migration phase 2/2: MORLAX finetuning for {int(mp.finetune_steps)} steps ===')
    return train_policy(
        finetune_cfg, env, eval_env, run=run, handle_params=handle_params
    )
