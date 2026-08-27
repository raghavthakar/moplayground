#!/usr/bin/env python3
"""Collect offline BC demos from frozen exploration teachers for MORLAX.

Example:
  python -m scripts.collect_teacher_demos \\
    config/morlax/mohopper_sparse_bc.yaml \\
    --output /nfs/hpc/share/thakarr/SMORL/data/mohopper_sparse_bc_demos.npz
"""

import argparse

import moplayground as mop
import minimal_mjx as mm
from moplayground.moppo.teacher_demos import (
    TeacherSpec,
    collect_all_teachers,
    save_demo_buffer,
)


def _parse_teachers(distill_cfg) -> list[TeacherSpec]:
    return [
        TeacherSpec(
            name=entry.get('name', 'teacher'),
            checkpoint=entry['checkpoint'],
            preference=list(entry['preference']),
        )
        for entry in distill_cfg.get('teachers', [])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Training YAML with distill_params')
    parser.add_argument('--output', type=str, required=True, help='Output .npz path')
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Episodes per teacher (default: distill_params.num_episodes)',
    )
    args = parser.parse_args()

    config = mm.utils.config.create_config_dict(mop.utils.read_config(args.config))
    distill = config.learning_params.morlax_params.get('distill_params')
    if distill is None or not distill.get('enabled', False):
        raise ValueError('distill_params.enabled must be true in the config.')

    teachers = _parse_teachers(distill)
    if not teachers:
        raise ValueError('distill_params.teachers is empty.')

    num_episodes = args.episodes or int(distill.get('num_episodes', 64))
    episode_length = int(config.learning_params.base_ppo_params.episode_length)
    num_steps = num_episodes * episode_length

    mm.utils.setupGPU.run_setup()
    env, _ = mop.envs.create_environment(config, for_training=True)
    env = mop.learning.training.mo_wrapper(
        env,
        episode_length=episode_length,
        action_repeat=int(config.learning_params.base_ppo_params.action_repeat),
    )

    buffer = collect_all_teachers(
        env,
        teachers,
        num_steps=num_steps,
        seed=int(config.learning_params.base_ppo_params.seed),
    )
    save_demo_buffer(args.output, buffer)
    print(f'Saved demo buffer to {args.output}')


if __name__ == '__main__':
    main()
