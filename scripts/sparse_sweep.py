"""Sweep driver for the sparse-run MOCheetah sanity study.

Loads a base YAML config (expects the sparse-run schema: a `run_milestone`
objective plus `env_config.reward.run_milestone.step_size` and
`env_config.reward.weights.run_milestone`), then iterates over the cartesian
product of {milestone_weight, step_size}, overriding those two reward fields in
a deep-copied config and running `train_policy` in-process for each combo.

The grid is enumerated deterministically as
`itertools.product(milestone_weights, step_sizes)`, so a Slurm array can map
`$SLURM_ARRAY_TASK_ID` -> combo via `--index`.

Usage:
    # run the whole grid sequentially (one process / local):
    python -m scripts.sparse_sweep \
        --base config/morlax/mocheetah_sparse.yaml \
        --milestone-weights 20,100,500 \
        --step-sizes 1.0,3.0,6.0

    # run a single combo (for a Slurm array task):
    python -m scripts.sparse_sweep \
        --base config/morlax/mocheetah_sparse.yaml \
        --milestone-weights 20,100,500 \
        --step-sizes 1.0,3.0,6.0 \
        --index "${SLURM_ARRAY_TASK_ID}"
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import itertools
import os
import traceback

import wandb

import moplayground as mop
import minimal_mjx as mm


def parse_csv(s, cast=str):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def fmt(x):
    """Compact float formatting for run names (20.0 -> 20, 3.0 -> 3)."""
    return f"{x:g}"


def build_combos(milestone_weights, step_sizes):
    return list(itertools.product(milestone_weights, step_sizes))


def apply_overrides(base_config, milestone_weight, step_size):
    cfg = copy.deepcopy(base_config)
    reward = cfg.env_config.reward
    reward.weights.run_milestone = milestone_weight
    reward.run_milestone.step_size = step_size

    base_name = cfg.name
    cfg.name = f"{base_name}-w={fmt(milestone_weight)}-ss={fmt(step_size)}"
    return cfg


def run_one(cfg):
    env, _      = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    name = cfg.save_dir + '/' + cfg.name
    run = mm.utils.logging.initialize_wandb(
        name    = name.replace('/', ''),
        entity  = 'raghavthakar-oregon-state-university',
        project = 'SMORL',
        config  = dict(cfg),
    )
    try:
        mop.learning.train_policy(cfg, env, eval_env, run, warn_github_changes=False)
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True,
                        help='Path to base sparse-run YAML config.')
    parser.add_argument('--milestone-weights', type=str, default='20,100,500',
                        help='CSV of weights[run_milestone] values.')
    parser.add_argument('--step-sizes', type=str, default='1.0,3.0,6.0',
                        help='CSV of reward.run_milestone.step_size values (meters).')
    parser.add_argument('--index', type=int, default=None,
                        help='If set, run only the combo at this index in the '
                             'product(milestone_weights, step_sizes) grid '
                             '(for Slurm array tasks).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combo if save_dir/name already exists.')
    args = parser.parse_args()

    base_config       = mop.utils.read_config(args.base)
    milestone_weights = parse_csv(args.milestone_weights, cast=float)
    step_sizes        = parse_csv(args.step_sizes, cast=float)

    combos = build_combos(milestone_weights, step_sizes)
    print(f'Sweep: {len(combos)} combos')
    for i, (w, ss) in enumerate(combos):
        print(f'  [{i}] milestone_weight={w:g} step_size={ss:g}')

    if args.index is not None:
        if not (0 <= args.index < len(combos)):
            raise SystemExit(
                f'--index {args.index} out of range for {len(combos)} combos.'
            )
        combos = [combos[args.index]]
        print(f'Running only combo index {args.index}: '
              f'weight={combos[0][0]:g} step_size={combos[0][1]:g}')

    results = []
    for milestone_weight, step_size in combos:
        cfg = apply_overrides(base_config, milestone_weight, step_size)
        run_path = os.path.join(cfg.save_dir, cfg.name)
        if args.skip_existing and os.path.isdir(run_path) and os.listdir(run_path):
            print(f'[skip] {cfg.name} (exists at {run_path})')
            results.append((cfg.name, 'skipped'))
            continue

        print(f'\n===== Running {cfg.name} =====')
        try:
            run_one(cfg)
            results.append((cfg.name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {cfg.name}: {e}')
            traceback.print_exc()
            results.append((cfg.name, f'fail: {e}'))

    print('\n===== Sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
