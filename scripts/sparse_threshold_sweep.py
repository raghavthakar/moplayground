"""Sweep driver for the naive episodic-return sparsity study (MOHopper).

Sweeps a single *sparsity fraction* and sets each objective's episodic-return
threshold to `fraction * per-objective max return`. No algorithmic changes:
only the environment is sparsified (via EpisodicThresholdWrapper, toggled from
config). The point is to find the sparsity level at which vanilla MORLAX breaks.

fraction = 0.0  -> thresholds disabled entirely (dense baseline).
fraction > 0.0  -> thresholds = fraction * obj_maxes, enabled.

The grid is enumerated deterministically over `fractions`, so a Slurm array
maps `$SLURM_ARRAY_TASK_ID` -> combo via `--index`.

Usage:
    # whole sweep sequentially:
    python -m scripts.sparse_threshold_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --fractions 0.0,0.25,0.5,0.7,0.85,0.95 \
        --obj-maxes 2500,2200

    # single combo (Slurm array task):
    python -m scripts.sparse_threshold_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --fractions 0.0,0.25,0.5,0.7,0.85,0.95 \
        --obj-maxes 2500,2200 \
        --index "${SLURM_ARRAY_TASK_ID}"
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import os
import traceback

import wandb

import moplayground as mop
import minimal_mjx as mm


def parse_csv(s, cast=str):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def fmt(x):
    return f"{x:g}"


def apply_overrides(base_config, fraction, obj_maxes):
    cfg = copy.deepcopy(base_config)
    thr_cfg = cfg.env_config.reward.episodic_threshold
    if fraction <= 0.0:
        thr_cfg.enabled = False
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = [fraction * m for m in obj_maxes]

    base_name = cfg.name
    cfg.name = f"{base_name}-sparsity={fmt(fraction)}"
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
                        help='Path to base sparse-hopper YAML config.')
    parser.add_argument('--fractions', type=str, default='0.0,0.25,0.5,0.7,0.85,0.95',
                        help='CSV of sparsity fractions (of per-objective max return).')
    parser.add_argument('--obj-maxes', type=str, default='2500,2200',
                        help='CSV of per-objective max returns (order matches objectives).')
    parser.add_argument('--index', type=int, default=None,
                        help='If set, run only the fraction at this index (Slurm array).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combo if save_dir/name already exists.')
    args = parser.parse_args()

    base_config = mop.utils.read_config(args.base)
    fractions   = parse_csv(args.fractions, cast=float)
    obj_maxes   = parse_csv(args.obj_maxes, cast=float)

    print(f'Sweep: {len(fractions)} sparsity levels (obj_maxes={obj_maxes})')
    for i, f in enumerate(fractions):
        thr = 'disabled (dense)' if f <= 0.0 else [f * m for m in obj_maxes]
        print(f'  [{i}] sparsity={f:g} thresholds={thr}')

    if args.index is not None:
        if not (0 <= args.index < len(fractions)):
            raise SystemExit(
                f'--index {args.index} out of range for {len(fractions)} levels.'
            )
        fractions = [fractions[args.index]]
        print(f'Running only index {args.index}: sparsity={fractions[0]:g}')

    results = []
    for fraction in fractions:
        cfg = apply_overrides(base_config, fraction, obj_maxes)
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
