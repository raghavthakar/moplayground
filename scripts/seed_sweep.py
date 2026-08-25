"""Seed sweep for a single sparsity threshold (MOHopper), grouped in W&B.

Runs the same fixed episodic-return threshold across multiple seeds and puts
every run in one W&B *group* (which renders as a collapsible sub-grouping in
the project and lets you aggregate mean/std across seeds) — so seed replicates
don't clutter the flat run list.

The seed list is enumerated deterministically, so a Slurm array maps
`$SLURM_ARRAY_TASK_ID` -> seed via `--index`.

Usage:
    # all seeds sequentially:
    python -m scripts.seed_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --threshold "50,50" \
        --seeds "0,1,2,3,4"

    # single seed (Slurm array task):
    python -m scripts.seed_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --threshold "50,50" \
        --seeds "0,1,2,3,4" \
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


def fmt(x):
    return f"{x:g}"


def parse_threshold(s):
    return [float(x.strip()) for x in s.split(',') if x.strip() != '']


def threshold_tag(thresholds):
    return 'x'.join(fmt(t) for t in thresholds)


def apply_overrides(base_config, thresholds, seed):
    cfg = copy.deepcopy(base_config)

    thr_cfg = cfg.env_config.reward.episodic_threshold
    if all(t <= 0.0 for t in thresholds):
        thr_cfg.enabled = False
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = list(thresholds)

    cfg.learning_params.base_ppo_params.seed = seed

    base_name = cfg.name
    cfg.name = f"{base_name}-thr={threshold_tag(thresholds)}-seed={seed}"
    return cfg


def run_one(cfg, group):
    env, _      = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    name = cfg.save_dir + '/' + cfg.name
    run = mm.utils.logging.initialize_wandb(
        name     = name.replace('/', ''),
        entity   = 'raghavthakar-oregon-state-university',
        project  = 'SMORL',
        config   = dict(cfg),
        group    = group,
        job_type = 'seed',
        tags     = ['low-end sparsity sweep', 'seed-sweep', group],
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
    parser.add_argument('--threshold', type=str, default='50,50',
                        help="Single ','-separated per-objective threshold vector "
                             '(order matches objectives). All-zero => dense baseline.')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4',
                        help='CSV of integer seeds to run.')
    parser.add_argument('--group', type=str, default=None,
                        help='W&B group name. Defaults to "sparse-thr=<run>x<jump>".')
    parser.add_argument('--index', type=int, default=None,
                        help='If set, run only the seed at this index (Slurm array).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip run if save_dir/name already exists.')
    args = parser.parse_args()

    base_config = mop.utils.read_config(args.base)
    thresholds  = parse_threshold(args.threshold)
    seeds       = [int(x.strip()) for x in args.seeds.split(',') if x.strip() != '']

    num_objs = len(base_config.env_config.reward.optimization.objectives)
    if len(thresholds) != num_objs:
        raise SystemExit(
            f'Threshold vector {thresholds} has {len(thresholds)} entries but the '
            f'config has {num_objs} objectives.'
        )

    group = args.group or f"sparse-thr={threshold_tag(thresholds)}"

    print(f'Seed sweep: thresholds={thresholds} group="{group}" seeds={seeds}')

    if args.index is not None:
        if not (0 <= args.index < len(seeds)):
            raise SystemExit(
                f'--index {args.index} out of range for {len(seeds)} seeds.'
            )
        seeds = [seeds[args.index]]
        print(f'Running only index {args.index}: seed={seeds[0]}')

    results = []
    for seed in seeds:
        cfg = apply_overrides(base_config, thresholds, seed)
        run_path = os.path.join(cfg.save_dir, cfg.name)
        if args.skip_existing and os.path.isdir(run_path) and os.listdir(run_path):
            print(f'[skip] {cfg.name} (exists at {run_path})')
            results.append((cfg.name, 'skipped'))
            continue

        print(f'\n===== Running {cfg.name} =====')
        try:
            run_one(cfg, group)
            results.append((cfg.name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {cfg.name}: {e}')
            traceback.print_exc()
            results.append((cfg.name, f'fail: {e}'))

    print('\n===== Seed sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
