"""Sweep driver for the naive episodic-return sparsity study (MOHopper).

Sweeps over a list of *literal* per-objective episodic-return thresholds. No
algorithmic changes: only the environment is sparsified (via
EpisodicThresholdWrapper, toggled from config). The point is to find the
threshold level at which vanilla MORLAX breaks.

Each objective's per-step reward is withheld (0) until that objective's running
episodic return reaches its threshold, then the true per-step reward flows.
An all-zero threshold vector disables gating entirely (dense baseline).

Thresholds are given as a ';'-separated list of runs, each run a ','-separated
vector ordered to match `reward.optimization.objectives` (i.e. [run, jump]):

    --thresholds "0,0;100,100;250,250;500,500"

The list is enumerated deterministically, so a Slurm array maps
`$SLURM_ARRAY_TASK_ID` -> run via `--index`.

Usage:
    # whole sweep sequentially:
    python -m scripts.sparse_threshold_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --thresholds "0,0;100,100;250,250;500,500;1000,1000"

    # single run (Slurm array task):
    python -m scripts.sparse_threshold_sweep \
        --base config/morlax/mohopper_sparse.yaml \
        --thresholds "0,0;100,100;250,250;500,500;1000,1000" \
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


def parse_threshold_sets(s):
    """Parse "0,0;250,250" -> [[0.0, 0.0], [250.0, 250.0]]."""
    sets = []
    for chunk in s.split(';'):
        chunk = chunk.strip()
        if not chunk:
            continue
        vec = [float(x.strip()) for x in chunk.split(',') if x.strip() != '']
        sets.append(vec)
    return sets


def apply_overrides(base_config, thresholds):
    cfg = copy.deepcopy(base_config)
    thr_cfg = cfg.env_config.reward.episodic_threshold
    if all(t <= 0.0 for t in thresholds):
        thr_cfg.enabled = False
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = list(thresholds)

    base_name = cfg.name
    cfg.name = f"{base_name}-thr={'x'.join(fmt(t) for t in thresholds)}"
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
        tags    = ['low-end sparsity sweep'],
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
    parser.add_argument('--thresholds', type=str,
                        default='0,0;25,25;50,50;75,75;100,100;125,125;150,150;175,175;200,200',
                        help="';'-separated runs; each run a ','-separated "
                             'per-objective threshold vector (order matches '
                             'objectives). All-zero => dense baseline.')
    parser.add_argument('--index', type=int, default=None,
                        help='If set, run only the threshold set at this index (Slurm array).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip run if save_dir/name already exists.')
    args = parser.parse_args()

    base_config    = mop.utils.read_config(args.base)
    threshold_sets = parse_threshold_sets(args.thresholds)

    num_objs = len(base_config.env_config.reward.optimization.objectives)
    for vec in threshold_sets:
        if len(vec) != num_objs:
            raise SystemExit(
                f'Threshold vector {vec} has {len(vec)} entries but the config '
                f'has {num_objs} objectives.'
            )

    print(f'Sweep: {len(threshold_sets)} threshold sets')
    for i, vec in enumerate(threshold_sets):
        label = 'disabled (dense)' if all(t <= 0.0 for t in vec) else vec
        print(f'  [{i}] thresholds={label}')

    if args.index is not None:
        if not (0 <= args.index < len(threshold_sets)):
            raise SystemExit(
                f'--index {args.index} out of range for {len(threshold_sets)} sets.'
            )
        threshold_sets = [threshold_sets[args.index]]
        print(f'Running only index {args.index}: thresholds={threshold_sets[0]}')

    results = []
    for thresholds in threshold_sets:
        cfg = apply_overrides(base_config, thresholds)
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
