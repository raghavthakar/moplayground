"""IntrinsicPPO + RND scale × seed sweep at a fixed unlock threshold.

Sweeps a log-spaced RND novelty coefficient (``rnd_params.scale``) across
seeds. Training is scalar PPO on scaled RND only; eval reports ungated
extrinsic returns and unlock rates vs the threshold.

All runs land in one W&B group so the 5×5 grid stays together in the project.

The (scale, seed) grid is enumerated deterministically (scale-major: all
seeds of scale 0, then scale 1, ...), so a Slurm array maps
``$SLURM_ARRAY_TASK_ID`` -> one cell via ``--index``.

Usage:
    # whole grid sequentially:
    python -m scripts.intrinsic_scale_sweep \
        --base config/intrinsic/mohopper_rnd.yaml \
        --threshold "50,50" \
        --scales "0.01,0.1,1,10,100" \
        --seeds "0,1,2,3,4"

    # single cell (Slurm array task):
    python -m scripts.intrinsic_scale_sweep \
        --base config/intrinsic/mohopper_rnd.yaml \
        --threshold "50,50" \
        --scales "0.01,0.1,1,10,100" \
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


def parse_floats(s):
    return [float(x.strip()) for x in s.split(',') if x.strip() != '']


def parse_ints(s):
    return [int(x.strip()) for x in s.split(',') if x.strip() != '']


def threshold_tag(thresholds):
    return 'x'.join(fmt(t) for t in thresholds)


def apply_overrides(base_config, thresholds, scale, seed):
    cfg = copy.deepcopy(base_config)

    thr_cfg = cfg.env_config.reward.episodic_threshold
    if all(t <= 0.0 for t in thresholds):
        thr_cfg.enabled = False
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = list(thresholds)

    cfg.learning_params.base_ppo_params.seed = seed
    cfg.learning_params.intrinsic_ppo_params.rnd_params.scale = scale

    base_name = cfg.name
    cfg.name = (
        f"{base_name}-thr={threshold_tag(thresholds)}"
        f"-iscale={fmt(scale)}-seed={seed}"
    )
    return cfg


def run_one(cfg, group):
    env, _      = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    name = cfg.save_dir + '/' + cfg.name
    scale = cfg.learning_params.intrinsic_ppo_params.rnd_params.scale
    run = mm.utils.logging.initialize_wandb(
        name     = name.replace('/', ''),
        entity   = 'raghavthakar-oregon-state-university',
        project  = 'SMORL',
        config   = dict(cfg),
        group    = group,
        job_type = f'iscale={fmt(scale)}',
        tags     = ['intrinsic-ppo', 'rnd-scale-sweep', group],
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
                        help='Path to base intrinsic-PPO YAML config.')
    parser.add_argument('--threshold', type=str, default='50,50',
                        help="','-separated per-objective unlock threshold "
                             '(order matches objectives).')
    parser.add_argument('--scales', type=str, default='0.01,0.1,1,10,100',
                        help='CSV of RND novelty scales (rnd_params.scale).')
    parser.add_argument('--seeds', type=str, default='0,1,2,3,4',
                        help='CSV of integer seeds.')
    parser.add_argument('--group', type=str, default=None,
                        help='W&B group name. Defaults to '
                             '"intrinsic-rnd-thr=<run>x<jump>".')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override config.save_dir for this sweep.')
    parser.add_argument('--index', type=int, default=None,
                        help='If set, run only the (scale, seed) cell at this '
                             'index (scale-major; Slurm array).')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip run if save_dir/name already exists.')
    args = parser.parse_args()

    base_config = mop.utils.read_config(args.base)
    if args.save_dir:
        base_config.save_dir = args.save_dir

    thresholds = parse_floats(args.threshold)
    scales     = parse_floats(args.scales)
    seeds      = parse_ints(args.seeds)

    num_objs = len(base_config.env_config.reward.optimization.objectives)
    if len(thresholds) != num_objs:
        raise SystemExit(
            f'Threshold vector {thresholds} has {len(thresholds)} entries but the '
            f'config has {num_objs} objectives.'
        )

    grid = [(scale, seed) for scale in scales for seed in seeds]
    group = args.group or f"intrinsic-rnd-thr={threshold_tag(thresholds)}"

    print(
        f'Intrinsic scale sweep: thresholds={thresholds} group="{group}" '
        f'scales={scales} seeds={seeds} ({len(grid)} cells)'
    )
    for i, (scale, seed) in enumerate(grid):
        print(f'  [{i}] scale={fmt(scale)} seed={seed}')

    if args.index is not None:
        if not (0 <= args.index < len(grid)):
            raise SystemExit(
                f'--index {args.index} out of range for {len(grid)} cells.'
            )
        grid = [grid[args.index]]
        scale, seed = grid[0]
        print(f'Running only index {args.index}: scale={fmt(scale)} seed={seed}')

    results = []
    for scale, seed in grid:
        cfg = apply_overrides(base_config, thresholds, scale, seed)
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

    print('\n===== Intrinsic scale sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
