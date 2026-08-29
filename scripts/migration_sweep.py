"""Budget-split migration experiment: plain MORLAX vs explore->BC->finetune.

Every run consumes the same total sample budget (``--total-m``). Variants:

  * baseline-{N}m : plain cold-start MORLAX for the full budget.
  * e{E}-f{F}     : E M IntrinsicPPO exploration, BC of the Pareto archive into
                    the MORLAX hypernetwork, then F M MORLAX finetuning (E+F=N).

BC head-start is logged to the ``bc`` W&B run (``bc/cold/eval/*``, ``bc/eval/*``)
and on disk as ``<run>/bc_eval.json``. Finetune logs ``migration/bc_init_*`` at
step 0 before any gradient update.

Usage
-----
    python -m scripts.migration_sweep --list
    python -m scripts.migration_sweep --total-m 100 --splits "20,80;25,75;30,70" \\
        --seeds 0,1,2,3,4,5,6,7,8,9 --index "${SLURM_ARRAY_TASK_ID}"
"""

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

ENTITY = 'raghavthakar-oregon-state-university'
PROJECT = 'SMORL'

# Defaults match the original 50M sweep (override via CLI for new experiments).
DEFAULT_TOTAL_M = 50
DEFAULT_SPLITS_M = [(5, 45), (10, 40), (15, 35), (20, 30), (25, 25)]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_GROUP = 'mohopper-thr50-budget-sweep'
DEFAULT_SAVE_DIR = '/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep'
DEFAULT_BASE = 'config/morlax/mohopper_sparse_migration.yaml'


def parse_splits(splits_str, total_m):
    """Parse ``"20,80;25,75"`` into ``[(20, 80), (25, 75), ...]``."""
    pairs = []
    for part in splits_str.split(';'):
        part = part.strip()
        if not part:
            continue
        e, f = (int(x.strip()) for x in part.split(','))
        if e + f != total_m:
            raise ValueError(f'Split ({e},{f}) does not sum to {total_m}M.')
        pairs.append((e, f))
    if not pairs:
        raise ValueError('No splits parsed.')
    return pairs


def parse_threshold(s):
    return [float(x.strip()) for x in s.split(',') if x.strip() != '']


def apply_threshold(cfg, thresholds):
    thr_cfg = cfg.env_config.reward.episodic_threshold
    if all(t <= 0.0 for t in thresholds):
        thr_cfg.enabled = False
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = list(thresholds)


def build_matrix(seeds, total_m, splits_m):
    """Deterministic (variant, seed) cells; index maps 1:1 to a Slurm array id."""
    baseline = f'baseline-{total_m}m'
    variants = [(baseline, 'baseline', 0, total_m)]
    variants += [(f'e{e}-f{f}', 'migration', e, f) for (e, f) in splits_m]
    cells = []
    for name, kind, e, f in variants:
        for seed in seeds:
            cells.append({
                'variant': name, 'kind': kind, 'seed': int(seed),
                'explore_m': int(e), 'finetune_m': int(f), 'total_m': int(total_m),
            })
    return cells


def _import_training_deps():
    import matplotlib
    matplotlib.use('Agg')
    import wandb
    import moplayground as mop
    import minimal_mjx as mm
    return wandb, mop, mm


def _init_run(mm, cfg, name, group, job_type, tags, extra):
    wandb_config = cfg.to_dict()
    wandb_config.update(extra)
    return mm.utils.logging.initialize_wandb(
        name=name.replace('/', ''),
        entity=ENTITY,
        project=PROJECT,
        config=wandb_config,
        group=group,
        job_type=job_type,
        tags=tags,
        reinit=True,
    )


def _write_meta(root, cfg, cell, group, run_names):
    meta = dict(cell)
    meta.update({
        'name': cfg.name,
        'group': group,
        'save_dir': str(root),
        'run_dir': str(Path(root) / cfg.name),
        'wandb_runs': run_names,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    })
    run_dir = Path(root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2) + '\n')


def run_cell(wandb, mop, mm, base_config, cell, group, save_dir, thresholds=None):
    cfg = copy.deepcopy(base_config)
    if thresholds is not None:
        apply_threshold(cfg, thresholds)
    cfg.save_dir = save_dir
    cfg.name = f"{cell['variant']}-seed{cell['seed']}"
    cfg.learning_params.base_ppo_params.seed = cell['seed']
    tags = [group, cell['variant'], f"seed{cell['seed']}", cell['kind']]

    env, _ = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    run_names = []

    if cell['kind'] == 'baseline':
        cfg.algorithm = 'morlax'
        cfg.learning_params.base_ppo_params.num_timesteps = cell['total_m'] * 1_000_000
        run = _init_run(mm, cfg, cfg.name, group, 'baseline', tags, cell)
        run_names.append(run.name)
        try:
            mop.learning.train_policy(cfg, env, eval_env, run)
        finally:
            try:
                wandb.finish()
            except Exception:
                pass
    else:
        cfg.migration_params.explore_steps = cell['explore_m'] * 1_000_000
        cfg.migration_params.finetune_steps = cell['finetune_m'] * 1_000_000

        def run_factory(phase, base_name):
            run = _init_run(
                mm, cfg, f'{base_name}-{phase}', group, phase,
                tags + [phase], {**cell, 'phase': phase},
            )
            run_names.append(run.name)
            return run

        mop.train_migration(cfg, env, eval_env, run_factory=run_factory)

    _write_meta(save_dir, cfg, cell, group, run_names)


def default_splits_for(total_m):
    if total_m == DEFAULT_TOTAL_M:
        return list(DEFAULT_SPLITS_M)
    return [(e, total_m - e) for e in range(20, total_m // 2 + 1, 5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=DEFAULT_BASE, help='Migration base YAML config.')
    parser.add_argument('--save-dir', default=DEFAULT_SAVE_DIR, help='Experiment root on disk.')
    parser.add_argument('--group', default=DEFAULT_GROUP, help='W&B group for the whole experiment.')
    parser.add_argument('--total-m', type=int, default=DEFAULT_TOTAL_M, help='Total budget in millions.')
    parser.add_argument('--splits', default=None,
                        help='Semicolon-separated explore,finetune pairs in M, e.g. "20,80;25,75".')
    parser.add_argument('--threshold', type=str, default=None,
                        help="Episodic unlock thresholds, e.g. '50,50' or '100,100'.")
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)), help='CSV of seeds.')
    parser.add_argument('--index', type=int, default=None, help='Run only this cell (Slurm array id).')
    parser.add_argument('--list', action='store_true', help='Print the run matrix and exit.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip only cells that already completed (have run_meta.json). '
                             'Partial dirs from failed cells are re-run.')
    args = parser.parse_args()

    splits_m = parse_splits(args.splits, args.total_m) if args.splits else default_splits_for(args.total_m)
    thresholds = parse_threshold(args.threshold) if args.threshold else None
    seeds = [int(x) for x in args.seeds.split(',') if x.strip() != '']
    cells = build_matrix(seeds, args.total_m, splits_m)
    if thresholds is not None:
        for c in cells:
            c['thresholds'] = thresholds

    if args.list:
        print(f'Experiment: {args.group}  budget={args.total_m}M  ({len(cells)} runs; '
              f'Slurm --array=0-{len(cells) - 1})')
        for i, c in enumerate(cells):
            print(f"  [{i:2d}] {c['variant']:<14} seed={c['seed']}  "
                  f"explore={c['explore_m']}M finetune={c['finetune_m']}M ({c['kind']})")
        return

    wandb, mop, mm = _import_training_deps()
    base_config = mop.utils.read_config(args.base)

    selected = cells if args.index is None else [cells[args.index]]
    if args.index is not None and not (0 <= args.index < len(cells)):
        raise SystemExit(f'--index {args.index} out of range for {len(cells)} cells.')

    results = []
    for cell in selected:
        name = f"{cell['variant']}-seed{cell['seed']}"
        run_dir = Path(args.save_dir) / name
        # A cell is "done" only once run_meta.json is written (at successful
        # completion). Partial explore/ dirs from a crashed cell do NOT count,
        # so a rerun re-does failed cells instead of silently skipping them.
        if args.skip_existing and (run_dir / 'run_meta.json').is_file():
            print(f'[skip] {name} (complete at {run_dir})')
            results.append((name, 'skipped'))
            continue
        print(f'\n===== Running {name} ({cell["kind"]}, '
              f'explore={cell["explore_m"]}M finetune={cell["finetune_m"]}M) =====')
        try:
            run_cell(wandb, mop, mm, base_config, cell, args.group, args.save_dir, thresholds)
            results.append((name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {name}: {e}')
            traceback.print_exc()
            results.append((name, f'fail: {e}'))

    print('\n===== Migration sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')

    # Exit non-zero if any cell failed so Slurm marks the task FAILED (not
    # COMPLETED). Exceptions were caught above for the summary; surface them here.
    if any(str(status).startswith('fail') for _, status in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
