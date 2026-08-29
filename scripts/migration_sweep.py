"""Budget-split migration experiment: plain MORLAX vs explore->BC->finetune.

Tests the migration concept end-to-end on sparse MOHopper (50x50), multi-seed.
Every run consumes the SAME 50M sample budget; the only differences are:

  * baseline-50m : plain cold-start MORLAX for the full 50M.
  * e{E}-f{F}    : E million IntrinsicPPO exploration, BC of the Pareto archive
                   into the MORLAX hypernetwork, then F million MORLAX finetuning
                   (E + F = 50).

All variants derive from ONE config (config/morlax/mohopper_sparse_migration.yaml)
so the baseline and every finetune phase share identical MORLAX settings -- the
sole difference is cold vs BC-seeded init. This keeps the comparison honest.

Organisation
------------
On disk (under --save-dir, one root for the whole experiment):
    <root>/baseline-50m-seed0/                 (plain MORLAX; archive.csv, progress.svg)
    <root>/e10-f40-seed0/explore/              (IntrinsicPPO archive + ckpts)
    <root>/e10-f40-seed0/finetune/             (MORLAX archive + fronts)
    <root>/<run>/run_meta.json                 (variant, seed, budget, wandb ids)
Aggregate with scripts/analyze_migration_sweep.py -> <root>/summary.csv.

W&B: all runs share one --group; job_type is baseline/explore/bc/finetune and
each run's config carries variant/seed/explore_m/finetune_m for UI grouping.

Usage
-----
    # print the run matrix and exit (no conda / PYTHONPATH needed)
    python -m scripts.migration_sweep --list

    # training runs need the SMORL conda env + src on PYTHONPATH:
    export PYTHONPATH=/nfs/hpc/share/thakarr/SMORL/moplayground/src
    conda activate /nfs/hpc/share/thakarr/SMORL
    python -m scripts.migration_sweep --index "${SLURM_ARRAY_TASK_ID}"

    # everything sequentially (local)
    python -m scripts.migration_sweep
"""

import argparse
import copy
import json
import time
import traceback
from pathlib import Path

ENTITY = 'raghavthakar-oregon-state-university'
PROJECT = 'SMORL'

# Explore/finetune budget splits in millions (must each sum to TOTAL_M).
TOTAL_M = 50
SPLITS_M = [(5, 45), (10, 40), (15, 35), (20, 30), (25, 25)]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_GROUP = 'mohopper-thr50-budget-sweep'
DEFAULT_SAVE_DIR = '/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep'
DEFAULT_BASE = 'config/morlax/mohopper_sparse_migration.yaml'


def build_matrix(seeds):
    """Deterministic (variant, seed) cells; index maps 1:1 to a Slurm array id."""
    variants = [('baseline-50m', 'baseline', 0, TOTAL_M)]
    variants += [(f'e{e}-f{f}', 'migration', e, f) for (e, f) in SPLITS_M]
    cells = []
    for name, kind, e, f in variants:
        for seed in seeds:
            cells.append({
                'variant': name, 'kind': kind, 'seed': int(seed),
                'explore_m': int(e), 'finetune_m': int(f),
            })
    return cells


def _import_training_deps():
    """Lazy import so ``--list`` works without conda / PYTHONPATH."""
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


def run_cell(wandb, mop, mm, base_config, cell, group, save_dir):
    cfg = copy.deepcopy(base_config)
    cfg.save_dir = save_dir
    cfg.name = f"{cell['variant']}-seed{cell['seed']}"
    cfg.learning_params.base_ppo_params.seed = cell['seed']
    tags = [group, cell['variant'], f"seed{cell['seed']}", cell['kind']]

    env, _ = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    run_names = []

    if cell['kind'] == 'baseline':
        cfg.algorithm = 'morlax'
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default=DEFAULT_BASE, help='Migration base YAML config.')
    parser.add_argument('--save-dir', default=DEFAULT_SAVE_DIR, help='Experiment root on disk.')
    parser.add_argument('--group', default=DEFAULT_GROUP, help='W&B group for the whole experiment.')
    parser.add_argument('--seeds', default=','.join(map(str, DEFAULT_SEEDS)), help='CSV of seeds.')
    parser.add_argument('--index', type=int, default=None, help='Run only this cell (Slurm array id).')
    parser.add_argument('--list', action='store_true', help='Print the run matrix and exit.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip a cell whose run dir exists.')
    args = parser.parse_args()

    for e, f in SPLITS_M:
        if e + f != TOTAL_M:
            raise SystemExit(f'Split ({e},{f}) does not sum to {TOTAL_M}M.')

    seeds = [int(x) for x in args.seeds.split(',') if x.strip() != '']
    cells = build_matrix(seeds)

    if args.list:
        print(f'Experiment: {args.group}  ({len(cells)} runs; Slurm --array=0-{len(cells) - 1})')
        for i, c in enumerate(cells):
            print(f"  [{i:2d}] {c['variant']:<12} seed={c['seed']}  "
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
        if args.skip_existing and run_dir.is_dir() and any(run_dir.iterdir()):
            print(f'[skip] {name} (exists at {run_dir})')
            results.append((name, 'skipped'))
            continue
        print(f'\n===== Running {name} ({cell["kind"]}, '
              f'explore={cell["explore_m"]}M finetune={cell["finetune_m"]}M) =====')
        try:
            run_cell(wandb, mop, mm, base_config, cell, args.group, args.save_dir)
            results.append((name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {name}: {e}')
            traceback.print_exc()
            results.append((name, f'fail: {e}'))

    print('\n===== Migration sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
