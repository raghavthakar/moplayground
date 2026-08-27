"""Export one W&B group to a local folder (shareable analysis packet).

Dumps run metadata + scalar history. Skips images, HTML plots, and the
eval/performances table (those are huge and not needed for the verdict).

Usage (machine that is `wandb login`'d):

    python -m scripts.export_wandb_group \
        --group intrinsic-rnd-thr=50x50 \
        --out results/wandb_export/intrinsic-rnd-thr=50x50

Then zip that folder and drop it in the repo, or point me at the path.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import wandb


ENTITY = 'raghavthakar-oregon-state-university'
PROJECT = 'SMORL'

# Scalars that decide the explorer verdict. Others are kept if present.
KEEP_PREFIXES = (
    'eval/return/',
    'eval/unlock/',
    'eval/hypervolume',
    'eval/sparsity',
    'eval/avg_episode_length',
    'training/intrinsic_reward_mean',
    'training/rnd_loss',
)


def _keep_column(name: str) -> bool:
    if name in ('_step', 'step'):
        return True
    return any(name == p or name.startswith(p) for p in KEEP_PREFIXES)


def _cfg_get(cfg, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', default=ENTITY)
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--group', required=True)
    parser.add_argument('--out', required=True, help='Output directory for the packet.')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    path = f'{args.entity}/{args.project}'
    runs = api.runs(path, filters={'group': args.group})
    runs = list(runs)
    if not runs:
        raise SystemExit(f'No runs found in {path} group="{args.group}".')

    meta_rows = []
    histories = []
    for run in runs:
        cfg = dict(run.config) if run.config else {}
        scale = _cfg_get(
            cfg, 'learning_params', 'intrinsic_ppo_params', 'rnd_params', 'scale'
        )
        seed = _cfg_get(cfg, 'learning_params', 'base_ppo_params', 'seed')
        meta_rows.append({
            'run_id': run.id,
            'name': run.name,
            'state': run.state,
            'group': run.group,
            'job_type': run.job_type,
            'seed': seed,
            'iscale': scale,
            'url': run.url,
        })

        hist = run.history(pandas=True, samples=10_000)
        if hist is None or hist.empty:
            print(f'[skip history] {run.name} ({run.id}) empty')
            continue
        keep = [c for c in hist.columns if _keep_column(str(c))]
        hist = hist[keep].copy()
        hist.insert(0, 'run_id', run.id)
        hist.insert(1, 'name', run.name)
        hist.insert(2, 'seed', seed)
        hist.insert(3, 'iscale', scale)
        histories.append(hist)
        print(f'[ok] {run.name}  state={run.state}  rows={len(hist)}')

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out / 'runs.csv', index=False)

    if histories:
        history = pd.concat(histories, ignore_index=True)
        history.to_csv(out / 'history.csv', index=False)
    else:
        history = pd.DataFrame()

    summary = {
        'entity': args.entity,
        'project': args.project,
        'group': args.group,
        'n_runs': len(meta_rows),
        'n_history_rows': int(len(history)),
        'states': meta['state'].value_counts().to_dict(),
        'scales': sorted({r['iscale'] for r in meta_rows if r['iscale'] is not None}),
        'seeds': sorted({r['seed'] for r in meta_rows if r['seed'] is not None}),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'\nWrote {out}/runs.csv, history.csv, summary.json')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
