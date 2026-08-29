"""Aggregate a migration budget sweep on disk into one summary table.

Walks the experiment root (see scripts/migration_sweep.py), reads each run's
archive(s) and ``bc_eval.json``, and writes ``<root>/summary.csv``.

**Metric guide (read this before interpreting numbers):**

* ``wandb_final_hv`` / ``wandb_bc_init_hv`` — from W&B export if provided; these
  reflect the *current* Pareto front at eval time (trust these for policy quality).
* ``archive_*`` — from on-disk ``archive.csv`` (cumulative unlock log; can
  overstate retained performance; use only for coverage diagnostics).
* ``bc_*`` — from ``bc_eval.json``: cold vs post-BC rollout at fixed preferences
  *before any finetune gradient step* (the BC head-start measurement).

Usage:
    python -m scripts.analyze_migration_sweep --root <sweep_root>
    python -m scripts.analyze_migration_sweep --root <sweep_root> --wandb <wandb_export_dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def nondominated_mask(F):
    """Boolean mask of Pareto-nondominated rows (maximization)."""
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if (F[j] >= F[i]).all() and (F[j] > F[i]).any():
                keep[i] = False
                break
    return keep


def hypervolume_2d(F):
    """2D hypervolume dominated over the origin (maximization). 0 if not 2D."""
    if F.shape[1] != 2 or len(F) == 0:
        return 0.0
    P = np.clip(F, 0.0, None)
    P = P[nondominated_mask(P)]
    P = P[np.argsort(P[:, 0])]
    hv, prev_x = 0.0, 0.0
    for x, y in P:
        hv += y * (x - prev_x)
        prev_x = x
    return float(hv)


def archive_stats(csv_path, thresholds, prefix):
    """Coverage stats for one archive.csv, keyed with ``prefix``."""
    out = {f'{prefix}_members': 0, f'{prefix}_nd': 0, f'{prefix}_hv': 0.0}
    if not csv_path.exists():
        return out
    df = pd.read_csv(csv_path)
    if df.empty:
        return out
    obj_cols = list(df.columns[4:])
    F = df[obj_cols].to_numpy(dtype=float)
    thr = np.asarray(thresholds, dtype=float)

    def _fill(tag, sub):
        if len(sub) == 0:
            return
        out[f'{prefix}{tag}_members'] = int(len(sub))
        out[f'{prefix}{tag}_nd'] = int(nondominated_mask(sub).sum())
        out[f'{prefix}{tag}_hv'] = round(hypervolume_2d(sub), 1)
        for k, col in enumerate(obj_cols):
            out[f'{prefix}{tag}_max_{col}'] = round(float(sub[:, k].max()), 1)
        both = (sub >= thr).all(axis=1).sum() if sub.shape[1] == len(thr) else 0
        out[f'{prefix}{tag}_both_unlocked'] = int(both)

    _fill('', F)
    last = df[df['ckpt_step'] == df['ckpt_step'].max()]
    _fill('_final', last[obj_cols].to_numpy(dtype=float))
    return out


def bc_stats(bc_path):
    """Scalars from ``bc_eval.json`` for the BC head-start."""
    out = {
        'bc_skipped': True,
        'bc_cold_hv': np.nan,
        'bc_post_hv': np.nan,
        'bc_hv_gain': np.nan,
        'bc_cold_jump_max': np.nan,
        'bc_post_jump_max': np.nan,
        'bc_jump_gain': np.nan,
        'bc_post_unlock_both': np.nan,
        'bc_num_teachers': 0,
    }
    if not bc_path.exists():
        return out
    data = json.loads(bc_path.read_text())
    if data.get('skipped'):
        return out
    cold = data.get('cold', {})
    post = data.get('post_bc', {})
    gain = data.get('gain', {})
    out.update({
        'bc_skipped': False,
        'bc_cold_hv': cold.get('hypervolume', np.nan),
        'bc_post_hv': post.get('hypervolume', np.nan),
        'bc_hv_gain': gain.get('hypervolume', np.nan),
        'bc_cold_jump_max': cold.get('jump_max', np.nan),
        'bc_post_jump_max': post.get('jump_max', np.nan),
        'bc_jump_gain': gain.get('jump_max', np.nan),
        'bc_post_unlock_both': post.get('unlock_both', np.nan),
        'bc_num_teachers': data.get('num_teachers', 0),
    })
    return out


def load_wandb_metrics(wandb_dir):
    """Last-step finetune/baseline HV and finetune step-0 BC init from W&B export."""
    wandb_dir = Path(wandb_dir)
    hist_path = wandb_dir / 'history.csv'
    if not hist_path.exists():
        return {}
    hist = pd.read_csv(hist_path, low_memory=False)
    out = {}
    for name, g in hist.groupby('name'):
        g = g.sort_values('_step')
        base = name.replace('-finetune', '').replace('-explore', '').replace('-bc', '')
        if name.endswith('-finetune'):
            if 'eval/hypervolume' in g.columns:
                s = g['eval/hypervolume'].dropna()
                if len(s):
                    out.setdefault(base, {})['wandb_final_hv'] = float(s.iloc[-1])
            if 'migration/bc_init_hypervolume' in g.columns:
                s = g['migration/bc_init_hypervolume'].dropna()
                if len(s):
                    out.setdefault(base, {})['wandb_bc_init_hv'] = float(s.iloc[0])
            for col, key in [
                ('eval/return/Jump_Height/max', 'wandb_final_jump_max'),
                ('eval/return/Forward_Distance/max', 'wandb_final_fwd_max'),
            ]:
                if col in g.columns:
                    s = g[col].dropna()
                    if len(s):
                        out.setdefault(base, {})[key] = float(s.iloc[-1])
        elif not name.endswith('-explore') and not name.endswith('-bc'):
            if 'eval/hypervolume' in g.columns:
                s = g['eval/hypervolume'].dropna()
                if len(s):
                    out.setdefault(base, {})['wandb_final_hv'] = float(s.iloc[-1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Experiment root (migration_sweep --save-dir).')
    parser.add_argument('--wandb', default=None, help='Optional W&B export dir (history.csv).')
    parser.add_argument('--thresholds', default='50,50', help='CSV unlock thresholds (obj order).')
    parser.add_argument('--out', default=None, help='Summary CSV path (default <root>/summary.csv).')
    args = parser.parse_args()

    root = Path(args.root)
    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip() != '']
    out_path = Path(args.out) if args.out else root / 'summary.csv'
    wandb_by_name = load_wandb_metrics(args.wandb) if args.wandb else {}

    metas = sorted(root.glob('*/run_meta.json'))
    if not metas:
        raise SystemExit(f'No run_meta.json found under {root}. Did the sweep run?')

    rows = []
    for meta_path in metas:
        meta = json.loads(meta_path.read_text())
        run_dir = meta_path.parent
        row = {
            'variant': meta.get('variant'),
            'kind': meta.get('kind'),
            'seed': meta.get('seed'),
            'explore_m': meta.get('explore_m'),
            'finetune_m': meta.get('finetune_m'),
            'name': meta.get('name'),
        }
        row.update(bc_stats(run_dir / 'bc_eval.json'))
        row.update(wandb_by_name.get(meta.get('name', ''), {}))
        if meta.get('kind') == 'baseline':
            row.update(archive_stats(run_dir / 'archive.csv', thresholds, 'archive'))
        else:
            row.update(archive_stats(run_dir / 'explore' / 'archive.csv', thresholds, 'explore_archive'))
            row.update(archive_stats(run_dir / 'finetune' / 'archive.csv', thresholds, 'archive'))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['variant', 'seed']).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    view_cols = [c for c in [
        'variant', 'seed',
        'bc_cold_hv', 'bc_post_hv', 'bc_hv_gain', 'bc_post_jump_max',
        'wandb_bc_init_hv', 'wandb_final_hv', 'wandb_final_jump_max',
        'archive_final_nd', 'archive_final_both_unlocked',
    ] if c in df.columns]
    print(f'\nWrote {out_path}  ({len(df)} runs)\n')
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(df[view_cols].to_string(index=False))

    if 'wandb_final_hv' in df.columns:
        agg = df.groupby('variant')['wandb_final_hv'].agg(['mean', 'std', 'count'])
        print('\nW&B final hypervolume by variant (mean over seeds):')
        print(agg.to_string())
    if 'bc_hv_gain' in df.columns:
        mig = df[df.kind == 'migration']
        if len(mig):
            print('\nBC hypervolume gain (post_bc - cold, migration runs only):')
            print(mig.groupby('variant')['bc_hv_gain'].agg(['mean', 'std']).to_string())


if __name__ == '__main__':
    main()
