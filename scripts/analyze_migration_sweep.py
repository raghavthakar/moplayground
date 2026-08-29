"""Aggregate a migration budget sweep on disk into one summary table.

Walks the experiment root (see scripts/migration_sweep.py), reads each run's
archive(s), and writes <root>/summary.csv with one row per run: final-eval and
whole-archive coverage (nondominated count, 2D hypervolume from the origin,
per-objective maxima, #both-unlocked). Pure numpy/pandas -- no GPU, no repo
imports -- so it runs anywhere the result files are.

Usage:
    python -m scripts.analyze_migration_sweep --root <sweep_root>
    python -m scripts.analyze_migration_sweep --root <sweep_root> --thresholds 50,50
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
    P = P[np.argsort(P[:, 0])]  # x ascending (=> y descending for a ND front)
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
    obj_cols = list(df.columns[4:])  # member,step,ckpt_step,nondominated,<objs...>
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

    _fill('', F)  # whole-archive coverage
    last = df[df['ckpt_step'] == df['ckpt_step'].max()]
    _fill('_final', last[obj_cols].to_numpy(dtype=float))  # final eval only
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Experiment root (migration_sweep --save-dir).')
    parser.add_argument('--thresholds', default='50,50', help='CSV unlock thresholds (obj order).')
    parser.add_argument('--out', default=None, help='Summary CSV path (default <root>/summary.csv).')
    args = parser.parse_args()

    root = Path(args.root)
    thresholds = [float(x) for x in args.thresholds.split(',') if x.strip() != '']
    out_path = Path(args.out) if args.out else root / 'summary.csv'

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
        if meta.get('kind') == 'baseline':
            row.update(archive_stats(run_dir / 'archive.csv', thresholds, 'result'))
        else:
            row.update(archive_stats(run_dir / 'explore' / 'archive.csv', thresholds, 'explore'))
            row.update(archive_stats(run_dir / 'finetune' / 'archive.csv', thresholds, 'result'))
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(['variant', 'seed']).reset_index(drop=True)
    df.to_csv(out_path, index=False)

    # Compact console view: the headline comparison metrics.
    view_cols = [c for c in [
        'variant', 'seed', 'result_final_nd', 'result_final_hv',
        'result_final_both_unlocked', 'result_nd', 'result_hv',
    ] if c in df.columns]
    print(f'\nWrote {out_path}  ({len(df)} runs)\n')
    with pd.option_context('display.max_rows', None, 'display.width', 160):
        print(df[view_cols].to_string(index=False))

    if 'result_final_hv' in df.columns:
        agg = df.groupby('variant')['result_final_hv'].agg(['mean', 'std', 'count'])
        print('\nFinal-eval hypervolume by variant (mean over seeds):')
        print(agg.to_string())


if __name__ == '__main__':
    main()
