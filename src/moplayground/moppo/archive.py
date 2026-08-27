"""Extrinsic archive: retain eval policies that cross a return threshold.

When an evaluation episode's return vector meets the threshold on *any*
objective, that return (and a copy of the policy checkpoint from that eval)
is migrated into the archive. The archive is never ranked by intrinsic
reward; it is the retained coverage set for later Pareto plots.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from moplayground.utils.pareto import compute_pareto_statistics, get_nondominated


class ExtrinsicArchive:
    """Growing set of unlocking eval return vectors (+ optional checkpoints)."""

    def __init__(
        self,
        thresholds: Sequence[float],
        output_dir: Path,
        labels: Optional[Sequence[str]] = None,
    ):
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.output_dir = Path(output_dir)
        self.ckpt_root = self.output_dir / 'archive' / 'ckpts'
        self.labels = [str(x) for x in (labels or [])]
        self.returns: list[np.ndarray] = []
        self.steps: list[int] = []
        self.ckpt_steps: list[int] = []
        self._seen: set[tuple] = set()

    @property
    def size(self) -> int:
        return len(self.returns)

    def snapshot(self) -> np.ndarray:
        if not self.returns:
            return np.zeros((0, int(self.thresholds.shape[0])), dtype=float)
        return np.stack(self.returns, axis=0)

    def ingest(self, step: int, rewards) -> int:
        """Add unique eval returns that unlock at least one objective.

        ``rewards`` is ``(n_eval, n_obj)``. Identical rows (deterministic eval)
        collapse to one member via rounding.
        """
        if hasattr(rewards, 'block_until_ready'):
            rewards = rewards.block_until_ready()
        rewards = np.asarray(rewards, dtype=float)
        if rewards.ndim == 1:
            rewards = rewards[None, :]
        if rewards.size == 0:
            return 0
        added = 0
        step = int(step)
        for r in rewards:
            if not np.any(r >= self.thresholds):
                continue
            key = tuple(np.round(r.astype(float), 3).tolist())
            if key in self._seen:
                continue
            self._seen.add(key)
            self.returns.append(np.asarray(r, dtype=float).copy())
            self.steps.append(step)
            self.ckpt_steps.append(step)
            added += 1
        return added

    def attach_checkpoint(self, step: int, src: Path) -> None:
        """Copy the eval's policy checkpoint if this step contributed members."""
        src = Path(src)
        if not src.is_dir():
            return
        if int(step) not in self.ckpt_steps:
            return
        dst = self.ckpt_root / f'{int(step):012d}'
        if dst.exists():
            return
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    def annotate(self, metrics: dict) -> dict:
        """Attach archive arrays + scalars for plotting / W&B."""
        F = self.snapshot()
        metrics['archive_reward'] = F
        metrics['archive/size'] = int(self.size)
        if F.shape[0] == 0:
            metrics['archive/num_nondominated'] = 0
            metrics['archive/hypervolume'] = 0.0
            metrics['archive/sparsity'] = float('nan')
            return metrics
        nd = get_nondominated(F)
        metrics['archive/num_nondominated'] = int(len(nd))
        try:
            stats = compute_pareto_statistics(F)
            metrics['archive/hypervolume'] = float(stats.hypervolume)
            metrics['archive/sparsity'] = float(stats.sparsity)
        except Exception as e:
            print(f'Warning: archive Pareto statistics failed: {e}')
            metrics['archive/hypervolume'] = 0.0
            metrics['archive/sparsity'] = float('nan')
        return metrics

    def save_csv(self, path: Optional[Path] = None) -> None:
        path = Path(path) if path is not None else self.output_dir / 'archive.csv'
        n_obj = int(self.thresholds.shape[0])
        names = list(self.labels) if len(self.labels) >= n_obj else [
            f'obj_{i}' for i in range(n_obj)
        ]
        rows = []
        F = self.snapshot()
        nd = set(int(i) for i in get_nondominated(F)) if F.shape[0] else set()
        for i, r in enumerate(self.returns):
            row = {
                'member': i,
                'step': self.steps[i],
                'ckpt_step': self.ckpt_steps[i],
                'nondominated': int(i in nd),
            }
            for j, name in enumerate(names[:n_obj]):
                row[name] = float(r[j])
            rows.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
