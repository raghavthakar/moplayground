"""Pareto-front statistics for multi-objective evaluation.

Hypervolume convention (matches the MO-Playground paper analysis pipeline)
---------------------------------------------------------------------------
Returns are treated as a **maximization** problem (higher is better).  We
convert to pymoo's minimization convention by negation and compute HV
against an explicit reference point.

Default reference point in *maximization* space:
    ``ref_point_max = 0`` for every objective.

Equivalently, in *minimization* space (what pymoo sees):
    ``ref_point_min = -ref_point_max``  (also the origin under the default).

Interpretation of the default: HV measures the volume dominated by the
non-dominated front relative to the origin.  Only points that strictly
improve on the origin in maximization space (positive returns that are
non-dominated) contribute.  Negative-return points do not dominate the
origin after negation and therefore do not add HV.

Sparsity / spacing
------------------
Computed with pymoo's ``SpacingIndicator`` on the non-dominated front
after per-front min-max normalization to ``[0, 1]``.  Because normalization
is relative to the *current* front's extent, sparsity is scale-invariant
within an eval but should not be compared across fronts with very
different absolute ranges without that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from pymoo.indicators.hv import HV
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.normalization import normalize
import numpy as np


# Default HV reference in maximization space (higher-is-better returns).
# Documented + logged wherever HV is reported.
DEFAULT_HV_REF_POINT_MAX = 0.0


def get_nondominated(F, epsilon=None):
    """Indices of the non-dominated front for a maximization problem."""
    nds = NonDominatedSorting(epsilon=epsilon)
    front_indices = nds.do(-F, only_non_dominated_front=True)
    return front_indices


def _as_ref_point_max(
    n_objs: int,
    ref_point_max: Optional[Union[float, Sequence[float], np.ndarray]] = None,
) -> np.ndarray:
    """Broadcast a maximization-space reference point to shape ``(n_objs,)``."""
    if ref_point_max is None:
        return np.full(n_objs, DEFAULT_HV_REF_POINT_MAX, dtype=float)
    ref = np.asarray(ref_point_max, dtype=float)
    if ref.ndim == 0:
        return np.full(n_objs, float(ref), dtype=float)
    if ref.shape != (n_objs,):
        raise ValueError(
            f'ref_point_max must be scalar or shape ({n_objs},), got {ref.shape}'
        )
    return ref


def hypervolume_from_nondominated(
    F_min: np.ndarray,
    ref_point_min: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    """Compute hypervolume of an already non-dominated front in min-space.

    Args:
        F_min: ``(n_points, n_objectives)`` points in minimization space
            (i.e. negated maximization returns).
        ref_point_min: Reference point in minimization space.  Defaults to
            the origin (matches ``ref_point_max = 0``).

    Returns:
        ``(hypervolume, ref_point_min)``.
    """
    if F_min.size == 0:
        n_objs = 0 if F_min.ndim < 2 else F_min.shape[1]
        ref = (
            np.zeros(n_objs, dtype=float)
            if ref_point_min is None
            else np.asarray(ref_point_min, dtype=float)
        )
        return float('nan'), ref

    if ref_point_min is None:
        ref_point_min = np.zeros(F_min.shape[1], dtype=float)
    else:
        ref_point_min = np.asarray(ref_point_min, dtype=float)

    hv = HV(ref_point=ref_point_min)
    hypervolume = hv(F_min)
    # pymoo returns None when no point dominates the reference.
    if hypervolume is None:
        return float('nan'), ref_point_min
    return float(hypervolume), ref_point_min


def sparsity_from_normalized_nondominated(F_min_norm: np.ndarray) -> float:
    """Spacing indicator on a normalized minimization-space front."""
    spacing = SpacingIndicator()
    sparsity = spacing(F_min_norm)
    if sparsity is None:
        return float('nan')
    return float(sparsity)


@dataclass(frozen=True)
class ParetoStatistics:
    """Full Pareto statistics with the conventions needed to interpret them."""

    hypervolume: float
    sparsity: float
    num_points: int
    num_nondominated: int
    # Reference point used for HV, expressed in *maximization* space
    # (same units as the raw returns).  Default is the origin.
    ref_point_max: np.ndarray
    # Same reference in the minimization space passed to pymoo.
    ref_point_min: np.ndarray
    # Ideal / nadir of the non-dominated front in maximization space
    # (useful context; sparsity normalizes using the front's own range).
    ideal_max: np.ndarray
    nadir_max: np.ndarray
    sense: str = 'maximize'
    sparsity_normalized: bool = True
    notes: str = (
        'HV: pymoo HV on negated non-dominated returns vs ref_point_min=-ref_point_max. '
        'Sparsity: pymoo SpacingIndicator on per-front min-max normalized ND front.'
    )

    def as_dict(self) -> dict:
        return {
            'hypervolume': self.hypervolume,
            'sparsity': self.sparsity,
            'num_points': self.num_points,
            'num_nondominated': self.num_nondominated,
            'ref_point_max': self.ref_point_max.tolist(),
            'ref_point_min': self.ref_point_min.tolist(),
            'ideal_max': self.ideal_max.tolist(),
            'nadir_max': self.nadir_max.tolist(),
            'sense': self.sense,
            'sparsity_normalized': self.sparsity_normalized,
            'notes': self.notes,
        }


def compute_pareto_statistics(
    F: np.ndarray,
    ref_point_max: Optional[Union[float, Sequence[float], np.ndarray]] = None,
) -> ParetoStatistics:
    """Compute hypervolume and sparsity with an explicit HV reference point.

    Args:
        F: ``(n_points, n_objectives)`` objective vectors, **higher is better**.
        ref_point_max: HV reference in maximization space.  Scalar broadcasts
            to all objectives.  Default ``0`` (paper / MO-Playground convention).

    Returns:
        ``ParetoStatistics`` including HV, sparsity, and the reference point
        in both max and min spaces.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    if F.size == 0:
        raise ValueError('F must contain at least one objective vector')

    n_points, n_objs = F.shape
    ref_max = _as_ref_point_max(n_objs, ref_point_max)
    ref_min = -ref_max

    nd_idx = get_nondominated(F)
    F_nd = F[nd_idx]
    num_nd = int(F_nd.shape[0])

    if num_nd == 0:
        return ParetoStatistics(
            hypervolume=float('nan'),
            sparsity=float('nan'),
            num_points=n_points,
            num_nondominated=0,
            ref_point_max=ref_max,
            ref_point_min=ref_min,
            ideal_max=np.full(n_objs, np.nan),
            nadir_max=np.full(n_objs, np.nan),
        )

    ideal_max = np.max(F_nd, axis=0)
    nadir_max = np.min(F_nd, axis=0)

    # Maximization -> minimization for pymoo HV.
    F_min = -F_nd.copy()
    hv, ref_min_used = hypervolume_from_nondominated(F_min, ref_point_min=ref_min)

    # Spacing on per-front normalized ND set (maximization values normalized,
    # then negated so the indicator sees a minimization front).
    if num_nd == 1:
        sparsity = float('nan')  # spacing is undefined for a singleton
    else:
        F_norm = normalize(F_nd.copy())
        F_min_norm = -F_norm
        sparsity = sparsity_from_normalized_nondominated(F_min_norm)

    return ParetoStatistics(
        hypervolume=hv,
        sparsity=sparsity,
        num_points=n_points,
        num_nondominated=num_nd,
        ref_point_max=ref_max,
        ref_point_min=ref_min_used,
        ideal_max=ideal_max,
        nadir_max=nadir_max,
    )


def get_pareto_statistics(
    F,
    ref_point_max: Optional[Union[float, Sequence[float], np.ndarray]] = None,
):
    """Backward-compatible ``(hypervolume, sparsity)`` wrapper.

    Prefer ``compute_pareto_statistics`` when the reference point / ideal /
    nadir must be logged or inspected.
    """
    stats = compute_pareto_statistics(F, ref_point_max=ref_point_max)
    return stats.hypervolume, stats.sparsity
