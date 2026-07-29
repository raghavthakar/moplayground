import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import time
from minimal_mjx.utils.plotting import get_subplot_grid
from moplayground.utils.pareto import (
    compute_pareto_statistics,
    get_pareto_statistics,
    get_nondominated,
)
import wandb
from pathlib import Path

@dataclass(frozen=False)
class MOTrainingPlottingInfo:
    start_time    : float
    times         : list = field(default_factory=list)
    iterations    : list = field(default_factory=list)
    paretos       : list = field(default_factory=list)
    directives    : list = field(default_factory=list)
    labels        : list = field(default_factory=list)
    
    def save(self, save_dir, create_time=True):
        pd.DataFrame(
            {
                'times': [self.start_time] + self.times if create_time else self.times,
                'iters': [0] + self.iterations if create_time else self.iterations
            }
        ).to_csv(save_dir)

def _as_numpy(x) -> np.ndarray:
    """Convert jax / list values to a concrete numpy array."""
    if hasattr(x, 'block_until_ready'):
        x = x.block_until_ready()
    return np.asarray(x)


def _objective_names(labels) -> list[str]:
    names = []
    for i, lab in enumerate(labels or []):
        if isinstance(lab, (list, tuple)):
            raw = '+'.join(str(x) for x in lab) if lab else f'obj_{i}'
        else:
            raw = str(lab) if lab not in (None, '') else f'obj_{i}'
        # wandb-friendly metric key fragment
        names.append(raw.replace(' ', '_').replace('/', '_'))
    return names


def _scalarize(value):
    """Reduce a metric value to a Python float when possible."""
    arr = _as_numpy(value)
    if arr.size == 1:
        return float(arr)
    return float(np.mean(arr))


def _log_mo_wandb(
    run: wandb.Run,
    num_steps: int,
    metrics: dict,
    rewards: np.ndarray,
    directives: np.ndarray,
    labels,
    elapsed_s: float,
    reward_plot_html: str | None = None,
):
    """Log MORL eval scalars, training losses, and a per-policy performance table."""
    rewards = np.asarray(rewards, dtype=float)
    directives = np.asarray(directives, dtype=float)
    if rewards.ndim == 1:
        rewards = rewards[None, :]
    if directives.ndim == 1:
        directives = directives[None, :]

    n_points, n_objs = rewards.shape
    obj_names = _objective_names(labels)
    while len(obj_names) < n_objs:
        obj_names.append(f'obj_{len(obj_names)}')

    nd_idx = set(int(i) for i in get_nondominated(rewards))
    try:
        stats = compute_pareto_statistics(rewards)  # default ref_point_max = 0
    except Exception as e:
        print(f'Warning: could not compute Pareto statistics: {e}')
        stats = None

    # --- Performance table: one row per evaluated preference / return vector ---
    columns = (
        ['step', 'eval_id', 'nondominated']
        + [f'w/{name}' for name in obj_names]
        + [f'return/{name}' for name in obj_names]
        + ['scalarized_return']
    )
    table = wandb.Table(columns=columns)
    for i in range(n_points):
        w = directives[i]
        r = rewards[i]
        scalarized = float(np.dot(w, r)) if w.shape[0] == r.shape[0] else float(np.sum(r))
        row = (
            [int(num_steps), int(i), i in nd_idx]
            + [float(x) for x in w]
            + [float(x) for x in r]
            + [scalarized]
        )
        table.add_data(*row)

    log_dict = {
        'eval/performances': table,
        'eval/num_points': int(n_points),
        'eval/num_nondominated': int(len(nd_idx)),
        'eval/coverage_ratio': float(len(nd_idx) / max(n_points, 1)),
        'time/elapsed_s': float(elapsed_s),
    }
    if reward_plot_html is not None:
        log_dict['reward_plot'] = wandb.Html(reward_plot_html)

    if stats is not None:
        log_dict.update({
            'eval/hypervolume': float(stats.hypervolume),
            'eval/sparsity': float(stats.sparsity),
        })
        for j, name in enumerate(obj_names):
            # Explicit HV reference (maximization-space units = raw returns).
            log_dict[f'eval/hv/ref_point_max/{name}'] = float(stats.ref_point_max[j])
            log_dict[f'eval/hv/ideal_max/{name}'] = float(stats.ideal_max[j])
            log_dict[f'eval/hv/nadir_max/{name}'] = float(stats.nadir_max[j])

        # Persist the HV convention once on the run config (idempotent).
        if not run.config.get('hv_definition'):
            run.config.update(
                {
                    'hv_definition': {
                        'sense': stats.sense,
                        'ref_point_max': stats.ref_point_max.tolist(),
                        'ref_point_min': stats.ref_point_min.tolist(),
                        'ref_point_max_meaning': (
                            'HV reference in maximization space (same units as '
                            'episode returns). Default is the origin: only '
                            'points that improve on 0 contribute volume.'
                        ),
                        'library': 'pymoo.indicators.hv.HV',
                        'sparsity': (
                            'pymoo SpacingIndicator on per-front min-max '
                            'normalized non-dominated front; NaN if |ND|<2'
                        ),
                        'notes': stats.notes,
                    }
                },
                allow_val_change=True,
            )

    for j, name in enumerate(obj_names):
        col = rewards[:, j]
        log_dict[f'eval/return/{name}/mean'] = float(np.mean(col))
        log_dict[f'eval/return/{name}/std'] = float(np.std(col))
        log_dict[f'eval/return/{name}/max'] = float(np.max(col))
        log_dict[f'eval/return/{name}/min'] = float(np.min(col))
        if j < directives.shape[1]:
            log_dict[f'eval/directive/{name}/mean'] = float(np.mean(directives[:, j]))

    # Training / eval timing scalars (when merged in from acting.run_evaluation)
    for key, value in metrics.items():
        if key in ('reward', 'directive'):
            continue
        if isinstance(key, str) and (
            key.startswith('training/')
            or key.startswith('eval/')
            or key.endswith('_loss')
            or key in ('total_loss', 'policy_loss', 'v_loss', 'entropy_loss')
        ):
            try:
                log_dict[key if '/' in key else f'training/{key}'] = _scalarize(value)
            except Exception:
                pass

    run.log(log_dict, step=num_steps)


def plot_mo_progress(
    num_steps       : int,
    metrics         : dict,
    training_data   : MOTrainingPlottingInfo,
    save_dir        : Path,
    run             : wandb.Run = None
):
    # print current time
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    # save data from iteration
    rewards = _as_numpy(metrics['reward'])
    directives = _as_numpy(metrics['directive'])
    training_data.iterations.append(num_steps)
    training_data.paretos.append(rewards)
    training_data.directives.append(directives)
    training_data.times.append(time.time())
    training_data.save(save_dir / 'progress.csv')

    if np.array(training_data.directives).shape[2] == 2:
        # create the plot
        fig, axs = plot_sequential_paretos(
            ax_titles   = training_data.iterations,
            paretos     = training_data.paretos,
            directives  = training_data.directives,
            objectives  = training_data.labels
        )
    else:
        fig, axs = plot_sequential_hypervolume(
            iterations    = training_data.iterations,
            paretos       = training_data.paretos
        )
    
    # save and upload to wandb
    fig.savefig(save_dir / 'progress.svg')
    plt.close(fig)
    if run:
        with open(save_dir / 'progress.svg', "r") as f:
            svg = f.read()
        elapsed_s = training_data.times[-1] - training_data.start_time
        _log_mo_wandb(
            run=run,
            num_steps=num_steps,
            metrics=metrics,
            rewards=rewards,
            directives=directives,
            labels=training_data.labels,
            elapsed_s=elapsed_s,
            reward_plot_html=svg,
        )

def default_coloring(tradeoff):
    # Map a tradeoff (or batch of tradeoffs) to an RGB color
    tradeoff = np.asarray(tradeoff, dtype=float)
    pad = max(0, 3 - tradeoff.shape[-1])
    if pad:
        tradeoff = np.pad(tradeoff, [(0, 0)] * (tradeoff.ndim - 1) + [(0, pad)])
    return tradeoff[..., :3]


def _decide_color_kwargs(colors, idx=None):
    """Build the matplotlib scatter color kwarg. A per-point ``(n, 3|4)`` array goes
    through ``c=`` (optionally indexed to ``idx``); anything else (a single named or
    RGB(A) color, or ``None``) goes through ``color=`` to avoid value-mapping."""
    if colors is None:
        return {}
    if isinstance(colors, np.ndarray) and colors.ndim == 2:
        return {"c": colors if idx is None else colors[idx]}
    return {"color": colors}


def plot_pareto(
    ax                    : plt.Axes,
    pareto                : np.ndarray,
    colors                : str | np.ndarray = None,
    objective             : list[str] = None,
    nondominated_alpha    : float = 1.0,
    dominated_alpha       : float = 1.0,
    label                 : str = None,
    set_lims              : bool = True,
):
    """
    Plot a pareto frontier.

    ``colors`` is a single matplotlib color or per-point color. RGB(A)
    ``label`` names the front in the legend; 
    ``set_lims`` zooms to the nondominated front.
    """
    num_objs = pareto.shape[1]
    if num_objs not in (2, 3):
        raise NotImplementedError('Only 2D and 3D paretos are supported for plotting')

    if objective is None: objective = [''] * num_objs
    c = np.asarray(colors) if isinstance(colors, (list, tuple, np.ndarray)) else colors
    nd_idx = get_nondominated(pareto)
    d_idx = np.setdiff1d(np.arange(pareto.shape[0]), nd_idx)
    clip = {'axlim_clip': True} if num_objs == 3 else {}

    ax.scatter(
        *(pareto[d_idx].T),
        s       = 8,
        alpha   = dominated_alpha,
        **_decide_color_kwargs(c, d_idx),
        **clip,
    )
    ax.scatter(
        *(pareto[nd_idx].T),
        alpha         = nondominated_alpha,
        zorder        = 1,
        s             = 12,
        edgecolors    = 'black',
        linewidths    = 1.5,
        label         = label,
        **_decide_color_kwargs(c, nd_idx),
        **clip,
    )

    if set_lims:
        ax.set_xlim((0.95 * np.min(pareto[nd_idx, 0]), 1.05 * np.max(pareto[nd_idx, 0])))
        ax.set_ylim((0.95 * np.min(pareto[nd_idx, 1]), 1.05 * np.max(pareto[nd_idx, 1])))
        if num_objs == 3:
            ax.set_zlim((1.00 * np.min(pareto[nd_idx, 2]), 1.05 * np.max(pareto[nd_idx, 2])))

    ax.set_xlabel(objective[0])
    ax.set_ylabel(objective[1])
    if num_objs == 3:
        ax.set_zlabel(objective[2])

    return ax

def plot_sequential_paretos(
    ax_titles: list[str],
    paretos: np.ndarray,
    directives: np.ndarray = None,
    objectives: list[str] = None
):
    if len(ax_titles) != len(paretos) != len(directives):
        raise Exception('Incompatible lengths of input arrays')

    if directives is None: directives = np.zeros_like(paretos)
    if objectives is None: objectives = [''] * paretos[0].shape[1]
    
    nrows, ncols = get_subplot_grid(len(ax_titles))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    if type(axs) == np.ndarray: axs = axs.flatten()
    else: axs = [axs]
    
    paretos = np.array(paretos)
    directives = np.array(directives)
    xlim   = np.array((np.min(paretos[..., 0]), np.max(paretos[..., 0])))
    ylim   = np.array((np.min(paretos[..., 1]), np.max(paretos[..., 1])))
    border = np.array([-1., 1.])
    xlim   = xlim + border * np.abs(xlim[1] - xlim[0]) * 0.1
    ylim   = ylim + border * np.abs(ylim[1] - ylim[0]) * 0.1
    for ax, x, y, d in zip(axs, ax_titles, paretos, directives):
        ax = plot_pareto(ax, y, default_coloring(d), objectives, set_lims=False)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(x)
        
    fig.set_size_inches((4 * ncols, 4 * nrows))
    fig.tight_layout()

    return fig, ax

def plot_sequential_hypervolume(
    iterations: list[int] | np.ndarray,
    paretos: np.ndarray
):
    fig, ax = plt.subplots()
    hvs = []
    sps = []
    for p in paretos:
        hv, sp = get_pareto_statistics(np.array(p.block_until_ready()))
        hvs.append(hv)
        sps.append(sp)
    ax2 = ax.twinx()
    ax.plot(iterations, hvs, label='Hypervolume')
    ax2.plot(iterations, sps, 'r-', label='Sparsity')
    fig.set_size_inches((10, 7))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Hypervolume')
    ax2.set_ylabel('Sparsity')
    ax.legend()
    ax2.legend()
    return fig, ax