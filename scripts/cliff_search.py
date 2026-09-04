"""Find the naive-MORLAX sparsity cliff for one domain (set-and-forget).

Pipeline (one GPU, sequential probes, domains run in parallel via Slurm):

  1. Dense 1-seed run (gating off) -> nadir / ideal / HV reference.
  2. 1-seed geometric scan in ``p`` (finer than the walker 2x ladder), then
     log-space bisection between last-alive and first-dead.
  3. 3-seed confirm at those two ``p`` values.

Thresholds are range-normalized:

    T = nadir + p * (ideal - nadir)

Hypervolume uses a maximization-space reference slightly below the dense
nadir so negative energy (walker/cheetah) still contributes. A probe is
*collapsed* if HV is no better than an all-zero front (gated plateau) or
< 2% of dense HV. Collapsed probes abort after a few dead evals.

Writes ``<save-dir>/cliff.json``. W&B group is ``cliff-search-<domain>-100m``.

Jobs are meant to fit an 8h GPU slot (not a multi-day reservation). Each
MORLAX probe is a full run, but collapsed ones abort after a few dead evals.
If the wallclock is hit, resubmit the same script with ``--resume``; finished
probes in ``cliff.json`` are skipped.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

ENTITY = 'raghavthakar-oregon-state-university'
PROJECT = 'SMORL'

DOMAINS = {
    'walker': 'config/morlax/mowalker_sparse.yaml',
    'hopper': 'config/morlax/mohopper_sparse.yaml',
    'cheetah': 'config/morlax/mocheetah_sparse.yaml',
    'ant': 'config/morlax/moant_sparse.yaml',
    'humanoid': 'config/morlax/mohumanoid_sparse.yaml',
}

DEFAULT_SAVE_ROOT = '/nfs/hpc/share/thakarr/SMORL/results/cliff_search'


def _import_training():
    import matplotlib
    matplotlib.use('Agg')
    import wandb
    import moplayground as mop
    import minimal_mjx as mm
    from moplayground.moppo.morlax import StopTraining
    from moplayground.utils.pareto import compute_pareto_statistics
    return wandb, mop, mm, StopTraining, compute_pareto_statistics


def thresholds_from_p(nadir, ideal, p):
    nadir = np.asarray(nadir, dtype=float)
    ideal = np.asarray(ideal, dtype=float)
    rng = np.maximum(ideal - nadir, 1e-6)
    return (nadir + float(p) * rng).tolist()


def geometric_ps(p_min, p_max, factor):
    ps = []
    p = float(p_min)
    while p <= p_max * 1.001:
        ps.append(p)
        p *= factor
    if ps[-1] < p_max * 0.999:
        ps.append(float(p_max))
    return ps


def hv_of(rewards, ref_point_max, compute_pareto_statistics):
    F = np.asarray(rewards, dtype=float)
    if F.ndim == 1:
        F = F[None, :]
    if F.size == 0:
        return float('nan')
    try:
        stats = compute_pareto_statistics(F, ref_point_max=ref_point_max)
        hv = float(stats.hypervolume)
    except Exception as exc:
        print(f'[cliff] HV failed: {exc}')
        return float('nan')
    return hv


def is_collapsed(hv, zero_hv, dense_hv):
    if hv != hv or hv is None:  # NaN
        return True
    if zero_hv is not None and np.isfinite(zero_hv) and hv <= max(zero_hv * 2.0, 1e-8):
        return True
    if dense_hv is not None and np.isfinite(dense_hv) and dense_hv > 0:
        if hv < 0.02 * dense_hv:
            return True
    return False


def scale_from_paretos(paretos):
    """Nadir/ideal from every eval return vector of a dense run."""
    blocks = [np.asarray(p, dtype=float) for p in paretos if p is not None]
    blocks = [b if b.ndim == 2 else b[None, :] for b in blocks if np.size(b)]
    if not blocks:
        raise RuntimeError('Dense run produced no eval returns.')
    F = np.concatenate(blocks, axis=0)
    nadir = F.min(axis=0)
    ideal = F.max(axis=0)
    rng = np.maximum(ideal - nadir, 1e-6)
    hv_ref = (nadir - 0.05 * rng).tolist()
    return nadir.tolist(), ideal.tolist(), hv_ref, F


def apply_run_overrides(
    base, *, name, save_dir, seed, timesteps, num_evals, thresholds, enabled,
):
    cfg = copy.deepcopy(base)
    cfg.name = name
    cfg.save_dir = str(save_dir)
    cfg.learning_params.base_ppo_params.seed = int(seed)
    cfg.learning_params.base_ppo_params.num_timesteps = int(timesteps)
    cfg.learning_params.base_ppo_params.num_evals = int(num_evals)
    thr_cfg = cfg.env_config.reward.episodic_threshold
    if not enabled or all(float(t) == 0.0 for t in thresholds):
        thr_cfg.enabled = False
        thr_cfg.thresholds = [0.0] * len(cfg.env_config.reward.optimization.objectives)
    else:
        thr_cfg.enabled = True
        thr_cfg.thresholds = [float(t) for t in thresholds]
    return cfg


def run_morlax(
    wandb, mop, mm, StopTraining, compute_pareto_statistics,
    cfg, group, job_type, extra, hv_ref, abort,
):
    env, _ = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    wandb_config = dict(cfg)
    wandb_config.update(extra)
    run = mm.utils.logging.initialize_wandb(
        name=cfg.name.replace('/', ''),
        entity=ENTITY,
        project=PROJECT,
        config=wandb_config,
        group=group,
        job_type=job_type,
        tags=['cliff-search', group, extra.get('domain', '')],
        reinit=True,
    )
    recorder = {'hvs': [], 'steps': [], 'paretos': [], 'aborted': False, 'final_hv': float('nan')}
    abort_hits = [0]

    def progress_fn(run, num_steps, metrics, save_dir, training_data):
        training_data.hv_ref_point_max = hv_ref
        rewards = metrics.get('reward')
        hv = float('nan')
        if rewards is not None:
            F = np.asarray(rewards, dtype=float)
            if F.ndim == 1:
                F = F[None, :]
            recorder['paretos'].append(F)
            hv = hv_of(F, hv_ref, compute_pareto_statistics)
        recorder['hvs'].append(hv)
        recorder['steps'].append(int(num_steps))
        recorder['final_hv'] = hv
        print(f'[cliff] step={int(num_steps)} hv={hv:.4g}')
        mop.utils.plotting.plot_mo_progress(
            num_steps=num_steps,
            metrics=metrics,
            training_data=training_data,
            save_dir=save_dir,
            run=run,
        )
        if not abort:
            return
        min_steps = int(abort['min_steps'])
        need = int(abort['consecutive'])
        dense_hv = abort.get('dense_hv')
        zero_hv = abort.get('zero_hv')
        if int(num_steps) < min_steps:
            abort_hits[0] = 0
            return
        if is_collapsed(hv, zero_hv, dense_hv):
            abort_hits[0] += 1
            print(f'[cliff] collapsed eval {abort_hits[0]}/{need}')
            if abort_hits[0] >= need:
                recorder['aborted'] = True
                raise StopTraining()
        else:
            abort_hits[0] = 0

    try:
        mop.learning.train_policy(
            cfg, env, eval_env, run=run, warn_github_changes=False,
            progress_fn=progress_fn,
        )
    finally:
        try:
            wandb.finish()
        except Exception:
            pass
    return recorder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', required=True, choices=sorted(DOMAINS))
    parser.add_argument('--base', default=None, help='Override domain YAML.')
    parser.add_argument('--save-dir', default=None)
    parser.add_argument('--group', default=None)
    parser.add_argument('--timesteps', type=int, default=100_000_000)
    parser.add_argument('--num-evals', type=int, default=21, help='~5M cadence at 100M.')
    parser.add_argument('--p-min', type=float, default=0.01)
    parser.add_argument('--p-max', type=float, default=0.50)
    parser.add_argument('--p-factor', type=float, default=1.25,
                        help='Geometric ratio (walker grid used 2.0).')
    parser.add_argument('--n-bisect', type=int, default=5,
                        help='Log-space bisections after the geometric bracket.')
    parser.add_argument('--confirm-seeds', default='0,1,2')
    parser.add_argument('--abort-min-steps', type=int, default=15_000_000)
    parser.add_argument('--abort-consecutive', type=int, default=3)
    parser.add_argument('--resume', action='store_true',
                        help='Skip dense/probes already recorded in cliff.json.')
    args = parser.parse_args()

    wandb, mop, mm, StopTraining, compute_pareto_statistics = _import_training()
    base_path = args.base or DOMAINS[args.domain]
    base = mop.utils.read_config(base_path)
    save_dir = Path(args.save_dir or f'{DEFAULT_SAVE_ROOT}/{args.domain}')
    save_dir.mkdir(parents=True, exist_ok=True)
    group = args.group or f'cliff-search-{args.domain}-100m'
    confirm_seeds = [int(x) for x in args.confirm_seeds.split(',') if x.strip() != '']
    n_obj = len(base.env_config.reward.optimization.objectives)
    cliff_path = save_dir / 'cliff.json'
    result = {
        'domain': args.domain,
        'base': base_path,
        'timesteps': args.timesteps,
        'p_min': args.p_min,
        'p_max': args.p_max,
        'p_factor': args.p_factor,
        'n_bisect': args.n_bisect,
        'probes': [],
        'confirm': [],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    }
    if args.resume and cliff_path.is_file():
        result.update(json.loads(cliff_path.read_text()))
        result.setdefault('probes', [])
        result.setdefault('confirm', [])
        print(f'[cliff] resumed from {cliff_path} '
              f'({len(result["probes"])} probes, {len(result["confirm"])} confirm)')

    print(f'=== Cliff search: {args.domain}  group={group}  T={args.timesteps} ===')

    def _probe_key(phase, p, seed):
        return (phase, round(float(p), 8), int(seed))

    def _cached(phase, p, seed):
        key = _probe_key(phase, p, seed)
        for row in result.get('probes', []) + result.get('confirm', []):
            if _probe_key(row['phase'], row['p'], row['seed']) == key:
                return row
        return None

    # --- 1. Dense scale ---
    if result.get('nadir') is not None and result.get('ideal') is not None:
        nadir = result['nadir']
        ideal = result['ideal']
        hv_ref = result['hv_ref_point_max']
        dense_hv = result['dense_hv']
        zero_hv = result['zero_hv']
        print('[cliff] dense scale already in cliff.json; skipping dense run.')
    else:
        dense_cfg = apply_run_overrides(
            base, name=f'{args.domain}-dense-seed0', save_dir=save_dir,
            seed=0, timesteps=args.timesteps, num_evals=args.num_evals,
            thresholds=[0.0] * n_obj, enabled=False,
        )
        print('\n===== Dense calibration (1 seed, gating off) =====')
        dense_rec = run_morlax(
            wandb, mop, mm, StopTraining, compute_pareto_statistics,
            dense_cfg, group, 'dense',
            {'domain': args.domain, 'phase': 'dense', 'p': 0.0, 'seed': 0},
            hv_ref=None, abort=None,
        )
        nadir, ideal, hv_ref, _F = scale_from_paretos(dense_rec['paretos'])
        dense_hv = hv_of(
            dense_rec['paretos'][-1], hv_ref, compute_pareto_statistics,
        )
        zero_hv = hv_of(np.zeros((1, n_obj)), hv_ref, compute_pareto_statistics)
        result.update({
            'nadir': nadir, 'ideal': ideal, 'hv_ref_point_max': hv_ref,
            'dense_hv': dense_hv, 'zero_hv': zero_hv,
        })
        print(f'[cliff] nadir={nadir}  ideal={ideal}  hv_ref={hv_ref}')
        print(f'[cliff] dense_hv={dense_hv:.4g}  zero_hv={zero_hv:.4g}')
        cliff_path.write_text(json.dumps(result, indent=2) + '\n')

    abort = {
        'min_steps': args.abort_min_steps,
        'consecutive': args.abort_consecutive,
        'dense_hv': dense_hv,
        'zero_hv': zero_hv,
    }

    def probe(p, seed, phase):
        cached = _cached(phase, p, seed)
        if cached is not None:
            print(f'[cliff] skip cached {phase} p={p:g} seed={seed} hv={cached.get("hv")}')
            return cached
        T = thresholds_from_p(nadir, ideal, p)
        tag = 'x'.join(f'{t:g}' for t in T)
        name = f'{args.domain}-{phase}-p={p:g}-thr={tag}-seed={seed}'
        cfg = apply_run_overrides(
            base, name=name, save_dir=save_dir, seed=seed,
            timesteps=args.timesteps, num_evals=args.num_evals,
            thresholds=T, enabled=True,
        )
        print(f'\n===== {phase}: p={p:g} T={T} seed={seed} =====')
        rec = run_morlax(
            wandb, mop, mm, StopTraining, compute_pareto_statistics,
            cfg, group, phase,
            {
                'domain': args.domain, 'phase': phase, 'p': float(p),
                'seed': int(seed), 'thresholds': T,
            },
            hv_ref=hv_ref,
            abort=abort if phase in ('probe', 'confirm-cliff') else None,
        )
        collapsed = is_collapsed(rec['final_hv'], zero_hv, dense_hv)
        row = {
            'p': float(p), 'thresholds': T, 'seed': int(seed),
            'phase': phase, 'hv': rec['final_hv'],
            'aborted': rec['aborted'], 'collapsed': collapsed,
            'steps': rec['steps'][-1] if rec['steps'] else 0,
        }
        print(f'[cliff] -> hv={rec["final_hv"]:.4g} collapsed={collapsed} aborted={rec["aborted"]}')
        bucket = 'confirm' if str(phase).startswith('confirm') else 'probes'
        result[bucket].append(row)
        cliff_path.write_text(json.dumps(result, indent=2) + '\n')
        return row

    # --- 2. Geometric scan + log bisection ---
    if result.get('p_alive') is not None and result.get('p_cliff') is not None:
        p_alive = result['p_alive']
        p_dead = result['p_cliff']
        print(f'[cliff] scan already finished: p_alive={p_alive:g} p_cliff={p_dead:g}')
    else:
        ps = geometric_ps(args.p_min, args.p_max, args.p_factor)
        print(f'[cliff] geometric p grid ({len(ps)}): ' + ', '.join(f'{p:.4g}' for p in ps))
        p_alive = None
        p_dead = None
        for p in ps:
            row = probe(p, seed=0, phase='probe')
            if row['collapsed']:
                p_dead = p
                break
            p_alive = p
        if p_dead is None:
            print('[cliff] no collapse up to p_max; treating p_max as still-alive.')
            p_alive = ps[-1]
            p_dead = ps[-1]
        elif p_alive is None:
            print('[cliff] collapsed at p_min; no fully-alive p found.')
            p_alive = args.p_min
            lo, hi = max(args.p_min / (args.p_factor ** 2), 1e-4), p_dead
            for i in range(args.n_bisect):
                mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                row = probe(mid, seed=0, phase='probe')
                if row['collapsed']:
                    hi = mid
                    p_dead = mid
                else:
                    lo = mid
                    p_alive = mid
            p_dead = hi
        else:
            lo, hi = p_alive, p_dead
            for i in range(args.n_bisect):
                if hi <= lo * 1.02:
                    break
                mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
                row = probe(mid, seed=0, phase='probe')
                if row['collapsed']:
                    hi = mid
                    p_dead = mid
                else:
                    lo = mid
                    p_alive = mid

        result['p_alive'] = p_alive
        result['p_cliff'] = p_dead
        result['thresholds_alive'] = thresholds_from_p(nadir, ideal, p_alive)
        result['thresholds_cliff'] = thresholds_from_p(nadir, ideal, p_dead)
        cliff_path.write_text(json.dumps(result, indent=2) + '\n')

    # --- 3. Confirm ---
    for p, label in ((p_alive, 'confirm-alive'), (p_dead, 'confirm-cliff')):
        for seed in confirm_seeds:
            row = probe(p, seed=seed, phase=label)

    cliff_path.write_text(json.dumps(result, indent=2) + '\n')
    print('\n===== Cliff search done =====')
    print(json.dumps({
        k: result[k] for k in (
            'domain', 'nadir', 'ideal', 'hv_ref_point_max', 'dense_hv',
            'p_alive', 'p_cliff', 'thresholds_alive', 'thresholds_cliff',
        )
    }, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
