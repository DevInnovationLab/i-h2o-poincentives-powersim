#!/usr/bin/env python3
"""Run power simulation and output MDE tables.

Produces a table of minimum detectable effects (MDE) at 80% power,
with rows = baseline compliance, columns = AR(1) persistence.
One table per study duration (6 months, 1 year).

Usage:
    python run_simulation.py [--n_sims 1000] [--n_workers 6]

    # Quick test run:
    python run_simulation.py --n_sims 100
"""

import argparse
import itertools
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    AP_CONFIG, PARAM_GRID, STUDY_DURATIONS, SEED, N_SIMS,
    sigma_from_mu,
)
from estimate import run_single_sim

DEFAULT_WORKERS = min(6, cpu_count())

H_INIT = 0.0


def run_power_for_combo(args):
    params, n_sims, base_seed = args
    results = []
    for i in range(n_sims):
        res = run_single_sim(params, seed=base_seed + i)
        results.append(res)

    rej_arr = np.array([r['rejected'] for r in results], dtype=float)
    att_arr = np.array([r['att_hat'] for r in results], dtype=float)
    se_arr = np.array([r['se'] for r in results], dtype=float)

    return {
        'mu_baseline': params['mu_baseline'],
        'target_att': params['target_att'],
        'rho': params['rho'],
        'study_end_week': params['study_end_week'],
        'power': np.nanmean(rej_arr),
        'mean_att': np.nanmean(att_arr),
        'mean_se': np.nanmean(se_arr),
    }


def compute_mde(power_df, mu_baseline, rho):
    sub = power_df[
        (power_df['mu_baseline'] == mu_baseline) & (power_df['rho'] == rho)
    ].sort_values('target_att')

    powers = sub['power'].values
    target_atts = sub['target_att'].values

    if not (powers >= 0.80).any():
        return np.nan

    first_idx = np.where(powers >= 0.80)[0][0]
    if first_idx == 0:
        return target_atts[0]

    p_lo, p_hi = powers[first_idx - 1], powers[first_idx]
    t_lo, t_hi = target_atts[first_idx - 1], target_atts[first_idx]
    if p_hi > p_lo:
        return t_lo + (0.80 - p_lo) / (p_hi - p_lo) * (t_hi - t_lo)
    return t_hi


def format_mde_table(power_df, mu_baselines, rhos):
    records = []
    for mu in mu_baselines:
        row = {'Baseline Compliance': mu}
        for rho in rhos:
            mde = compute_mde(power_df, mu, rho)
            row[f'rho={rho}'] = mde
        records.append(row)
    return pd.DataFrame(records).set_index('Baseline Compliance')


def main():
    parser = argparse.ArgumentParser(description="Run power simulation and output MDE tables.")
    parser.add_argument('--n_sims', type=int, default=N_SIMS,
                        help=f"Simulations per parameter combo (default: {N_SIMS})")
    parser.add_argument('--n_workers', type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mu_baselines = sorted(PARAM_GRID['mu_baseline'])
    target_atts = sorted(PARAM_GRID['target_att'])
    rhos = sorted(PARAM_GRID['rho'])

    tasks = []
    combo_idx = 0
    for study_end_week in STUDY_DURATIONS.values():
        for combo in itertools.product(mu_baselines, target_atts, rhos):
            mu, target_att, rho = combo
            params = {
                'mu_baseline': mu,
                'sigma_baseline': sigma_from_mu(mu),
                'target_att': target_att,
                'rho': rho,
                'h_init': H_INIT,
                'study_end_week': study_end_week,
            }
            base_seed = SEED + combo_idx * args.n_sims
            tasks.append((params, args.n_sims, base_seed))
            combo_idx += 1

    print(f"Sites: {AP_CONFIG['n_sites']} ({AP_CONFIG['n_treated']} treated, "
          f"{AP_CONFIG['n_sites'] - AP_CONFIG['n_treated']} control)")
    print(f"Installation: {AP_CONFIG['n_sites']} villages over "
          f"{len(AP_CONFIG['install_schedule'])} weeks "
          f"({AP_CONFIG['install_schedule'][0]}/week)")
    print(f"Durations: {', '.join(STUDY_DURATIONS.keys())}")
    print(f"Baseline compliance: {mu_baselines}")
    print(f"AR(1) persistence: {rhos}")
    print(f"Target effects: {target_atts}")
    print(f"Combos: {len(tasks)} | Sims/combo: {args.n_sims} | "
          f"Total: {len(tasks) * args.n_sims:,}")
    print(f"Workers: {args.n_workers}")
    print()

    t0 = time.time()
    results = []
    with Pool(processes=args.n_workers) as pool:
        for result in tqdm(pool.imap_unordered(run_power_for_combo, tasks),
                           total=len(tasks), desc="Power sweep"):
            results.append(result)
    elapsed = time.time() - t0

    df = pd.DataFrame(results)
    raw_path = os.path.join(args.output_dir, 'power_results.csv')
    df.to_csv(raw_path, index=False)

    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / 60:.1f} min)\n")

    for label, study_end_week in STUDY_DURATIONS.items():
        duration_df = df[df['study_end_week'] == study_end_week]
        table = format_mde_table(duration_df, mu_baselines, rhos)

        display = table.copy()
        for col in display.columns:
            display[col] = display[col].apply(
                lambda v: f'{v:.3f}' if pd.notna(v) else '>0.40')

        print(f"MDE at 80% Power — {label} "
              f"(n={AP_CONFIG['n_sites']}, 2 tests/week)")
        print("=" * 55)
        print(display.to_string())
        print()

        csv_path = os.path.join(
            args.output_dir, f'mde_table_{label.replace(" ", "_")}.csv')
        table.to_csv(csv_path)

    print(f"Raw results: {raw_path}")
    for label in STUDY_DURATIONS:
        print(f"MDE table ({label}): "
              f"{os.path.join(args.output_dir, f'mde_table_{label.replace(chr(32), chr(95))}.csv')}")


if __name__ == '__main__':
    main()
