# PO Incentives Power Simulation

## Overview
Power simulation for a staggered rollout RCT across 40 villages in AP, estimating minimum detectable effect (MDE) at 80% power for pump operator chlorination incentives. Uses site-level DiD (Callaway & Sant'Anna style). **No TWFE.**

## Running
```bash
pip install -r requirements.txt
python3 run_simulation.py --n_sims 1000
```

Output: MDE tables (rows = baseline compliance, columns = AR(1) persistence) for 6-month and 1-year durations.

## Key Parameters
- All sweep ranges are defined in `sweep_params.csv` — edit this file to change what gets swept
- `target_att`: the expected dynamic effect on chlorination rates (the estimand)
- `tau` (per-period impulse): derived from target_att using finite-horizon AR(1) formula
- 40 villages (20 treated, 20 control), installed at constant rate of 5/week over 8 weeks
- 2 chlorine measurements per week
- Study durations: 6 months (26 weeks) and 1 year (52 weeks)
- Training/baseline: 4 weeks (relative weeks 5-8)

## File Structure
- `sweep_params.csv` — Single source of truth for all parameter sweep ranges
- `config.py` — Loads sweep_params.csv, parameter grids, constants, install schedules
- `generate_data.py` — DGP: staggered rollout panel generation
- `estimate.py` — Site-level DiD estimator
- `run_simulation.py` — Main entry point: runs power sweep and outputs MDE tables
- `submit_hpc.sh` — SLURM submission (single node, 48 cores)

## Rules
- **No TWFE** — only use Callaway & Sant'Anna style DiD
- **Never run simulations locally** — always use HPC
- **Always keep README.md up to date** when making changes to the codebase, parameters, or run procedures
- **Commit with descriptive messages** after significant code changes
