#!/bin/bash
#SBATCH --job-name=po_power_sim
#SBATCH --account=ssd
#SBATCH --partition=ssd
#SBATCH --qos=ssd
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# -------------------------------------------------------------------
# HPC SLURM submission for PO Incentives Power Simulation
#
# Runs full sweep: 1800 combos x 1000 sims on a single 48-core node.
# Output: MDE tables per (duration, h_init) combination.
#
# Usage:
#   sbatch submit_hpc.sh
# -------------------------------------------------------------------

mkdir -p logs output

module load python/3.11.9

echo "Running power sweep on $(hostname) with 48 cores"
echo "Start: $(date)"

python3 run_simulation.py --n_sims 1000 --n_workers 48

echo "Done: $(date)"
