#!/usr/bin/env bash
# Slurm job for expert-iteration training (template: fill in the lines marked FILL).
#
#   sbatch experiments/remote/slurm_train.sbatch
#   TAG=rl2_gen_m SIZE=M sbatch --export=ALL experiments/remote/slurm_train.sbatch
#
# The job is requeue-safe: run_train.sh resumes from the last completed iteration, so a
# preempted or time-limited job can simply be submitted again (or requeued by Slurm).
# Guide: docs/methods/remote_training.md
#
#SBATCH --job-name=rl2-train
#SBATCH --cpus-per-task=32            # simulation workers = cpus - 1
#SBATCH --mem=48G
#SBATCH --gres=gpu:1                  # FILL: GPU syntax of your cluster (or remove for CPU-only)
#SBATCH --time=24:00:00               # FILL: within the partition limit
#SBATCH --requeue
#SBATCH --output=rl2-train-%j.out
##SBATCH --partition=FILL             # FILL: e.g. a GPU partition
##SBATCH --account=FILL               # FILL: if your cluster needs an account

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

# FILL: environment setup of your cluster, for example
# module load python/3.11 cuda/12.1
source .venv/bin/activate

nvidia-smi || true
bash experiments/remote/run_train.sh
