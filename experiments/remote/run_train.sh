#!/usr/bin/env bash
# Expert-iteration training on any Linux/macOS machine (workstation, cloud VM, or a
# cluster job). Resume-aware: if the run TAG already has a saved state, it continues
# from the last completed iteration instead of starting again.
#
#   TAG=rl2_gen_m SIZE=M bash experiments/remote/run_train.sh
#
# Every variable has a default; override any of them in the environment.
# Guide: docs/methods/remote_training.md
set -euo pipefail

cd "$(dirname "$0")/../.."                       # repository root

TAG="${TAG:-rl2_gen_s_big}"
SIZE="${SIZE:-S}"
ITERS="${ITERS:-10}"
EPISODES0="${EPISODES0:-4000}"
EPISODES="${EPISODES:-2000}"
EPOCHS0="${EPOCHS0:-8}"
EPOCHS="${EPOCHS:-4}"
REPLAY="${REPLAY:-2}"
VAL="${VAL:-50}"
PATIENCE="${PATIENCE:-2}"
DEVICE="${DEVICE:-auto}"                         # auto = CUDA if present
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
WORKERS="${WORKERS:-$(( CPUS > 2 ? CPUS - 1 : 1 ))}"   # one core for the trainer
THREADS="${THREADS:-4}"                          # CPU threads of the trainer

export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# one thread per simulation worker (the workers set it too; this covers BLAS)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "results/experiments/${TAG}"
LOG="results/experiments/${TAG}/train.log"

if [[ -f "results/experiments/${TAG}/state.pkl" ]]; then
    echo "$(date '+%F %T') resuming ${TAG}" | tee -a "$LOG"
    python experiments/rl2_train.py --tag "$TAG" --resume --iters "$ITERS" \
        --workers "$WORKERS" --threads "$THREADS" --device "$DEVICE" \
        --patience "$PATIENCE" 2>&1 | tee -a "$LOG"
else
    echo "$(date '+%F %T') starting ${TAG}" | tee -a "$LOG"
    python experiments/rl2_train.py --tag "$TAG" --size "$SIZE" --iters "$ITERS" \
        --episodes0 "$EPISODES0" --episodes "$EPISODES" --epochs0 "$EPOCHS0" \
        --epochs "$EPOCHS" --replay "$REPLAY" --val "$VAL" --patience "$PATIENCE" \
        --workers "$WORKERS" --threads "$THREADS" --device "$DEVICE" 2>&1 | tee -a "$LOG"
fi
