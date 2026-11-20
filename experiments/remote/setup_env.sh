#!/usr/bin/env bash
# One-time setup of a training machine (Linux/macOS): virtual environment, PyTorch,
# project requirements, and a 2-minute end-to-end check.
#
#   bash experiments/remote/setup_env.sh            # PyTorch default wheel
#   TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash experiments/remote/setup_env.sh
#
# TORCH_INDEX selects a CUDA build of PyTorch matching the machine's driver (see
# https://pytorch.org/get-started/locally/). Guide: docs/methods/remote_training.md
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-python3}"
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -n "${TORCH_INDEX:-}" ]]; then
    pip install torch --index-url "$TORCH_INDEX"
fi
pip install -r requirements-rl.txt

python - <<'EOF'
import os, torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      "| CPUs:", os.cpu_count())
EOF

# fast tests of the training code, then a tiny end-to-end run (deleted afterwards)
python -m pytest tests/test_fastsim.py tests/test_rl2_model.py tests/test_rl2_search.py \
    tests/test_rl2_training.py -q
python experiments/rl2_train.py --tag setup_check --iters 1 --episodes0 16 --episodes 8 \
    --epochs0 1 --epochs 1 --val 2 --horizon 60 --device auto
rm -rf models/setup_check* results/experiments/setup_check
echo "setup OK"
