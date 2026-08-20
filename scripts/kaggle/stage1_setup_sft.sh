#!/usr/bin/env bash
# Kaggle session 1, part A. Run from the repo root:
#   bash scripts/kaggle/stage1_setup_sft.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_deps
gpu_check
advantage_hook_check
python -m pytest tests -q

mkdir -p runs
if [ -d runs/sft/final ]; then
  echo "[skip] runs/sft/final already exists"
else
  python scripts/train_sft.py --config configs/sft.yaml
fi
