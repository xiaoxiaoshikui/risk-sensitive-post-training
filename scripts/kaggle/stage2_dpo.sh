#!/usr/bin/env bash
# Kaggle session 1, part B. Needs runs/sft/final from stage1 (restore it first
# if this is a fresh session -- see the README's Kaggle section).
#   bash scripts/kaggle/stage2_dpo.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_deps
if [ ! -d runs/sft/final ]; then
  echo "runs/sft/final not found -- restore it from the stage1 output before running this" >&2
  exit 1
fi

if [ -f data/dpo_pairs.jsonl ]; then
  echo "[skip] data/dpo_pairs.jsonl already exists"
else
  python scripts/build_pairs.py --config configs/dpo.yaml
fi

if [ -d runs/dpo/final ]; then
  echo "[skip] runs/dpo/final already exists"
else
  python scripts/train_dpo.py --config configs/dpo.yaml
fi
