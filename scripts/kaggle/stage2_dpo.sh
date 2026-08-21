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

if [ -f data/dpo_pairs.jsonl ] && [ ! -f data/dpo_pairs.jsonl.progress ]; then
  echo "[skip] data/dpo_pairs.jsonl already complete"
else
  # --limit 500, not build_pairs.py's default 2000: measured throughput on a
  # T4 was ~33s/prompt (bf16, before the fp16 fix), so 2000 prompts risks
  # exceeding a Kaggle session's wall-clock cap -- this happened once
  # already, losing everything because the old code only wrote output at
  # the very end. build_pairs.py now writes incrementally and resumes from
  # data/dpo_pairs.jsonl.progress if this gets cut off again regardless.
  python scripts/build_pairs.py --config configs/dpo.yaml --limit 500
fi

if [ -d runs/dpo/final ]; then
  echo "[skip] runs/dpo/final already exists"
else
  ACCEL_FLAGS=$(accel_launch_flags)
  echo "[rsp] accelerate launch $ACCEL_FLAGS"
  accelerate launch $ACCEL_FLAGS scripts/train_dpo.py --config configs/dpo.yaml
fi
