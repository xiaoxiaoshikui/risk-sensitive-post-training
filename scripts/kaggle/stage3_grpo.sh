#!/usr/bin/env bash
# One GRPO arm per call -- run each in its own Kaggle session slot so a
# session timeout only costs you one arm, not all three.
#   bash scripts/kaggle/stage3_grpo.sh mean
#   bash scripts/kaggle/stage3_grpo.sh cvar
#   bash scripts/kaggle/stage3_grpo.sh entropic
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

arm="${1:?usage: stage3_grpo.sh mean|cvar|entropic}"
case "$arm" in
  mean|cvar|entropic) ;;
  *) echo "unknown arm '$arm', expected mean|cvar|entropic" >&2; exit 1 ;;
esac

ensure_deps
advantage_hook_check
if [ ! -d runs/sft/final ]; then
  echo "runs/sft/final not found -- restore it from the stage1 output before running this" >&2
  exit 1
fi

outdir="runs/grpo_${arm}"
if [ -d "${outdir}/final" ]; then
  echo "[skip] ${outdir}/final already exists"
else
  ACCEL_FLAGS=$(accel_launch_flags)
  echo "[rsp] accelerate launch $ACCEL_FLAGS"
  accelerate launch $ACCEL_FLAGS scripts/train_grpo.py --config "configs/grpo_${arm}.yaml"
fi
