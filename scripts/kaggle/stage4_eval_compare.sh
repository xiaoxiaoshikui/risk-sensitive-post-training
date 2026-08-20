#!/usr/bin/env bash
# Final session. Needs runs/{sft,dpo,grpo_mean,grpo_cvar,grpo_entropic}/final
# all restored first.
#   bash scripts/kaggle/stage4_eval_compare.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ensure_deps

for r in sft dpo grpo_mean grpo_cvar grpo_entropic; do
  if [ ! -d "runs/$r/final" ]; then
    echo "runs/$r/final not found -- restore all five checkpoints before running this" >&2
    exit 1
  fi
  if [ -f "runs/$r/eval/report.json" ]; then
    echo "[skip] runs/$r/eval already exists"
  else
    python scripts/evaluate.py --model "runs/$r/final" --k 8 --limit 200 --out "runs/$r/eval"
  fi
done

python scripts/compare.py runs/grpo_mean/eval runs/grpo_cvar/eval     | tee runs/compare_cvar.json
python scripts/compare.py runs/grpo_mean/eval runs/grpo_entropic/eval | tee runs/compare_entropic.json

echo "== done =="
echo "per-run reports: runs/*/eval/report.json"
echo "paired comparisons: runs/compare_cvar.json runs/compare_entropic.json"
