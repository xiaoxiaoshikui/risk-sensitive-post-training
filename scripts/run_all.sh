#!/usr/bin/env bash
# Full pipeline for a rented GPU box (vast.ai / RunPod, PyTorch+CUDA template).
# Run inside tmux/screen -- this takes several hours and an SSH drop should
# not kill the job.
#
#   tmux new -s rsp
#   bash scripts/run_all.sh 2>&1 | tee runs/run_all.log
#
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/kaggle/_common.sh

echo "== GPU check =="
gpu_check

echo "== installing deps =="
ensure_deps

mkdir -p runs

echo "== smoke test: confirm the installed trl matches the pinned patch =="
advantage_hook_check

echo "== unit tests =="
python -m pytest tests -q

ACCEL_FLAGS=$(accel_launch_flags)
echo "[rsp] accelerate launch $ACCEL_FLAGS"

echo "== [1/5] SFT =="
accelerate launch $ACCEL_FLAGS scripts/train_sft.py --config configs/sft.yaml

echo "== [2/5] DPO: build on-policy pairs, then train =="
python scripts/build_pairs.py --config configs/dpo.yaml
accelerate launch $ACCEL_FLAGS scripts/train_dpo.py --config configs/dpo.yaml

echo "== [3/5] GRPO: mean baseline (control) =="
accelerate launch $ACCEL_FLAGS scripts/train_grpo.py --config configs/grpo_mean.yaml

echo "== [4/5] GRPO: cvar / entropic (treatment arms) =="
accelerate launch $ACCEL_FLAGS scripts/train_grpo.py --config configs/grpo_cvar.yaml
accelerate launch $ACCEL_FLAGS scripts/train_grpo.py --config configs/grpo_entropic.yaml

echo "== [5/5] evaluation, k=8, 200 held-out prompts =="
for r in sft dpo grpo_mean grpo_cvar grpo_entropic; do
  python scripts/evaluate.py --model runs/$r/final --k 8 --limit 200 --out runs/$r/eval
done

echo "== paired comparisons vs the mean-baseline control =="
python scripts/compare.py runs/grpo_mean/eval runs/grpo_cvar/eval     | tee runs/compare_cvar.json
python scripts/compare.py runs/grpo_mean/eval runs/grpo_entropic/eval | tee runs/compare_entropic.json

echo "== done =="
echo "per-run reports: runs/*/eval/report.json"
echo "paired comparisons: runs/compare_cvar.json runs/compare_entropic.json"
