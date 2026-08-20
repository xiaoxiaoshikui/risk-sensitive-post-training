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

echo "== GPU check =="
python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible -- wrong instance/template?"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY

echo "== installing deps =="
pip install -r requirements.txt -q

mkdir -p runs

echo "== smoke test: confirm the installed trl matches the pinned patch =="
python - <<'PY'
import trl
pinned = "1.10.0"  # keep in sync with PINNED_TRL_VERSION in scripts/train_grpo.py
if trl.__version__ != pinned:
    raise SystemExit(
        f"installed trl=={trl.__version__} but rsp/_trl_patches/grpo_1_10_0.py "
        f"is a full-method copy pinned to trl=={pinned} -- re-diff and update "
        f"the patch before spending GPU time (see rsp/_trl_patches/README.md)"
    )
print("trl version ok:", trl.__version__)
PY

echo "== unit tests =="
python -m pytest tests -q

echo "== [1/5] SFT =="
python scripts/train_sft.py --config configs/sft.yaml

echo "== [2/5] DPO: build on-policy pairs, then train =="
python scripts/build_pairs.py --config configs/dpo.yaml
python scripts/train_dpo.py   --config configs/dpo.yaml

echo "== [3/5] GRPO: mean baseline (control) =="
python scripts/train_grpo.py --config configs/grpo_mean.yaml

echo "== [4/5] GRPO: cvar / entropic (treatment arms) =="
python scripts/train_grpo.py --config configs/grpo_cvar.yaml
python scripts/train_grpo.py --config configs/grpo_entropic.yaml

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
