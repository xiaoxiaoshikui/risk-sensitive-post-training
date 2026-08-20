# Sourced by the stageN scripts. Not meant to be run directly.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ensure_deps() {
  pip install -r requirements.txt -q
}

gpu_check() {
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible -- check Settings > Accelerator"
print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
PY
}

advantage_hook_check() {
  # Keep this literal in sync with PINNED_TRL_VERSION in scripts/train_grpo.py.
  python - <<'PY'
import trl
pinned = "1.10.0"
if trl.__version__ != pinned:
    raise SystemExit(
        f"installed trl=={trl.__version__} but rsp/_trl_patches/grpo_1_10_0.py "
        f"is a full-method copy pinned to trl=={pinned} -- re-diff and update "
        f"the patch before training the risk arms (see rsp/_trl_patches/README.md)"
    )
print("trl version ok:", trl.__version__)
PY
}
