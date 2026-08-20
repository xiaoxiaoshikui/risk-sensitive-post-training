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
  python - <<'PY'
from trl import GRPOTrainer
for name in ("_compute_advantages", "compute_advantages"):
    if hasattr(GRPOTrainer, name):
        print("advantage hook ok:", name)
        break
else:
    raise SystemExit(
        "no known advantage hook on GRPOTrainer -- TRL's API has moved, fix "
        "scripts/train_grpo.py:patch_advantages before training the risk arms"
    )
PY
}
