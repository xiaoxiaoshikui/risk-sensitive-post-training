# Sourced by the stageN scripts. Not meant to be run directly.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Kaggle captures stdout through a pipe, not a TTY, so Python's default block
# buffering can sit on minutes of training output before flushing it -- from
# the log viewer this is indistinguishable from actually being stuck. Force
# unbuffered output so `kernels logs` reflects real progress.
export PYTHONUNBUFFERED=1

ensure_deps() {
  pip install -r requirements.txt -q
  # vllm pulls in torchaudio/torchcodec for multimodal audio/video decoding this
  # repo never touches (GSM8K is text-only). Both ship compiled extensions
  # built against a specific torch build; when pip's resolver lands on a
  # different torch than that build targeted -- which happened in practice,
  # since vllm's own torch pin can differ from the preinstalled one -- their
  # .so fails to load, and that's a *hard* crash (OSError / RuntimeError, not
  # ImportError) that propagates straight through transformers' AutoProcessor
  # lazy-load and vllm's own sampling_params import, breaking `import trl` and
  # `import vllm` entirely even though nothing here uses either package.
  # Uninstalling is safe: transformers/vllm both degrade an ImportError (a
  # clean "not installed") to "feature unavailable", just not a broken one.
  pip uninstall -y torchaudio torchcodec -q 2>/dev/null || true
}

gpu_check() {
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "no CUDA GPU visible -- check Settings > Accelerator"
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i}: {torch.cuda.get_device_name(i)} {p.total_memory/1e9:.0f} GB")
print(f"device_count={n}")
PY
}

# Echoes `accelerate launch` flags sized to however many GPUs this session
# actually got -- Kaggle's "T4 x2" accelerator is two physical GPUs, but a
# plain `python script.py` only ever uses GPU0. Detecting the count instead
# of hardcoding it means this doesn't break on a session that only hands out
# one GPU.
accel_launch_flags() {
  local nproc
  nproc=$(python -c "import torch; print(torch.cuda.device_count())")
  local flags="--num_processes $nproc --mixed_precision bf16"
  if [ "$nproc" -gt 1 ]; then
    flags="$flags --multi_gpu"
  fi
  echo "$flags"
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
