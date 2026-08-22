# Build (this repo is CUDA-only via prebuilt wheels, no CUDA toolkit needed at
# build time -- vllm has no arm64 wheels, so cross-build from Apple Silicon
# needs the explicit platform flag):
#   docker build --platform linux/amd64 -t risk-sensitive-post-training .
#
# Run on a CUDA host (needs the NVIDIA Container Toolkit so --gpus is
# recognized -- the image itself carries no CUDA toolkit, only the CUDA
# runtime libraries pip's torch/vllm wheels bundle):
#   docker run --gpus all -it risk-sensitive-post-training
#   docker run --gpus all risk-sensitive-post-training \
#       python scripts/train_sft.py --config configs/sft.yaml
#
# For Apptainer/Singularity (e.g. CSCS Alps and similar HPC clusters, where
# an unprivileged Docker daemon usually isn't available): pull this image
# straight from a registry with `apptainer pull docker://...` -- untested
# here since it needs a real Apptainer host, but this is the standard,
# documented path for running a Docker image under Slurm without a daemon.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/risk-sensitive-post-training

COPY requirements.txt .
# A from-scratch resolve (nothing preinstalled to leave stale) is exactly
# the case that stayed consistent throughout this project's own debugging
# -- see the README's "What broke, and how it was caught": every observed
# torch/vllm/torchaudio CUDA-build mismatch came from a *preinstalled* torch
# on a rented GPU box's base image being left untouched while vllm resolved
# a different one, not from resolving everything together in one pass. The
# defensive uninstall stays anyway, cheaply, in case a future requirements.txt
# change reintroduces a stale torchaudio/torchcodec pull (see
# scripts/kaggle/_common.sh's ensure_deps(), the same guard used for every
# rented box in this project).
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y --no-cache-dir torchaudio torchcodec 2>/dev/null || true

COPY . .

# CPU-only smoke gate: the risk math, estimator limits, reward parsing, and
# the trl-patch-sync check (tests/test_trl_patch_sync.py) against whatever
# trl version requirements.txt resolved -- none of this needs a GPU, so it
# runs at build time and fails the build if it fails, instead of surfacing
# hours into a rented-GPU training run.
RUN python -m pytest tests -q

CMD ["bash"]
