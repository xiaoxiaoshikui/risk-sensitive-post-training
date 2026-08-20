#!/usr/bin/env python3
"""
GRPO with pluggable risk-sensitive advantage estimation.

Wraps TRL's GRPOTrainer and overrides only the advantage computation, so the
comparison between estimators is genuinely controlled: same rollouts, same
rewards, same optimizer, same seed -- one function differs.

    python scripts/train_grpo.py --config configs/grpo_mean.yaml
    python scripts/train_grpo.py --config configs/grpo_cvar.yaml

TRL's internal API for advantage computation has moved between releases. The
override is applied defensively and the script fails loudly rather than
silently falling back to standard GRPO, because a silent fallback would make
the two arms of the experiment identical while still producing plausible
numbers.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rsp.rewards import gsm8k_reward  # noqa: E402
from rsp.risk import RiskConfig, batch_advantages  # noqa: E402

SYSTEM_PROMPT = (
    "Solve the problem. Reason step by step, then give the final numeric "
    "answer inside <answer></answer> tags."
)


def build_reward_fn(log_sink: list):
    """TRL reward functions receive lists of completions and dataset columns."""

    def reward_fn(completions, answer, **kwargs):
        out = []
        for completion, gold in zip(completions, answer):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            b = gsm8k_reward(text, gold)
            log_sink.append((b.correct, b.format_ok))
            out.append(b.reward)
        return out

    return reward_fn


def patch_advantages(trainer, cfg: RiskConfig, group_size: int):
    """Replace GRPO's mean-baseline advantage with the configured estimator."""
    import torch

    target = None
    for name in ("_compute_advantages", "compute_advantages"):
        if hasattr(trainer, name):
            target = name
            break
    if target is None:
        raise RuntimeError(
            "Could not locate TRL's advantage hook on GRPOTrainer. Inspect "
            "trl.GRPOTrainer for the current method name and update "
            "patch_advantages -- do not run the experiment until this is fixed, "
            "or the risk arm will silently be standard GRPO."
        )

    def _risk_advantages(rewards, *args, **kwargs):
        flat = rewards.detach().float().cpu().numpy().reshape(-1)
        if flat.size % group_size != 0:
            raise RuntimeError(
                f"reward tensor of size {flat.size} is not divisible by group "
                f"size {group_size}; check num_generations"
            )
        grouped = flat.reshape(-1, group_size)
        adv = batch_advantages(grouped, cfg).reshape(-1)
        return torch.as_tensor(adv, dtype=rewards.dtype, device=rewards.device).view_as(rewards)

    setattr(trainer, target, _risk_advantages)
    print(f"[rsp] patched {target} -> estimator={cfg.estimator} "
          f"alpha={cfg.alpha} beta={cfg.beta} lambda={cfg.lambda_risk}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conf = yaml.safe_load(pathlib.Path(args.config).read_text())
    risk = RiskConfig(**conf["risk"])
    outdir = pathlib.Path(args.out or conf["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    ds = load_dataset(conf["dataset"], conf.get("dataset_config"), split=conf["split"])
    if conf.get("max_samples"):
        ds = ds.select(range(min(conf["max_samples"], len(ds))))

    ds = ds.map(lambda r: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": r[conf["question_field"]]}],
        "answer": r[conf["answer_field"]],
    })

    train_args = GRPOConfig(output_dir=str(outdir), **conf["training"])
    group_size = train_args.num_generations

    reward_log: list = []
    trainer = GRPOTrainer(
        model=conf["model"],
        args=train_args,
        train_dataset=ds,
        reward_funcs=build_reward_fn(reward_log),
        peft_config=LoraConfig(**conf["lora"]) if conf.get("lora") else None,
    )

    if risk.estimator != "mean":
        patch_advantages(trainer, risk, group_size)

    trainer.train()
    trainer.save_model(str(outdir / "final"))

    arr = np.array(reward_log, dtype=bool)
    (outdir / "run.json").write_text(json.dumps({
        "config": conf,
        "risk": {"estimator": risk.estimator, "alpha": risk.alpha,
                 "beta": risk.beta, "lambda_risk": risk.lambda_risk},
        "group_size": group_size,
        "train_rollouts": int(arr.shape[0]) if arr.size else 0,
        "train_correct_rate": float(arr[:, 0].mean()) if arr.size else None,
        "train_format_ok_rate": float(arr[:, 1].mean()) if arr.size else None,
    }, indent=2), encoding="utf-8")
    print(f"[rsp] wrote {outdir/'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
