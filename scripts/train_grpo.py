#!/usr/bin/env python3
"""
GRPO with pluggable risk-sensitive advantage estimation.

Wraps TRL's GRPOTrainer and overrides only the advantage computation, so the
comparison between estimators is genuinely controlled: same rollouts, same
rewards, same optimizer, same seed -- one function differs.

    python scripts/train_grpo.py --config configs/grpo_mean.yaml
    python scripts/train_grpo.py --config configs/grpo_cvar.yaml

No public TRL release exposes a small, dedicated hook for this -- advantage
computation is inlined inside GRPOTrainer's ~600-line
`_generate_and_score_completions`, alongside generation, tool calls and
logging. The override binds a patched copy of that whole method
(rsp/_trl_patches/grpo_1_10_0.py), pinned to an exact TRL version, and fails
loudly rather than silently applying a stale patch to a mismatched TRL
release -- see rsp/_trl_patches/README.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from rsp.checkpoint_utils import fix_saved_base_model_path, resolved_checkpoint  # noqa: E402
from rsp.rewards import gsm8k_reward  # noqa: E402
from rsp.risk import RiskConfig  # noqa: E402

PINNED_TRL_VERSION = "1.10.0"

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


def patch_advantages(trainer, cfg: RiskConfig) -> None:
    """Replace GRPO's mean-baseline advantage with the configured estimator.

    Binds rsp/_trl_patches/grpo_1_10_0.py's copy of
    `_generate_and_score_completions` onto `trainer`, rebuilt against TRL's
    own module globals (not this script's) so every name the copied TRL code
    relies on -- torch, nanstd, gather_object, ... -- resolves against TRL's
    live module state rather than being re-imported by hand.
    """
    import trl
    from trl.trainer import grpo_trainer as grpo_module

    from rsp._trl_patches.grpo_1_10_0 import _generate_and_score_completions as patched_fn

    if trl.__version__ != PINNED_TRL_VERSION:
        raise RuntimeError(
            f"rsp's GRPO advantage patch is a full-method copy written against "
            f"trl=={PINNED_TRL_VERSION}, but trl=={trl.__version__} is installed. "
            f"Do not run the risk arms until rsp/_trl_patches/grpo_1_10_0.py has "
            f"been re-diffed and updated for this version (see "
            f"rsp/_trl_patches/README.md) -- a version mismatch here means the "
            f"patch may silently disagree with the running TRL internals."
        )
    if trainer.multi_objective_aggregation != "sum_then_normalize":
        raise RuntimeError(
            "rsp's GRPO advantage patch only implements the 'sum_then_normalize' "
            f"branch (this repo uses one combined reward function); trainer has "
            f"multi_objective_aggregation={trainer.multi_objective_aggregation!r}"
        )

    rebound = types.FunctionType(
        patched_fn.__code__,
        grpo_module.__dict__,
        patched_fn.__name__,
        patched_fn.__defaults__,
        patched_fn.__closure__,
    )
    rebound.__kwdefaults__ = patched_fn.__kwdefaults__

    trainer._rsp_risk_cfg = cfg
    trainer._generate_and_score_completions = types.MethodType(rebound, trainer)
    print(f"[rsp] patched _generate_and_score_completions (trl=={PINNED_TRL_VERSION}) "
          f"-> estimator={cfg.estimator} alpha={cfg.alpha} beta={cfg.beta} "
          f"lambda={cfg.lambda_risk}")


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
    # GRPOTrainer resolves model=<path> via AutoConfig.from_pretrained, which
    # can't load a bare LoRA adapter directory (runs/sft/final has no
    # config.json) the way AutoModelForCausalLM.from_pretrained can -- merge
    # onto the base model first. See rsp/checkpoint_utils.py.
    with resolved_checkpoint(conf["model"]) as model_path:
        trainer = GRPOTrainer(
            model=model_path,
            args=train_args,
            train_dataset=ds,
            reward_funcs=build_reward_fn(reward_log),
            peft_config=LoraConfig(**conf["lora"]) if conf.get("lora") else None,
        )

    if risk.estimator != "mean":
        patch_advantages(trainer, risk)

    trainer.train()
    final_dir = str(outdir / "final")
    trainer.save_model(final_dir)
    # save_model() recorded the merged temp dir (already deleted) as this
    # adapter's base -- repoint it at its immediate parent checkpoint, *not*
    # that parent's own base (which would silently drop the SFT stage from
    # every future load). See fix_saved_base_model_path.
    fix_saved_base_model_path(final_dir, conf["model"])

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
