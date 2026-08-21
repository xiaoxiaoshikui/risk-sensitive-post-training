#!/usr/bin/env python3
"""Evaluate a checkpoint with tail-aware metrics.

Samples k rollouts per prompt so per-prompt success rates -- and therefore the
tail statistics -- are estimable at all. A single greedy rollout per prompt
gives a mean and nothing else.

    python scripts/evaluate.py --model runs/grpo_cvar/final --k 8 --out runs/grpo_cvar/eval

Generation is batched across all prompts via vLLM (n=k samples per prompt in
one call) rather than one prompt at a time -- see build_pairs.py for why.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from rsp.metrics import evaluate_rollouts
from rsp.rewards import gsm8k_reward
from rsp.checkpoint_utils import resolved_checkpoint

SYSTEM_PROMPT = ("Solve the problem. Reason step by step, then give the final "
                 "numeric answer inside <answer></answer> tags.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True)
        for row in ds
    ]

    # vLLM needs a full checkpoint (config.json + full weights); trainer
    # checkpoints here are LoRA-only, so merge onto the base model first --
    # see rsp/checkpoint_utils.py. The merged copy (~3GB) only needs to exist for
    # LLM() to load it; release it right after instead of for the whole run,
    # or five arms' worth of these plus build_pairs.py is tight on a 16GB
    # disk.
    with resolved_checkpoint(args.model) as vllm_model_path:
        # bf16 needs Ampere+ (sm_80+); native on the rented RTX 4090 (Ada,
        # sm_89) this is tuned for. Not valid on Kaggle's T4 (Turing,
        # sm_75) -- use float16 there.
        llm = LLM(model=vllm_model_path, dtype="bfloat16", seed=args.seed)
    sampling = SamplingParams(n=args.k, temperature=args.temperature,
                               max_tokens=args.max_new_tokens)
    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    correct = np.zeros((len(ds), args.k), dtype=bool)
    fmt_ok = np.zeros((len(ds), args.k), dtype=bool)
    records = []
    for i, (row, out) in enumerate(zip(ds, outputs)):
        texts = [o.text for o in out.outputs]
        for j, t in enumerate(texts):
            b = gsm8k_reward(t, row["answer"])
            correct[i, j], fmt_ok[i, j] = b.correct, b.format_ok
        records.append({"index": i, "correct": correct[i].tolist(),
                        "format_ok": fmt_ok[i].tolist()})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(ds)}", file=sys.stderr)

    outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / "rollouts.npz", correct=correct, format_ok=fmt_ok)
    (outdir / "per_prompt.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8")

    rep = evaluate_rollouts(correct, format_ok=fmt_ok, seed=args.seed)
    (outdir / "report.json").write_text(rep.to_json(), encoding="utf-8")
    print(rep.summary_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
