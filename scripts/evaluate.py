#!/usr/bin/env python3
"""Evaluate a checkpoint with tail-aware metrics.

Samples k rollouts per prompt so per-prompt success rates -- and therefore the
tail statistics -- are estimable at all. A single greedy rollout per prompt
gives a mean and nothing else.

    python scripts/evaluate.py --model runs/grpo_cvar/final --k 8 --out runs/grpo_cvar/eval
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from rsp.metrics import evaluate_rollouts
from rsp.rewards import gsm8k_reward

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

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        # T4 (Turing, sm_75) has no bf16 tensor cores; fp16 is native there.
        args.model, torch_dtype=torch.float16, device_map="auto").eval()
    torch.manual_seed(args.seed)

    correct = np.zeros((len(ds), args.k), dtype=bool)
    fmt_ok = np.zeros((len(ds), args.k), dtype=bool)
    records = []

    for i, row in enumerate(ds):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": row["question"]}],
            tokenize=False, add_generation_prompt=True)
        enc = tok([prompt] * args.k, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        texts = [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                 for s in gen]
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
