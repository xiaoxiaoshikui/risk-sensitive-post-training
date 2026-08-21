#!/usr/bin/env python3
"""Build on-policy DPO preference pairs from the SFT policy's own rollouts.

For each prompt, samples k rollouts from the SFT model and pairs one correct
completion against one incorrect completion for the same prompt. Prompts
where the SFT policy is all-correct or all-incorrect are skipped -- no
contrastive pair exists there. This keeps the preference signal verifiable
rather than judged, the same property the GRPO rewards have, so DPO and GRPO
stay comparable.

    python scripts/build_pairs.py --config configs/dpo.yaml
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rsp.rewards import gsm8k_reward

SYSTEM_PROMPT = ("Solve the problem. Reason step by step, then give the final "
                 "numeric answer inside <answer></answer> tags.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft-model", default=None,
                     help="override the SFT checkpoint path (default: configs/dpo.yaml's model field)")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--dataset-config", default="main")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import yaml
    conf = yaml.safe_load(pathlib.Path(args.config).read_text())
    model_path = args.sft_model or conf["model"]
    out_path = pathlib.Path(conf["pairs_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))

    # Resumable: a session can be cut off (Kaggle sessions have a wall-clock
    # cap) long before `limit` prompts are done. Progress is written after
    # every prompt, not batched at the end, so a resumed run picks up right
    # after the last completed prompt instead of losing everything generated
    # so far.
    progress_path = out_path.with_suffix(out_path.suffix + ".progress")
    start_idx = int(progress_path.read_text()) if progress_path.exists() else 0
    if start_idx:
        print(f"[rsp] resuming from prompt {start_idx} ({out_path} already has partial output)",
              file=sys.stderr)
    else:
        out_path.write_text("", encoding="utf-8")  # fresh start: truncate any stale file

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # T4 (Turing, sm_75) has no bf16 tensor cores; fp16 is native there.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto").eval()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    n_pairs = sum(1 for _ in open(out_path, encoding="utf-8")) if start_idx else 0
    remaining = ds.select(range(start_idx, len(ds)))
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(remaining, start=start_idx):
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["question"]}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok([prompt] * args.k, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                      max_new_tokens=args.max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
            texts = [tok.decode(s[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                     for s in gen]

            correct = [t for t in texts if gsm8k_reward(t, row["answer"]).correct]
            incorrect = [t for t in texts if not gsm8k_reward(t, row["answer"]).correct]
            if correct and incorrect:
                pair = {
                    "prompt": messages,
                    "chosen": [{"role": "assistant", "content": rng.choice(correct)}],
                    "rejected": [{"role": "assistant", "content": rng.choice(incorrect)}],
                }
                f.write(json.dumps(pair) + "\n")
                f.flush()
                n_pairs += 1

            progress_path.write_text(str(i + 1), encoding="utf-8")
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(ds)} prompts, {n_pairs} pairs so far", file=sys.stderr)

    progress_path.unlink(missing_ok=True)
    print(f"[rsp] wrote {n_pairs} pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
