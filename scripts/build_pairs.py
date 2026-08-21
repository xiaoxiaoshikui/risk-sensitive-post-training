#!/usr/bin/env python3
"""Build on-policy DPO preference pairs from the SFT policy's own rollouts.

For each prompt, samples k rollouts from the SFT model and pairs one correct
completion against one incorrect completion for the same prompt. Prompts
where the SFT policy is all-correct or all-incorrect are skipped -- no
contrastive pair exists there. This keeps the preference signal verifiable
rather than judged, the same property the GRPO rewards have, so DPO and GRPO
stay comparable.

    python scripts/build_pairs.py --config configs/dpo.yaml

Generation is batched across prompts via vLLM (n=k samples per prompt in one
call), not one prompt at a time -- plain HF generate() one-prompt-at-a-time
left the GPU mostly idle (~4h for 2000 prompts on a 4090). Progress is still
checkpointed every --chunk-size prompts so a cut-off run resumes instead of
restarting.
"""
from __future__ import annotations
import argparse, json, pathlib, random, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rsp.rewards import gsm8k_reward
from rsp.checkpoint_utils import resolved_checkpoint

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
    ap.add_argument("--chunk-size", type=int, default=200,
                     help="prompts per vLLM generate() call / progress checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import yaml
    conf = yaml.safe_load(pathlib.Path(args.config).read_text())
    model_path = args.sft_model or conf["model"]
    out_path = pathlib.Path(conf["pairs_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    ds = ds.select(range(min(args.limit, len(ds))))

    # Resumable: a session can be cut off long before `limit` prompts are
    # done. Progress is checkpointed every chunk, not batched at the end, so
    # a resumed run picks up right after the last completed chunk instead of
    # losing everything generated so far.
    progress_path = out_path.with_suffix(out_path.suffix + ".progress")
    start_idx = int(progress_path.read_text()) if progress_path.exists() else 0
    if start_idx:
        print(f"[rsp] resuming from prompt {start_idx} ({out_path} already has partial output)",
              file=sys.stderr)
    else:
        out_path.write_text("", encoding="utf-8")  # fresh start: truncate any stale file

    tok = AutoTokenizer.from_pretrained(model_path)
    rng = random.Random(args.seed)
    n_pairs = sum(1 for _ in open(out_path, encoding="utf-8")) if start_idx else 0

    # vLLM needs a full checkpoint (config.json + full weights); trainer
    # checkpoints here are LoRA-only, so merge onto the base model first --
    # see rsp/checkpoint_utils.py. The merged copy (~3GB) only needs to exist for
    # LLM() to load it; release it right after instead of for the whole run,
    # or five arms' worth of eval merges plus this one is tight on a 16GB
    # disk.
    with resolved_checkpoint(model_path) as vllm_model_path:
        # bf16 needs Ampere+ (sm_80+); native on the rented RTX 4090 (Ada,
        # sm_89) this is tuned for. Not valid on Kaggle's T4 (Turing,
        # sm_75) -- use float16 there.
        llm = LLM(model=vllm_model_path, dtype="bfloat16", seed=args.seed)
    sampling = SamplingParams(n=args.k, temperature=args.temperature,
                               max_tokens=args.max_new_tokens)

    with open(out_path, "a", encoding="utf-8") as f:
        for chunk_start in range(start_idx, len(ds), args.chunk_size):
            chunk = ds.select(range(chunk_start, min(chunk_start + args.chunk_size, len(ds))))
            messages_per_row = [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": row["question"]}]
                for row in chunk
            ]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages_per_row]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)

            for messages, row, out in zip(messages_per_row, chunk, outputs):
                texts = [o.text for o in out.outputs]
                correct = [t for t in texts if gsm8k_reward(t, row["answer"]).correct]
                incorrect = [t for t in texts if not gsm8k_reward(t, row["answer"]).correct]
                if correct and incorrect:
                    pair = {
                        "prompt": messages,
                        "chosen": [{"role": "assistant", "content": rng.choice(correct)}],
                        "rejected": [{"role": "assistant", "content": rng.choice(incorrect)}],
                    }
                    f.write(json.dumps(pair) + "\n")
                    n_pairs += 1
            f.flush()

            done = min(chunk_start + args.chunk_size, len(ds))
            progress_path.write_text(str(done), encoding="utf-8")
            print(f"  {done}/{len(ds)} prompts, {n_pairs} pairs so far", file=sys.stderr)

    progress_path.unlink(missing_ok=True)
    print(f"[rsp] wrote {n_pairs} pairs -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
