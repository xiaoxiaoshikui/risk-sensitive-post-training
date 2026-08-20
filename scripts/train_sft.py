#!/usr/bin/env python3
"""SFT baseline. Establishes the output contract before any RL runs.

Without an SFT stage the base model rarely emits well-formed <answer> tags,
format violations dominate the reward signal, and the risk-estimator
comparison ends up measuring formatting rather than reasoning.

    python scripts/train_sft.py --config configs/sft.yaml
"""
from __future__ import annotations
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import yaml

SYSTEM_PROMPT = ("Solve the problem. Reason step by step, then give the final "
                 "numeric answer inside <answer></answer> tags.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    conf = yaml.safe_load(pathlib.Path(args.config).read_text())

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    ds = load_dataset(conf["dataset"], conf.get("dataset_config"), split=conf["split"])
    if conf.get("max_samples"):
        ds = ds.select(range(min(conf["max_samples"], len(ds))))

    def to_chat(r):
        sol = r[conf["answer_field"]]
        body, _, final = sol.rpartition("####")
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r[conf["question_field"]]},
            {"role": "assistant",
             "content": f"{body.strip()}\n<answer>{final.strip()}</answer>"},
        ]}

    ds = ds.map(to_chat, remove_columns=ds.column_names)
    trainer = SFTTrainer(
        model=conf["model"],
        args=SFTConfig(output_dir=conf["output_dir"], **conf["training"]),
        train_dataset=ds,
        peft_config=LoraConfig(**conf["lora"]) if conf.get("lora") else None,
    )
    trainer.train()
    trainer.save_model(conf["output_dir"] + "/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
