#!/usr/bin/env python3
"""DPO on self-generated preference pairs.

Pairs are built from the SFT policy's own rollouts: a correct completion is
chosen over an incorrect one for the same prompt. This keeps the preference
data on-policy and, unlike a human or LLM-judged preference set, makes the
preference signal verifiable -- the same property the GRPO rewards have, so
DPO and GRPO stay comparable.

    python scripts/build_pairs.py --config configs/dpo.yaml
    python scripts/train_dpo.py  --config configs/dpo.yaml
"""
from __future__ import annotations
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    conf = yaml.safe_load(pathlib.Path(args.config).read_text())

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    ds = load_dataset("json", data_files=conf["pairs_file"], split="train")
    trainer = DPOTrainer(
        model=conf["model"],
        args=DPOConfig(output_dir=conf["output_dir"], **conf["training"]),
        train_dataset=ds,
        peft_config=LoraConfig(**conf["lora"]) if conf.get("lora") else None,
    )
    trainer.train()
    trainer.save_model(conf["output_dir"] + "/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
