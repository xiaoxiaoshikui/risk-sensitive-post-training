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
from rsp.checkpoint_utils import fix_saved_base_model_path, resolved_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    conf = yaml.safe_load(pathlib.Path(args.config).read_text())

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    ds = load_dataset("json", data_files=conf["pairs_file"], split="train")
    # DPOTrainer resolves model=<path> via AutoConfig.from_pretrained, which
    # can't load a bare LoRA adapter directory (runs/sft/final has no
    # config.json) the way AutoModelForCausalLM.from_pretrained can -- merge
    # onto the base model first. See rsp/checkpoint_utils.py.
    with resolved_checkpoint(conf["model"]) as model_path:
        trainer = DPOTrainer(
            model=model_path,
            args=DPOConfig(output_dir=conf["output_dir"], **conf["training"]),
            train_dataset=ds,
            peft_config=LoraConfig(**conf["lora"]) if conf.get("lora") else None,
        )
    trainer.train()
    final_dir = conf["output_dir"] + "/final"
    trainer.save_model(final_dir)
    # save_model() recorded the merged temp dir (already deleted) as this
    # adapter's base -- repoint it at its immediate parent checkpoint, *not*
    # that parent's own base (which would silently drop the SFT stage from
    # every future load). See fix_saved_base_model_path.
    fix_saved_base_model_path(final_dir, conf["model"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
