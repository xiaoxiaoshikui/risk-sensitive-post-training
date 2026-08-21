"""Shared helper for pointing a loader at a possibly-LoRA-only checkpoint.

`trainer.save_model()` on a PEFT model saves only the adapter
(`adapter_config.json` + adapter weights, no `config.json`), so every
checkpoint this repo produces from an earlier stage (runs/sft/final, etc.) is
LoRA-only. Several loaders that later stages hand such a path to expect a
full checkpoint and don't know how to resolve a bare adapter directory on
their own:

- vLLM's `LLM(model=...)` -- unlike `AutoModelForCausalLM.from_pretrained`,
  it has no fallback for a bare adapter directory.
- trl's `DPOTrainer`/`GRPOTrainer` resolve `model=<path>` via
  `AutoConfig.from_pretrained` first, which raises ("Should have a
  `model_type` key in its config.json") on an adapter-only directory rather
  than detecting and loading the referenced base model the way
  `AutoModelForCausalLM.from_pretrained` does.

Both need the adapter pre-merged onto its base model.
"""
from __future__ import annotations

import contextlib
import gc
import json
import pathlib
import tempfile
from typing import Iterator


def _load_merged(model_path: str):
    """Load model_path as a plain (non-PEFT) model, recursively merging a
    chain of LoRA adapters onto their ultimate base if needed.

    AutoModelForCausalLM.from_pretrained has its own adapter-directory
    auto-loading, but it attaches the adapter rather than merging it -- the
    returned object is still PEFT-wrapped. Wrapping *that* in a second
    PeftModel.from_pretrained (for a checkpoint chained on top, e.g. GRPO on
    top of SFT) nests two PEFT wrappers, and merge_and_unload() on the outer
    one leaves the inner adapter still unmerged, which breaks save_pretrained
    later. Recursing explicitly and merging at every hop keeps each step
    working with a genuinely plain model.
    """
    import torch
    from transformers import AutoModelForCausalLM

    path = pathlib.Path(model_path)
    adapter_cfg = path / "adapter_config.json"
    if not adapter_cfg.exists():
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)

    from peft import PeftModel

    base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
    print(f"[rsp] merging LoRA adapter ({model_path}) onto base {base_name}...")
    base = _load_merged(base_name)
    return PeftModel.from_pretrained(base, model_path).merge_and_unload()


@contextlib.contextmanager
def resolved_checkpoint(model_path: str) -> Iterator[str]:
    """Yield a checkpoint path any of the above loaders can load directly.

    `model_path` unchanged if it's already a full checkpoint; otherwise a
    temporary directory holding the (possibly chained) LoRA adapter(s)
    merged onto their base model, deleted on exit (so several arms' worth of
    ~3GB merges don't pile up on a 16GB disk).
    """
    path = pathlib.Path(model_path)
    if not (path / "adapter_config.json").exists():
        yield model_path
        return

    import torch
    from transformers import AutoTokenizer

    merged = _load_merged(model_path)
    tok = AutoTokenizer.from_pretrained(model_path)

    with tempfile.TemporaryDirectory(prefix="rsp_merged_") as tmp:
        merged.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        del merged
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield tmp


def fix_saved_base_model_path(output_dir: str, parent_checkpoint: str) -> None:
    """Repoint a just-saved adapter's base_model_name_or_path.

    A trainer built from `resolved_checkpoint()`'s merged temp directory
    records *that* transient path as the adapter's base when it saves -- e.g.
    training on top of runs/sft/final produces an adapter_config.json
    pointing at a /tmp/rsp_merged_* directory that no longer exists once
    resolved_checkpoint()'s `with` block exits.

    Call this right after trainer.save_model() with `parent_checkpoint` set
    to conf["model"] itself (runs/sft/final), *not* runs/sft/final's own
    base_model_name_or_path (Qwen/Qwen2.5-1.5B-Instruct) -- the latter would
    point a DPO/GRPO adapter straight at the un-SFT-tuned base, silently
    dropping the SFT stage from every future load of this checkpoint.
    resolved_checkpoint()'s _load_merged() resolves a chain of adapters one
    hop at a time, so pointing each adapter at its immediate parent is
    correct even when that parent is itself a bare adapter directory.
    """
    cfg_path = pathlib.Path(output_dir) / "adapter_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = parent_checkpoint
    cfg_path.write_text(json.dumps(cfg, indent=2))
