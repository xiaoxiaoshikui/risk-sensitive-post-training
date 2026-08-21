"""Guards the one invariant rsp/_trl_patches/README.md depends on: that
grpo_1_10_0.py is a verbatim copy of trl's real
GRPOTrainer._generate_and_score_completions outside the marked rsp-patch
block. This has already broken once (see commit "Fix GRPO risk-advantage
patch: no TRL release exposes the assumed hook") -- catching drift here is
cheaper than catching it after a training run silently uses upstream's
mean/std baseline instead of the configured risk estimator.
"""
from __future__ import annotations

import difflib
import inspect
import textwrap

import pytest

trl = pytest.importorskip("trl")

from rsp._trl_patches.grpo_1_10_0 import _generate_and_score_completions as patched_fn

START_MARK = "# --- rsp patch start"
END_MARK = "# --- rsp patch end"


def test_patch_is_verbatim_outside_marked_block():
    from trl.trainer.grpo_trainer import GRPOTrainer

    live_src = textwrap.dedent(inspect.getsource(GRPOTrainer._generate_and_score_completions))
    patch_src = inspect.getsource(patched_fn)

    patch_lines = patch_src.splitlines()
    start = next(i for i, l in enumerate(patch_lines) if START_MARK in l)
    end = next(i for i, l in enumerate(patch_lines) if END_MARK in l)

    sm = difflib.SequenceMatcher(a=live_src.splitlines(), b=patch_lines, autojunk=False)
    stray = [
        (tag, j1, j2)
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal" and not (j1 >= start and j2 <= end + 1)
    ]
    assert not stray, (
        f"rsp/_trl_patches/grpo_1_10_0.py has diverged from the installed "
        f"trl=={trl.__version__} outside the rsp-patch block (lines {start + 1}-{end + 1}): "
        f"{stray}. Re-diff against the installed source and update the patch -- "
        f"see rsp/_trl_patches/README.md."
    )
