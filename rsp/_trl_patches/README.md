# Why this is a full-method patch

`scripts/train_grpo.py` needs to replace GRPO's mean/std advantage baseline
with `rsp.risk.batch_advantages`. The original plan was to override a small,
dedicated method for that -- TRL's own `GRPOTrainer` docs and older forks
suggested one existed under a name like `_compute_advantages` or
`compute_advantages`.

That method does not exist in any TRL release with a public `GRPOTrainer`.
Checked directly against the installed source across `0.15.0` through the
pinned `1.10.0`: advantage computation has always been inlined inside
`_generate_and_score_completions`, a ~600-line method that also handles
rollout generation, tool calls, multimodal inputs, vLLM importance sampling,
and logging. There is no smaller point to override without either
reimplementing all of that surrounding machinery or risking silently
diverging from it.

## What `grpo_1_10_0.py` is

A copy of `GRPOTrainer._generate_and_score_completions` from `trl==1.10.0`
(see `requirements.txt` -- **trl is pinned to this exact version**; this
patch's correctness depends on line-for-line matching that source) with one
block replaced: the `sum_then_normalize` branch's
`advantages = (rewards - mean_grouped_rewards) / (std_rewards + eps)`
becomes a call into `rsp.risk.batch_advantages`. Everything else --
generation, reward calculation, NaN handling, logging, metrics -- is
untouched, so the comparison between estimators stays exactly what the
README claims: same rollouts, same rewards, same optimizer, only the
advantage function differs.

`patch_advantages()` in `scripts/train_grpo.py` binds this function onto the
trainer instance with the *original* `trl.trainer.grpo_trainer` module as its
`__globals__` (via `types.FunctionType`, not `exec`), so every name the copied
code relies on (`torch`, `nanstd`, `gather_object`, ...) resolves against
TRL's live module state instead of being re-imported by hand and risking
drift.

## If `trl` is upgraded

This patch will not automatically track a new TRL version. Bumping the pin in
`requirements.txt` requires re-diffing
`GRPOTrainer._generate_and_score_completions` against this file and
re-applying the same substitution -- `patch_advantages()` checks
`trl.__version__` at setup time and raises rather than silently applying a
stale patch to a mismatched version.
