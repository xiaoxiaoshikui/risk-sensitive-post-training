# Risk-Sensitive Post-Training

**Does GRPO's mean baseline hide a tail — and does fixing it survive contact with real training?**

GRPO computes advantages against the group mean, which makes it an optimizer
for *expected* reward. That is the right objective if you care about average
behaviour. It is the wrong one if you care about how bad a model gets on its
worst prompts — a policy can raise its mean while leaving, or worsening, a
tail of systematic failures. This repository swaps GRPO's group baseline for
two risk-sensitive alternatives (CVaR, exponential/entropic) and runs a
controlled, five-arm comparison — SFT → DPO → GRPO×3 — to find out whether
that actually moves the tail, on Qwen2.5-1.5B / GSM8K.

**TL;DR**

- Byte-identical GRPO configs across all three risk arms — only the advantage
  estimator differs — so any measured difference is attributable to that one
  variable, not incidental config drift.
- Full test coverage of the risk math (`tests/test_core.py`) *and* a
  regression test that diffs the project's hand-maintained TRL internals
  patch against whatever TRL is actually installed
  (`tests/test_trl_patch_sync.py`), so a future TRL upgrade fails loudly
  instead of silently running plain GRPO under a risk-sensitive label.
- Caught a real bug via its statistical fingerprint before trusting the
  results: a checkpoint-provenance bug was silently re-basing every
  post-SFT model onto the raw pretrained weights, and a paired-bootstrap
  "significant" result on the headline metric turned out to be a k=8
  sampling artifact once re-measured at k=64. Both are written up below —
  see [What broke, and how it was caught](#what-broke-and-how-it-was-caught).
- vLLM-accelerated generation throughout (rollouts, preference-pair building,
  evaluation) after profiling showed the naive path idle at ~27% GPU
  utilization; 5–20× faster wall-clock per stage as a result.
- Multi-seed replication of the core comparison, because a single training
  run is not evidence.

The motivation comes from a benchmark result: in
[CrosswalkBench-Global](https://github.com/xiaoxiaoshikui/crosswalkbench-global),
a method's aggregate lead turned out to vanish on the subset with the highest
deployment stakes. Aggregate scores hid tail failures at *evaluation* time.
The natural follow-up is whether the same blind spot exists at *training*
time, and whether the optimizer can be pointed at the tail directly.

**Stack:** PyTorch · TRL (GRPO/DPO/SFT trainers) · PEFT/LoRA · vLLM ·
`accelerate` · numpy/pytest for the statistics.

## The estimators

For a group of `G` rollouts from one prompt:

| Estimator | Baseline | Objective |
|---|---|---|
| `mean` | group mean | expected reward — standard GRPO |
| `cvar` | α-quantile, gradient supported on the tail | CVaR<sub>α</sub> — Rockafellar–Uryasev form |
| `entropic` | risk-tilted mean, `exp(-βr)` weights | exponential utility, smooth |

`cvar` sets `A_i = 1{r_i ≤ q_α} · (r_i − q_α) / α`. Pure CVaR discards `(1−α)`
of every group, which is a substantial variance increase at small `G`, so
`lambda_risk` interpolates between the mean and tail estimators and can be
annealed if a run destabilises. `entropic` uses every rollout and recovers the
mean estimator as `β → 0`.

Implementation: [`rsp/risk.py`](rsp/risk.py). Group degeneracy — every rollout
scoring identically — returns zero advantage under all estimators rather than
large normalized noise, which is a well-known way to wreck a GRPO run late in
training.

## What is actually controlled

`configs/grpo_mean.yaml` and `configs/grpo_cvar.yaml` are byte-identical
except for the `risk` block: same model, dataset, group size, learning rate,
KL coefficient, temperature and seed. Only the advantage function differs.

The TRL advantage hook is patched defensively and **raises rather than falling
back** if the method name has moved between releases. A silent fallback would
make both arms standard GRPO while still producing plausible-looking numbers —
the failure mode most likely to produce a confidently wrong result.

## Evaluation

Reported per prompt first, aggregated second
([`rsp/metrics.py`](rsp/metrics.py)):

- `mean_acc` — the usual headline
- `cvar_acc`, `worst_decile` — mean success over the worst α-fraction of prompts
- `zero_solve_rate` — prompts no rollout solved, the hard tail
- `format_violation_rate` — tracked separately from incorrectness, because a
  policy degenerating into malformed output is a different failure from one
  that is merely wrong
- bootstrap CIs resampled over **prompts**, not rollouts, since prompts are the
  independent unit and resampling rollouts would understate uncertainty for
  exactly these tail statistics

`scripts/compare.py` runs a paired bootstrap between two runs on the same
prompt set.

## Results

Qwen2.5-1.5B-Instruct, GSM8K, 200 held-out test prompts, `k=64` rollouts per
prompt (see [why k=64 and not k=8](#what-broke-and-how-it-was-caught) below).
Single seed so far; a 3-seed replication of the three GRPO arms is in
progress to confirm the pattern holds — this table gets updated when it
finishes, not rewritten.

| Run | mean acc | cvar acc (α=0.25) | worst decile | zero-solve | fmt viol |
|---|---|---|---|---|---|
| SFT | 0.470 | 0.093 | 0.030 | 0.020 | 0.013 |
| DPO | 0.469 | 0.096 | 0.034 | 0.030 | 0.011 |
| GRPO `mean` | 0.469 | 0.098 | 0.030 | 0.025 | 0.013 |
| GRPO `cvar` α=0.25 | 0.467 | 0.096 | 0.034 | 0.020 | 0.014 |
| GRPO `entropic` β=2.0 | 0.469 | 0.095 | 0.030 | 0.030 | 0.013 |

Paired bootstrap vs. the `mean` control, resampled over prompts:

| | mean acc Δ | p | worst-decile Δ | p |
|---|---|---|---|---|
| `cvar` vs `mean` | −0.002 | 0.60 | +0.005 | 0.48 |
| `entropic` vs `mean` | +0.001 | 0.79 | 0.000 | 0.97 |

**Reading this honestly: this is a null result**, and it's reported as one.
Neither risk-sensitive estimator moved mean accuracy or worst-decile accuracy
by a statistically detectable amount relative to GRPO's mean baseline, at
this model scale (1.5B), this group size (`G=8`), and this task (GSM8K). That
is itself informative — it says the mean baseline is not obviously leaving
tail performance on the table here, which narrows where a real effect (if any
exists) would have to come from: a larger `G` so hard-but-not-impossible
prompts stop producing degenerate all-zero training groups, a harder task
with more headroom in the tail, or a bigger base model. See
[Scope and limits](#scope-and-limits).

## What broke, and how it was caught

Two findings worth a technical read, because the process matters as much as
the number:

**A checkpoint-provenance bug silently dropped the SFT stage from every
downstream model.** `trainer.save_model()` on a PEFT model saves only the
LoRA adapter, so training DPO/GRPO on top of `runs/sft/final` requires
merging that adapter onto its base first — and when the trainer then saves
*its own* adapter, it records whatever path it was built from as that new
adapter's declared base. That path was a temporary merged-checkpoint
directory, already deleted by the time anything looked at it again. The fix
naively pointed the recorded base at `runs/sft/final`'s *own* base
(`Qwen/Qwen2.5-1.5B-Instruct`) instead of at `runs/sft/final` itself — which
is wrong in a way that doesn't error, it just silently skips the SFT stage on
every future load. First full evaluation run surfaced it immediately:
`format_violation_rate` jumped from SFT's ~1% to ~61% on every downstream
checkpoint, because generation was happening against the barely-tuned raw
base, not the SFT policy. Root-caused from the symptom, fixed by resolving
adapter chains recursively and repointing each checkpoint at its *immediate*
parent (`rsp/checkpoint_utils.py`), re-verified by inspecting raw completions
before re-trusting any eval number.

**A "statistically significant" result turned out to be a measurement-floor
artifact.** With `k=8` rollouts/prompt, `worst_decile` (mean success on the
worst 10% of prompts) came out to *exactly* 0.0 for all five arms, and a
paired bootstrap showed `cvar`/`entropic` beating the mean baseline on
overall accuracy at `p<0.05`. Rather than report that, the worst-decile
prompts were isolated and resampled at `k=64` — 15 of 17 turned out solvable
1.5–22% of the time, not impossible; `k=8` was just too few draws to reliably
surface a low-probability success. Re-running the full evaluation at `k=64`
did two things at once: `worst_decile` became measurable (no longer floored
at exactly zero) *and* the earlier "significant" accuracy gap collapsed to
`p≈0.6–0.8` — the original finding was `k=8` sampling noise, not signal. This
is also why GRPO training itself uses `G=8`: a prompt with a real 5–10%
success rate frequently produces an all-zero training group at that size,
and an all-zero group gives *every* estimator here (mean, CVaR, entropic)
identically zero advantage — exactly the prompts CVaR is meant to help with
are the ones most likely to be statistically invisible to it at this group
size. `k=64`/larger-`G` numbers above reflect this fix.

Smaller things caught along the way, each with a regression test or a pinned
version behind it now: an unbounded `vllm` version resolved outside TRL's
declared-compatible range and dragged in a `torch` build mismatched with the
rest of the stack, crashing `import trl` entirely; `AutoModelForCausalLM`
silently auto-resolves a bare LoRA-adapter directory while vLLM and TRL's own
trainers don't, which needed an explicit merge step; and the ~600-line
hand-patched copy of TRL's GRPO internals ([`rsp/_trl_patches/`](rsp/_trl_patches))
now has a test that diffs it against whatever TRL is actually installed, so
drift is a loud CI failure instead of a quiet wrong gradient.

## Running it

Single GPU, LoRA throughout. Qwen2.5-1.5B-Instruct on GSM8K; roughly a few
hours per arm on one 4090 or A100.

```bash
pip install -r requirements.txt

python scripts/train_sft.py   --config configs/sft.yaml
python scripts/build_pairs.py --config configs/dpo.yaml
python scripts/train_dpo.py   --config configs/dpo.yaml
python scripts/train_grpo.py  --config configs/grpo_mean.yaml
python scripts/train_grpo.py  --config configs/grpo_cvar.yaml
python scripts/train_grpo.py  --config configs/grpo_entropic.yaml

for r in sft dpo grpo_mean grpo_cvar grpo_entropic; do
  python scripts/evaluate.py --model runs/$r/final --k 64 --limit 200 --out runs/$r/eval
done

python scripts/compare.py runs/grpo_mean/eval runs/grpo_cvar/eval
```

`k=8` rollouts per prompt is the floor for *training* group size (`G` in the
estimator table above); per-prompt success rates aren't estimable from one
greedy sample, so tail statistics are undefined without sampling a group at
all. For *evaluation* specifically, use the largest `k` you can afford —
`k=64` above, not `k=8` — for the reason in
[What broke, and how it was caught](#what-broke-and-how-it-was-caught).

On a rented single-GPU box (vast.ai, RunPod), [`scripts/run_all.sh`](scripts/run_all.sh)
runs the whole sequence unattended -- start it inside `tmux` so an SSH drop
doesn't kill an hours-long job.

### Free GPU (Kaggle)

A Kaggle GPU notebook (T4, free, 30 GPU-hours/week, ~9-12h per session) is
enough for this at no cost, but one session isn't long enough for the whole
pipeline, so [`scripts/kaggle/`](scripts/kaggle) splits it into stages that
each pick up where the last left off:

| Script | Produces |
|---|---|
| `stage1_setup_sft.sh` | deps, sanity checks, `runs/sft/final` |
| `stage2_dpo.sh` | `runs/dpo/final` |
| `stage3_grpo.sh mean` / `cvar` / `entropic` | one `runs/grpo_<arm>/final` per call |
| `stage4_eval_compare.sh` | `runs/*/eval/report.json`, `runs/compare_*.json` |

Kaggle wipes `/kaggle/working` between sessions, so `runs/` has to be carried
forward by hand:

1. New notebook -> Settings -> Accelerator: GPU T4 x2, Internet: on.
2. `!git clone https://github.com/xiaoxiaoshikui/risk-sensitive-post-training && cd risk-sensitive-post-training`
3. Run the next stage script for this session (`!bash scripts/kaggle/stageN_....sh`).
4. Before the session ends: **Save Version -> Save & Run All (Commit)**. The
   `runs/` directory under `/kaggle/working` becomes that version's Output.
5. Next session: create/open a notebook, **Add Input -> Notebook Output
   Files** and pick the previous version, then copy its `runs/` back in
   before running the next stage, e.g.
   `!cp -r /kaggle/input/<prev-notebook>/runs .`
6. Repeat per stage. Each `stage3_grpo.sh <arm>` call is its own session so a
   timeout mid-arm only costs that one arm, not the other two.

Kaggle's GPU accelerator is two T4s, not one, but a plain `python
scripts/train_*.py` only ever drives GPU0. The stage scripts instead launch
training with `accelerate launch` (flags sized to the detected device count
by `accel_launch_flags` in `scripts/kaggle/_common.sh`), which runs standard
data-parallel DDP across both GPUs when two are available and falls back to
single-GPU cleanly when only one is. No code changes needed beyond the launch
command -- SFTTrainer/DPOTrainer/GRPOTrainer are already distributed-aware,
and the GRPO risk patch ([`rsp/_trl_patches/`](rsp/_trl_patches)) only touches
the already-gathered, cross-process reward tensor, so it's correct under DDP
unmodified.

Each stage script skips work whose output already exists, so re-running a
stage after copying `runs/` back in is safe.

## Tests

```bash
python -m pytest tests -q
```

Covers the risk functionals, the estimator interpolation limits (`cvar` at
`lambda_risk=0` and `entropic` at `β→0` must both reproduce `mean` exactly),
the group-size constraint `α ≥ 1/G`, reward parsing, and the property that
motivates the whole repository — two policies with identical mean accuracy and
different tails must produce different `worst_decile`. A separate test
([`tests/test_trl_patch_sync.py`](tests/test_trl_patch_sync.py)) diffs
`rsp/_trl_patches/grpo_1_10_0.py` against whatever TRL is actually installed
and fails if anything outside the marked patch block has drifted.

## Scope and limits

Single model family, single task, LoRA rather than full fine-tuning, small
group size (`G=8`, both for training and — until the fix above — evaluation).
This is a controlled comparison at small scale, not a scaling study.
Verifiable rewards only — no learned or LLM judge, since a judge introduces
its own tail failures and this is trying to measure tail behaviour induced by
the optimizer. The null result above is scoped accordingly: it says the mean
baseline doesn't obviously leave tail performance on the table *at this
scale*, not that risk-sensitive RL objectives don't matter anywhere.

## License

MIT.
