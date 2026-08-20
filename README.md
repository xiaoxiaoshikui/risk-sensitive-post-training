# Risk-Sensitive Post-Training

**Does GRPO's mean baseline hide a tail?**

GRPO computes advantages against the group mean, which makes it an optimizer
for *expected* reward. That is the right objective if you care about average
behaviour. It is the wrong one if you care about how bad a model gets on its
worst prompts — a policy can raise its mean while leaving, or worsening, a
tail of systematic failures.

This repository swaps GRPO's group baseline for risk-sensitive alternatives
and measures what changes, using an evaluation protocol that reports the tail
of the per-prompt distribution rather than collapsing it into a single number.

The motivation comes from a benchmark result: in
[CrosswalkBench-Global](https://github.com/xiaoxiaoshikui/crosswalkbench-global),
a method's aggregate lead turned out to vanish on the subset with the highest
deployment stakes. Aggregate scores hid tail failures at *evaluation* time.
The natural follow-up is whether the same blind spot exists at *training*
time, and whether the optimizer can be pointed at the tail directly.

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

> Not yet filled in. Numbers appear here after the runs below complete;
> nothing in this section is claimed until then.

| Run | mean acc | worst decile | zero-solve | fmt viol |
|---|---|---|---|---|
| SFT | — | — | — | — |
| DPO | — | — | — | — |
| GRPO `mean` | — | — | — | — |
| GRPO `cvar` α=0.25 | — | — | — | — |
| GRPO `entropic` β=2.0 | — | — | — | — |

The hypothesis under test is that `cvar` trades mean accuracy for worst-decile
accuracy. **A null result is a real result here** and will be reported as one:
if the tail does not move, that is evidence that GRPO's mean baseline is not
in fact leaving tail performance on the table at this scale, which is worth
knowing.

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
  python scripts/evaluate.py --model runs/$r/final --k 8 --limit 200 --out runs/$r/eval
done

python scripts/compare.py runs/grpo_mean/eval runs/grpo_cvar/eval
```

`k=8` rollouts per prompt is a floor, not a preference: per-prompt success
rates are not estimable from one greedy sample, so tail statistics are
undefined without it.

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
different tails must produce different `worst_decile`.

## Scope and limits

Single model family, single task, LoRA rather than full fine-tuning, one seed
per arm unless stated. This is a controlled comparison at small scale, not a
scaling study. Verifiable rewards only — no learned or LLM judge, since a
judge introduces its own tail failures and this is trying to measure tail
behaviour induced by the optimizer.

## License

MIT.
