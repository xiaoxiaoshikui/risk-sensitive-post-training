"""
Tail-aware evaluation for post-trained policies.

A post-training run is normally reported as a single mean accuracy. That
number cannot distinguish a policy that improved uniformly from one that got
better on prompts it already handled while leaving a tail of systematic
failures untouched -- and the second is the outcome that matters when the
model is deployed.

Every metric here is reported per prompt first and aggregated second, so the
distribution over prompts stays visible.

Metrics
-------
mean_acc        mean per-prompt success rate
cvar_acc        mean success rate over the worst alpha-fraction of prompts
worst_decile    cvar_acc at alpha = 0.1, reported separately as the headline
                tail number
pass_at_k       fraction of prompts solved by at least one of k rollouts
solve_rate_std  spread of per-prompt success, a cheap reliability proxy
format_violation_rate  rollouts failing the output contract
zero_solve_rate fraction of prompts no rollout solved -- the hard tail
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

__all__ = ["TailReport", "evaluate_rollouts", "bootstrap_ci", "paired_bootstrap"]


@dataclass
class TailReport:
    n_prompts: int
    n_rollouts_per_prompt: int
    mean_acc: float
    cvar_acc: float
    worst_decile: float
    pass_at_k: float
    solve_rate_std: float
    zero_solve_rate: float
    format_violation_rate: float
    mean_acc_ci95: tuple[float, float]
    worst_decile_ci95: tuple[float, float]
    alpha: float

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in d.items()}, indent=2)

    def summary_line(self) -> str:
        return (
            f"mean {self.mean_acc:.3f} "
            f"[{self.mean_acc_ci95[0]:.3f},{self.mean_acc_ci95[1]:.3f}]  "
            f"worst-decile {self.worst_decile:.3f} "
            f"[{self.worst_decile_ci95[0]:.3f},{self.worst_decile_ci95[1]:.3f}]  "
            f"zero-solve {self.zero_solve_rate:.3f}  "
            f"fmt-viol {self.format_violation_rate:.3f}"
        )


def _cvar_lower(x: np.ndarray, alpha: float) -> float:
    """Mean of the worst alpha-fraction. Always includes at least one item."""
    if x.size == 0:
        return 0.0
    k = max(1, int(np.floor(alpha * x.size)))
    return float(np.sort(x)[:k].mean())


def bootstrap_ci(
    values: np.ndarray,
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap over prompts.

    Resampling prompts rather than rollouts is deliberate: prompts are the
    independent unit, and resampling rollouts would understate uncertainty
    for exactly the tail statistics this module exists to report.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        return (0.0, 0.0)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = statistic(values[idx])
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def evaluate_rollouts(
    correct: np.ndarray,
    *,
    format_ok: np.ndarray | None = None,
    alpha: float = 0.25,
    seed: int = 0,
) -> TailReport:
    """Score a policy from its rollout outcomes.

    correct:   (n_prompts, k) boolean or 0/1 -- did rollout j solve prompt i
    format_ok: (n_prompts, k) boolean -- did rollout j satisfy the output
               contract. Format violations are tracked separately from
               incorrectness because they are a different failure and a
               policy that collapses into malformed output can otherwise
               look merely inaccurate.
    """
    c = np.asarray(correct, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"expected (n_prompts, k), got shape {c.shape}")
    n_prompts, k = c.shape

    per_prompt = c.mean(axis=1)

    if format_ok is None:
        fmt_viol = 0.0
    else:
        f = np.asarray(format_ok, dtype=np.float64)
        if f.shape != c.shape:
            raise ValueError("format_ok must match correct in shape")
        fmt_viol = float(1.0 - f.mean())

    return TailReport(
        n_prompts=n_prompts,
        n_rollouts_per_prompt=k,
        mean_acc=float(per_prompt.mean()),
        cvar_acc=_cvar_lower(per_prompt, alpha),
        worst_decile=_cvar_lower(per_prompt, 0.10),
        pass_at_k=float((c.sum(axis=1) > 0).mean()),
        solve_rate_std=float(per_prompt.std()),
        zero_solve_rate=float((c.sum(axis=1) == 0).mean()),
        format_violation_rate=fmt_viol,
        mean_acc_ci95=bootstrap_ci(per_prompt, np.mean, seed=seed),
        worst_decile_ci95=bootstrap_ci(
            per_prompt, lambda v: _cvar_lower(v, 0.10), seed=seed
        ),
        alpha=alpha,
    )


def paired_bootstrap(
    per_prompt_a: np.ndarray,
    per_prompt_b: np.ndarray,
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap on the difference statistic(b) - statistic(a).

    Paired on prompt index, since both policies are evaluated on the same
    prompts and treating the runs as independent would inflate the interval.
    """
    a = np.asarray(per_prompt_a, dtype=np.float64)
    b = np.asarray(per_prompt_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires matching prompt sets")
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = statistic(b[idx]) - statistic(a[idx])
    point = statistic(b) - statistic(a)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # Two-sided bootstrap p-value: how often the resampled difference crosses zero.
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {"delta": float(point), "ci95_lo": lo, "ci95_hi": hi, "p_value": min(1.0, p)}
