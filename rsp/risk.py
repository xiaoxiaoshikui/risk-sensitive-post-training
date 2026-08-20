"""
Risk-sensitive advantage estimation for group-relative policy optimization.

Standard GRPO computes, for a group of G rollouts sampled from one prompt,

    A_i = (r_i - mean(r)) / (std(r) + eps)

which is an unbiased-ish estimate of the advantage under an *expected reward*
objective. Optimizing the mean is the right thing to do if you care about
average behaviour. It is the wrong thing to do if you care about how bad the
model gets on its worst prompts, because a policy can raise its mean while
leaving -- or making worse -- a tail of systematic failures.

This module replaces the group baseline with risk-sensitive alternatives so
the same GRPO machinery can optimize a tail objective instead.

Estimators
----------
mean      standard GRPO. Baseline = group mean, scaled by group std.

cvar      Conditional Value-at-Risk at level alpha. Following the
          Rockafellar-Uryasev variational form, the CVaR gradient is
          supported only on rollouts at or below the alpha-quantile:

              A_i = 1{r_i <= q_alpha} * (r_i - q_alpha) / alpha

          Pure CVaR discards (1 - alpha) of every group, which is a large
          variance increase for small G. `lambda_risk` interpolates between
          the mean estimator and the tail estimator so the tail objective can
          be introduced without destabilising early training.

entropic  Exponential utility, U(r) = -(1/beta) log E[exp(-beta r)]. Smooth,
          uses every rollout, and recovers the mean as beta -> 0. Cheaper
          variance-wise than CVaR but the risk aversion is implicit rather
          than tied to a nameable quantile.

All estimators are group-relative: they operate within the rollouts of a
single prompt and require no learned value function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["RiskConfig", "group_advantages", "batch_advantages", "cvar", "value_at_risk"]

Estimator = Literal["mean", "cvar", "entropic"]

_EPS = 1e-6


@dataclass(frozen=True)
class RiskConfig:
    estimator: Estimator = "mean"
    alpha: float = 0.25
    """CVaR level. Fraction of the group treated as the tail. With G=8,
    alpha=0.25 keeps the 2 worst rollouts. Values below 1/G degenerate to a
    single-sample baseline and are rejected at construction."""

    beta: float = 1.0
    """Entropic risk aversion. beta -> 0 recovers the mean estimator."""

    lambda_risk: float = 1.0
    """Interpolation between the mean estimator (0.0) and the pure risk
    estimator (1.0). Anneal from 0 if pure tail optimization is unstable."""

    normalize: bool = True
    """Divide by group std. Standard in GRPO; disable to keep the advantage
    on the reward's own scale, which matters when comparing risk levels."""

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if not 0.0 <= self.lambda_risk <= 1.0:
            raise ValueError(f"lambda_risk must be in [0, 1], got {self.lambda_risk}")
        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")


# ---------------------------------------------------------------------------
# Risk functionals
# ---------------------------------------------------------------------------

def value_at_risk(rewards: np.ndarray, alpha: float) -> float:
    """Lower alpha-quantile of the reward distribution.

    Uses the 'lower' interpolation so the threshold is an achieved reward
    value rather than an interpolated one -- with the small integer-valued
    reward groups typical of verifiable-reward RL, interpolating produces a
    threshold no rollout can attain and empties the tail set.
    """
    return float(np.quantile(rewards, alpha, method="lower"))


def cvar(rewards: np.ndarray, alpha: float) -> float:
    """Mean of the worst alpha-fraction of rewards.

    Falls back to the single worst rollout when alpha is small enough that
    the tail set would otherwise be empty.
    """
    q = value_at_risk(rewards, alpha)
    tail = rewards[rewards <= q]
    if tail.size == 0:
        return float(rewards.min())
    return float(tail.mean())


# ---------------------------------------------------------------------------
# Advantage estimation
# ---------------------------------------------------------------------------

def group_advantages(rewards: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    """Per-rollout advantages for one prompt's group.

    rewards: shape (G,). Returns shape (G,).

    A degenerate group -- every rollout scoring identically -- carries no
    learning signal under any estimator and returns zeros. This is the common
    case late in training on easy prompts, and silently returning huge
    normalized advantages there is a well known way to destabilise a run.
    """
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if r.size == 0:
        return r
    if np.allclose(r, r[0]):
        return np.zeros_like(r)

    mean_adv = r - r.mean()

    if cfg.estimator == "mean":
        adv = mean_adv
    elif cfg.estimator == "cvar":
        q = value_at_risk(r, cfg.alpha)
        in_tail = (r <= q).astype(np.float64)
        # Rockafellar-Uryasev gradient: supported on the tail, scaled by 1/alpha.
        tail_adv = in_tail * (r - q) / max(cfg.alpha, _EPS)
        adv = (1.0 - cfg.lambda_risk) * mean_adv + cfg.lambda_risk * tail_adv
    elif cfg.estimator == "entropic":
        if cfg.beta <= _EPS:
            adv = mean_adv
        else:
            # Exponential tilting, stabilised by subtracting the max exponent.
            z = -cfg.beta * r
            w = np.exp(z - z.max())
            w /= w.sum()
            tail_adv = r - float((w * r).sum())  # baseline = risk-tilted mean
            adv = (1.0 - cfg.lambda_risk) * mean_adv + cfg.lambda_risk * tail_adv
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown estimator {cfg.estimator!r}")

    if cfg.normalize:
        adv = adv / (r.std() + _EPS)
    return adv


def batch_advantages(rewards: np.ndarray, cfg: RiskConfig) -> np.ndarray:
    """Vectorised over prompts. rewards: (B, G) -> advantages: (B, G)."""
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"expected (B, G), got shape {r.shape}")
    if r.shape[1] < 2:
        raise ValueError("group size G must be at least 2")
    if cfg.estimator == "cvar" and cfg.alpha < 1.0 / r.shape[1]:
        raise ValueError(
            f"alpha={cfg.alpha} with G={r.shape[1]} leaves fewer than one rollout "
            f"in the tail; use alpha >= {1.0 / r.shape[1]:.3f} or a larger group"
        )
    return np.stack([group_advantages(row, cfg) for row in r])
