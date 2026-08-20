#!/usr/bin/env python3
"""Paired comparison of two evaluation runs on mean and tail statistics.

    python scripts/compare.py runs/grpo_mean/eval runs/grpo_cvar/eval
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from rsp.metrics import paired_bootstrap, evaluate_rollouts


def load(p: str) -> np.ndarray:
    return np.load(pathlib.Path(p) / "rollouts.npz")["correct"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline"); ap.add_argument("treatment")
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.treatment)
    if a.shape[0] != b.shape[0]:
        raise SystemExit("runs must cover the same prompt set to be paired")
    pa, pb = a.mean(axis=1), b.mean(axis=1)

    def tail(v: np.ndarray) -> float:
        k = max(1, int(np.floor(args.alpha * v.size)))
        return float(np.sort(v)[:k].mean())

    out = {
        "baseline": args.baseline,
        "treatment": args.treatment,
        "mean_acc": paired_bootstrap(pa, pb, np.mean),
        f"cvar_{args.alpha}": paired_bootstrap(pa, pb, tail),
        "zero_solve_rate": paired_bootstrap(
            (a.sum(axis=1) == 0).astype(float), (b.sum(axis=1) == 0).astype(float), np.mean),
        "baseline_report": json.loads(evaluate_rollouts(a).to_json()),
        "treatment_report": json.loads(evaluate_rollouts(b).to_json()),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
