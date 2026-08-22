#!/usr/bin/env python3
"""Generate assets/results.png from runs/*/eval_k64/report.json.

Two-panel bar chart (mean accuracy, worst-decile accuracy) with 95% CIs,
following the categorical-color and mark-spec conventions in the dataviz
skill: one hue for the measurement, error bars in muted ink, direct value
labels instead of a legend (single series per panel).

    python scripts/plot_results.py
"""
from __future__ import annotations
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = [
    ("sft", "SFT"),
    ("dpo", "DPO"),
    ("grpo_mean", "GRPO\nmean"),
    ("grpo_cvar", "GRPO\ncvar"),
    ("grpo_entropic", "GRPO\nentropic"),
]

# dataviz skill reference palette (light mode)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SERIES_MUTED = "#b7d3f6"  # sequential step 150 -- for the two non-GRPO-comparison arms


def load(metric: str, ci_key: str):
    vals, lo, hi = [], [], []
    for key, _ in ARMS:
        d = json.loads((ROOT / "runs" / key / "eval_k64" / "report.json").read_text())
        vals.append(d[metric])
        ci = d[ci_key]
        lo.append(d[metric] - ci[0])
        hi.append(ci[1] - d[metric])
    return vals, [lo, hi]


def panel(ax, metric, ci_key, title):
    vals, err = load(metric, ci_key)
    x = range(len(ARMS))
    # GRPO mean is the control; SFT/DPO are context, cvar/entropic are the
    # treatment arms actually being compared to it -- color cues that
    # grouping rather than treating all five as one undifferentiated set.
    colors = [SERIES_MUTED, SERIES_MUTED, SERIES_BLUE, SERIES_BLUE, SERIES_BLUE]
    bars = ax.bar(x, vals, yerr=err, color=colors, width=0.6,
                   edgecolor=SURFACE, linewidth=0,
                   error_kw=dict(ecolor=INK_MUTED, elinewidth=1.4, capsize=4, capthick=1.4))
    for rect, v in zip(bars, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(err[1]) * 1.15 + 0.006,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10.5, color=INK_SECONDARY)

    ax.set_xticks(list(x))
    ax.set_xticklabels([label for _, label in ARMS], fontsize=10.5, color=INK_SECONDARY)
    ax.set_title(title, fontsize=12.5, color=INK_PRIMARY, loc="left", pad=14, fontweight="medium")
    ax.set_ylim(0, max(vals) + max(err[1]) + 0.09)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0, labelsize=9.5, colors=INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def main() -> int:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    panel(axes[0], "mean_acc", "mean_acc_ci95", "Mean accuracy")
    panel(axes[1], "worst_decile", "worst_decile_ci95", "Worst-decile accuracy")

    fig.suptitle(
        "No estimator moves either metric past sampling noise vs. the GRPO-mean control",
        fontsize=12, color=INK_PRIMARY, y=1.06, x=0.01, ha="left", fontweight="bold",
    )
    fig.text(0.01, 0.995, "200 GSM8K test prompts, k=64 rollouts/prompt · error bars: 95% bootstrap CI",
              fontsize=9.5, color=INK_MUTED, ha="left", va="top")

    fig.subplots_adjust(wspace=0.28)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = ROOT / "assets" / "results.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
