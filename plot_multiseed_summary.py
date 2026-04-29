"""
Visualise the multi-seed evaluation produced by:
    python retina_attention_fusion.py --seeds 42 7 13 99 2024

Reads ``models/retina_attention_fusion_seed_summary.json`` and writes:
  - models/multiseed_metrics.png       per-metric strip+bar with mean ± std
  - models/multiseed_per_class_f1.png  per-class F1 across seeds (mean ± std)

These are the figures to use in the FYP-II report for the
"robustness across seeds" subsection.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Normal", "Class 1", "Class 3", "Class 4", "Class 6", "Class 7", "Class 8"]
SUMMARY_PATH = "models/retina_attention_fusion_seed_summary.json"
OUT_METRICS = "models/multiseed_metrics.png"
OUT_PCF1 = "models/multiseed_per_class_f1.png"


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"Not found: {path} — run multi-seed training first.")
    with open(path) as f:
        return json.load(f)


def main() -> None:
    summary = _load(SUMMARY_PATH)
    seeds = summary["seeds"]
    agg = summary["aggregate"]
    per_seed = summary["per_seed"]

    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    pretty = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "macro_f1": "Macro-F1",
    }

    # 1) Per-metric strip+bar
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.5), sharey=True)
    for ax, m in zip(axes, metrics):
        vals = np.array(agg[m]["values"])
        mean = float(agg[m]["mean"])
        std = float(agg[m]["std"])
        ax.bar([0], [mean], width=0.5, color="#cbd5e8", edgecolor="#3b3b3b", label="mean")
        ax.errorbar([0], [mean], yerr=[std], fmt="none", ecolor="#3b3b3b", capsize=8, linewidth=1.5)
        x_jitter = np.linspace(-0.18, 0.18, len(vals))
        ax.scatter(x_jitter, vals, s=72, c="#1f77b4", edgecolor="white", zorder=3)
        for x, v, s in zip(x_jitter, vals, seeds):
            ax.annotate(f"s={s}", (x, v), xytext=(2, 8), textcoords="offset points", fontsize=8)
        ax.set_xticks([0])
        ax.set_xticklabels([pretty[m]])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_title(f"{pretty[m]}\nmean={mean:.4f} ± {std:.4f}", fontsize=11)
    axes[0].set_ylabel("Score")
    fig.suptitle(
        f"Attention Mid-Level Fusion — {len(seeds)}-seed robustness (test set)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_METRICS, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_METRICS}")

    # 2) Per-class F1 across seeds
    classes_in_runs = sorted({int(c) for r in per_seed for c in r["per_class_f1"].keys()})
    arr = np.array(
        [
            [r["per_class_f1"].get(str(c), float("nan")) for c in classes_in_runs]
            for r in per_seed
        ]
    )  # (n_seeds, n_classes)
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(classes_in_runs))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color="#3b82c4", edgecolor="#1f3b5c")
    for xi, vals_per_seed in zip(x, arr.T):
        ax.scatter(np.full_like(vals_per_seed, xi), vals_per_seed, s=22, c="white", edgecolor="black", zorder=4)
    labels = [
        CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"Class {c}"
        for c in classes_in_runs
    ]
    supports = per_seed[0]["support"]
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n(n={supports.get(str(c), 0)})" for lbl, c in zip(labels, classes_in_runs)],
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (mean ± std across seeds)")
    ax.set_title(f"Per-class F1 — {len(seeds)}-seed average")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for xi, m in zip(x, means):
        ax.text(xi, m + 0.03, f"{m:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PCF1, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PCF1}")


if __name__ == "__main__":
    main()
