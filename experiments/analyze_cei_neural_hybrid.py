#!/usr/bin/env python3
"""Analyze the frozen-weight S3 confirmation benchmark and render its figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "pure_cei": "GP-cEI",
    "cei_gp_global": "cEI + GP uncertainty",
    "cei_nn_disagreement": "cEI + NN disagreement",
    "cei_nn_ntk": "cEI + NN NTK",
}
COLORS = {
    "pure_cei": "#4C78A8",
    "cei_gp_global": "#F58518",
    "cei_nn_disagreement": "#54A24B",
    "cei_nn_ntk": "#B279A2",
}


def bootstrap_seed_difference(frame, challenger, budget, baseline="pure_cei", repetitions=50000):
    part = frame[frame["budget"] == budget]
    pivot = part.pivot(index=["seed", "target_mH"], columns="method", values="bounded_loss_u")
    seed_diff = (pivot[challenger] - pivot[baseline]).unstack("target_mH").mean(axis=1).to_numpy()
    rng = np.random.default_rng(20261011 + int(budget))
    samples = seed_diff[rng.integers(0, len(seed_diff), (repetitions, len(seed_diff)))].mean(axis=1)
    return {
        "mean_difference": float(seed_diff.mean()),
        "ci95": [float(x) for x in np.quantile(samples, [.025, .975])],
        "seed_wins": int((seed_diff < 0).sum()),
        "seed_losses": int((seed_diff > 0).sum()),
        "seed_ties": int((seed_diff == 0).sum()),
    }


def analyze(frame):
    records = []
    for (method, budget), part in frame.groupby(["method", "budget"]):
        records.append({
            "method": method, "budget": int(budget), "cases": int(len(part)),
            "mean_u": float(part["bounded_loss_u"].mean()),
            "qualification_rate": float(part["qualified"].mean()),
            "exact_rate": float(part["exact_optimum"].mean()),
            "top10_rate": float(part["top10"].mean()),
            "mean_runtime_sec": float(part["runtime_sec"].mean()),
        })
    paired = {}
    supplement_pairs = {}
    changes = {}
    for budget in sorted(frame["budget"].unique()):
        paired[str(int(budget))] = {}
        supplement_pairs[str(int(budget))] = {}
        changes[str(int(budget))] = {}
        base = frame[(frame.method == "pure_cei") & (frame.budget == budget)].set_index(
            ["seed", "target_mH"]
        )
        for method in LABELS:
            if method == "pure_cei":
                continue
            paired[str(int(budget))][method] = bootstrap_seed_difference(frame, method, budget)
            challenger = frame[(frame.method == method) & (frame.budget == budget)].set_index(
                ["seed", "target_mH"]
            )
            common = base.index.intersection(challenger.index)
            changes[str(int(budget))][method] = float(np.mean(
                base.loc[common, "selected_indices"].to_numpy()
                != challenger.loc[common, "selected_indices"].to_numpy()
            ))
        for challenger, baseline in (
            ("cei_nn_ntk", "cei_gp_global"),
            ("cei_nn_ntk", "cei_nn_disagreement"),
            ("cei_nn_disagreement", "cei_gp_global"),
        ):
            key = f"{challenger}_minus_{baseline}"
            supplement_pairs[str(int(budget))][key] = bootstrap_seed_difference(
                frame, challenger, budget, baseline=baseline
            )
    return records, paired, supplement_pairs, changes


def render(summary, output):
    frame = pd.DataFrame(summary)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.7,
        "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.25), sharex=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.15, top=.88, hspace=.14)
    for method in LABELS:
        part = frame[frame.method == method].sort_values("budget")
        axes[0].plot(part.budget, part.mean_u, marker="o", ms=4, lw=1.5,
                     color=COLORS[method], label=LABELS[method])
        axes[1].plot(part.budget, 100 * part.exact_rate, marker="o", ms=4, lw=1.5,
                     color=COLORS[method])
    axes[0].set_ylabel("Mean bounded loss, $u$")
    axes[1].set_ylabel("Exact optimum (\%)")
    axes[1].set_xlabel("Total physical-evaluation budget")
    axes[1].set_xticks(sorted(frame.budget.unique()))
    for axis in axes:
        axis.grid(color="#D9D9D9", lw=.55)
    fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(output / f"s3_confirmation_by_budget.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--figure-output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    required = 4 * 2 * 10 * 87
    if len(frame) != required:
        raise RuntimeError(f"confirmation incomplete: {len(frame)} rows, expected {required}")
    for name in ("qualified", "exact_optimum", "top10"):
        frame[name] = frame[name].astype(str).str.lower().eq("true")
    summary, paired, supplement_pairs, changes = analyze(frame)
    output = args.results.parent
    pd.DataFrame(summary).to_csv(output / "analysis_summary.csv", index=False)
    payload = {"summary": summary, "paired_vs_pure_cei": paired,
               "paired_between_supplements": supplement_pairs,
               "acquisition_set_change_rate_vs_pure_cei": changes}
    (output / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    figure_output = args.figure_output or output
    figure_output.mkdir(parents=True, exist_ok=True)
    render(summary, figure_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
