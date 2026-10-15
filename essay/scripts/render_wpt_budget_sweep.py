#!/usr/bin/env python3
"""Render the B=30..80 WPT cEI versus 5%-mixture sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ESSAY = Path(__file__).resolve().parents[1]
DATA = ESSAY/"data/wpt_8d_budget_sweep_b30_b80"
FIGURES = ESSAY/"figures"


def main():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8.5,
                         "legend.fontsize": 7, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "pdf.fonttype": 42})
    summary = json.loads((DATA/"summary.json").read_text())
    labels = {"cei": "cEI only", "cei_all05": "Four-signal, 5% aux."}
    colors = {"cei": "#7F7F7F", "cei_all05": "#54A24B"}
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.25), sharex=True)
    fig.subplots_adjust(left=.19, right=.98, bottom=.14, top=.98, hspace=.12)
    for policy in labels:
        group = sorted((row for row in summary if row["policy"] == policy),
                       key=lambda row: row["budget"])
        budget = np.asarray([row["budget"] for row in group])
        qualification = 100*np.asarray([row["qualification_rate"] for row in group])
        mean = np.asarray([row["mean_R_qualified_ohm"] for row in group])
        std = np.asarray([row["std_R_qualified_ohm"] for row in group])
        axes[0].plot(budget, qualification, lw=1.6, color=colors[policy],
                     label=labels[policy])
        axes[1].plot(budget, mean, lw=1.6, color=colors[policy],
                     label=labels[policy])
        axes[1].fill_between(budget, mean-std, mean+std,
                             color=colors[policy], alpha=.16, linewidth=0)
    axes[0].set(ylabel=r"Qualified runs (\%)", xlim=(30, 80), ylim=(0, 105))
    axes[1].set(xlabel="Total physical-evaluation budget, $B$",
                ylabel=r"Mean qualified resistance ($\Omega$)", xlim=(30, 80))
    for axis in axes:
        axis.grid(color="#D9D9D9", lw=.55)
        axis.legend(frameon=False)
    FIGURES.mkdir(exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES/f"fig07_wpt_budget_sweep_b30_b80.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
