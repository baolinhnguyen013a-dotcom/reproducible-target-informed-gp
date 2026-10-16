#!/usr/bin/env python3
"""Render the mixed-uncertainty budget and 8-D WPT result figures."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ESSAY = Path(__file__).resolve().parents[1]
PROJECT = ESSAY.parent
FIGURES = ESSAY / "figures"
MIXED = PROJECT / "results/coil_2d_relative_tolerance/mixed_uncertainty"
WPT = ESSAY / "data/wpt_8d_mixed"
WPT_LOW = ESSAY / "data/wpt_8d_low_auxiliary"

BUDGET_LABELS = {"cei": "cEI only", "cei_all": "Four-signal, 15% aux."}
WPT_LABELS = {"cei": "cEI only", "cei_gp": "cEI + GP-global",
              "cei_neural": "cEI + NN pair", "cei_all": "Four-signal, 10% aux.",
              "cei_all05": "Four-signal, 5% aux."}
COLORS = {"cei": "#7F7F7F", "cei_gp": "#4C78A8",
          "cei_neural": "#F58518", "cei_all": "#54A24B", "cei_all05": "#76B7B2"}


def configure():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def budget_figure():
    rows = list(csv.DictReader((MIXED/"confirmation_b24_b30/results.csv").open(newline="")))
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["policy"], int(r["budget"]))].append(r)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.55), sharex=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.13, top=.98, hspace=.12)
    available = [policy for policy in BUDGET_LABELS if any(p == policy for p, _ in grouped)]
    for policy in available:
        budgets = sorted(b for p, b in grouped if p == policy)
        mean_loss, exact = [], []
        for b in budgets:
            g = grouped[(policy, b)]
            mean_loss.append(np.mean([float(r["bounded_loss_u"]) for r in g]))
            exact.append(100*np.mean([r["exact_optimum"] == "True" for r in g]))
        axes[0].plot(budgets, mean_loss, marker="o", lw=1.5, ms=3.5,
                     color=COLORS[policy], label=BUDGET_LABELS[policy])
        axes[1].plot(budgets, exact, marker="o", lw=1.5, ms=3.5, color=COLORS[policy])
    axes[0].set_ylabel("Mean bounded loss, $u$")
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].legend(frameon=False, ncol=2, fontsize=6.4)
    axes[1].set(xlabel="Total physical-evaluation budget, $B=6+x+12$",
                ylabel=r"Exact-optimum recovery (\%)", xticks=range(24, 31))
    for ax in axes:
        ax.grid(color="#D9D9D9", lw=.55)
    save(fig, "fig05_mixed_uncertainty_budget")


def screening_figure():
    summary = json.loads((MIXED/"development/summary.json").read_text())
    names = [r["policy"] for r in summary]
    losses = [r["mean_u"] for r in summary]
    labels = {"pure": "1/0/0/0", "gp20": ".8/.2/0/0", "neural20": ".8/0/.1/.1",
              "equal10": ".7/.1/.1/.1", "equal05": ".85/.05/.05/.05",
              "equal15": ".55/.15/.15/.15", "dis15_ntk05": ".7/.1/.15/.05",
              "dis05_ntk15": ".7/.1/.05/.15"}
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    fig.subplots_adjust(left=.20, right=.98, bottom=.29, top=.96)
    ax.bar(np.arange(len(names)), losses, color=["#54A24B" if n == "equal15" else "#9ECAE9" for n in names])
    ax.set(xticks=np.arange(len(names)), xticklabels=[labels[n] for n in names],
           ylabel="Development mean loss, $u$")
    ax.tick_params(axis="x", rotation=38)
    ax.grid(axis="y", color="#D9D9D9", lw=.55)
    save(fig, "fig05a_weight_screening")


def wpt_figure():
    summary = json.loads((WPT/"summary.json").read_text())
    summary.extend(json.loads((WPT_LOW/"summary.json").read_text()))
    order = ["cei", "cei_gp", "cei_neural", "cei_all", "cei_all05"]
    lookup = {r["policy"]: r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.25))
    fig.subplots_adjust(left=.08, right=.99, bottom=.25, top=.96, wspace=.28)
    x = np.arange(len(order))
    labels = [WPT_LABELS[n] for n in order]
    colors = [COLORS[n] for n in order]
    axes[0].bar(x, [100*lookup[n]["qualification_rate"] for n in order], color=colors)
    axes[0].set(ylabel=r"Qualified runs (\%)", ylim=(0, 105), xticks=x, xticklabels=labels)
    means = [lookup[n]["mean_R_qualified_ohm"] for n in order]
    stds = [lookup[n]["std_R_qualified_ohm"] for n in order]
    axes[1].bar(x, means, yerr=stds, capsize=3, color=colors)
    axes[1].set(ylabel=r"Best qualified resistance ($\Omega$)", xticks=x, xticklabels=labels)
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", color="#D9D9D9", lw=.55)
    save(fig, "fig06_wpt_8d_mixed")


def main():
    FIGURES.mkdir(exist_ok=True)
    configure()
    screening_figure()
    budget_figure()
    wpt_figure()


if __name__ == "__main__":
    main()
