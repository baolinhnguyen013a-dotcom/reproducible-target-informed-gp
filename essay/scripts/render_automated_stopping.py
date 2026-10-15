#!/usr/bin/env python3
"""Render the 2-D stopping comparison and deterministic 8-D WPT trajectory."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/"essay"/"data"
FIG = ROOT/"essay"/"figures"/"fig05_automated_stopping"


def load_json(path):
    return json.loads(path.read_text())


def main():
    summaries = [
        ("0.1%\nrandom", DATA/"center_cosine_stopping_smoke"/"summary.json",
         "ordinary_stop"),
        ("1%\nrandom", DATA/"center_stopping_1pct_smoke"/"summary.json",
         "ordinary_stop"),
        ("1%\nfixed $K=5$", DATA/"center_stopping_1pct_deterministic_gp_k5"/"summary.json",
         "ordinary_stop"),
    ]
    calls, exact = [], []
    for _, path, policy in summaries:
        result = next(row for row in load_json(path)["results"] if row["policy"] == policy)
        calls.append(result["mean_oracle_calls"])
        exact.append(100*result["exact_optimum_rate"])

    trace = list(csv.DictReader((DATA/"wpt_8d_automated_cei"/"trace.csv").open()))
    model_call = np.asarray([int(row["model_call"]) for row in trace])
    mutual = np.asarray([float(row["M_nH"]) for row in trace])
    feasible = np.abs(mutual-1500) <= .025*1500
    rho_x = [int(row["model_call"])-1 for row in trace if row["rho"]]
    rho = [float(row["rho"]) for row in trace if row["rho"]]
    wpt_result = load_json(DATA/"wpt_8d_automated_cei"/"result.json")
    rho_x.append(wpt_result["n_model_building"])
    rho.append(wpt_result["final_rho"])

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.62), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1.35, .9]})
    x = np.arange(3)
    bars = axes[0].bar(x, calls, color=["#8da0cb", "#66c2a5", "#1b9e77"], width=.68)
    axes[0].axhline(27, color="0.35", ls="--", lw=1)
    axes[0].set_xticks(x, [item[0] for item in summaries])
    axes[0].tick_params(axis="x", labelsize=6.5)
    axes[0].set_ylabel("Mean physical evaluations")
    axes[0].set_ylim(0, 30)
    axes[0].set_title("(a) Two-dimensional stopping screen")
    axes[0].text(1.97, 27.45, "fixed $B=27$", ha="right", va="bottom",
                 fontsize=6.2,
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": .35})
    for bar, value, accuracy in zip(bars, calls, exact):
        axes[0].text(bar.get_x()+bar.get_width()/2, value-.55,
                     f"{value:.2f}\n{accuracy:.0f}%", ha="center", va="top",
                     fontsize=6.2, color="white", fontweight="bold")

    ax = axes[1]
    ax.axhspan(1462.5, 1537.5, color="#a6d854", alpha=.25, label="feasible $M$ band")
    ax.scatter(model_call[~feasible], mutual[~feasible], s=17, color="0.55", label="infeasible call")
    ax.scatter(model_call[feasible], mutual[feasible], s=25, color="#1b9e77", label="feasible call", zorder=3)
    ax.axvline(34.5, color="#d95f02", ls="--", lw=1.1, label="stop after call 34")
    ax.set(xlabel="Model-building evaluation", ylabel="Mutual inductance (nH)")
    ax.set_title("(b) 8-D WPT trajectory", pad=28)
    ax.set_xlim(.3, 40.5)
    ax.legend(frameon=False, fontsize=5.8, loc="lower center",
              bbox_to_anchor=(.5, 1.005), ncol=2, columnspacing=.8,
              handletextpad=.35, borderaxespad=0)

    ax = axes[2]
    ax.plot(rho_x, 100*np.asarray(rho), color="#7570b3", marker="o",
            ms=2.7, lw=1.15)
    ax.axhline(1, color="#d95f02", ls="--", lw=1.1, label="1% threshold")
    ax.axvspan(32, 34.25, color="#a6d854", alpha=.24,
               label="three refits")
    ax.set_yscale("log")
    ax.set(xlabel="Observations at refit", ylabel="Maximum relative cEI (%)")
    ax.set_title("(c) Stopping evidence", pad=28)
    ax.set_xlim(10.5, 35)
    ax.legend(frameon=False, fontsize=5.8, loc="lower center",
              bbox_to_anchor=(.5, 1.005), ncol=2, columnspacing=.7,
              handletextpad=.3, borderaxespad=0)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(FIG.with_suffix("."+suffix), dpi=300)


if __name__ == "__main__":
    main()
