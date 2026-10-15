#!/usr/bin/env python3
"""Audit and render joint-acquisition/diverse-confirmation results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ESSAY = Path(__file__).resolve().parents[1]
DATA, FIGURES = ESSAY/"data", ESSAY/"figures"
SEEDS = tuple(range(34261011, 34261021))


def load(path):
    return list(csv.DictReader(path.open(newline="")))


def ci(values, rng):
    values = np.asarray(values, float)
    sampled = values[rng.integers(0, len(values), (200_000, len(values)))]
    return np.quantile(sampled.mean(1), [.025, .975]).tolist()


def main():
    rows = load(DATA/"wpt_8d_average_acquisition_diversity/results.csv")
    pure_rows = load(DATA/"wpt_8d_sigma_mixture_gp_control/results.csv")
    sigma_rows = load(DATA/"wpt_8d_sigma_mixture/results.csv")
    policies = list(dict.fromkeys(row["confirmation_policy"] for row in rows))
    table = {(row["confirmation_policy"], int(row["seed"])): row for row in rows}
    pure = {int(row["seed"]): row for row in pure_rows}
    sigma90 = {int(row["seed"]): row for row in sigma_rows
               if row["policy"] == "sigma_90_05_05"}
    assert len(rows) == 7*10
    assert all(row["qualified"] == "True" and int(row["budget"]) == 50 for row in rows)
    assert all(len({table[p, s]["active_designs"] for p in policies}) == 1
               for s in SEEDS)
    active_equals_sigma90 = all(
        table["original_top12", s]["active_designs"] == sigma90[s]["active_designs"]
        for s in SEEDS)
    rng = np.random.default_rng(20261012)
    pure_values = np.asarray([float(pure[s]["best_R_ohm"]) for s in SEEDS])
    original = np.asarray([float(table["original_top12", s]["best_R_ohm"])
                           for s in SEEDS])
    results = []
    for policy in policies:
        values = np.asarray([float(table[policy, s]["best_R_ohm"]) for s in SEEDS])
        vs_pure, vs_original = values-pure_values, values-original
        results.append({
            "policy": policy, "qualification": "10/10",
            "mean_R_ohm": float(values.mean()),
            "population_std_R_ohm": float(values.std()),
            "mean_pair_min_distance": float(np.mean([
                float(table[policy, s]["confirmation_pair_min_distance"]) for s in SEEDS])),
            "mean_difference_vs_pure_cei_ohm": float(vs_pure.mean()),
            "paired_bootstrap_95pct_ci_vs_pure_cei_ohm": ci(vs_pure, rng),
            "mean_difference_vs_joint_original_confirmation_ohm": float(vs_original.mean()),
            "paired_bootstrap_95pct_ci_vs_joint_original_ohm": ci(vs_original, rng),
            "seed_wins_vs_joint_original": int(np.sum(vs_original < 0)),
            "seed_ties_vs_joint_original": int(np.sum(vs_original == 0)),
            "wilcoxon_p_vs_joint_original": (None if np.all(vs_original == 0)
                                              else float(wilcoxon(vs_original).pvalue)),
        })
    report = {
        "active_score": "0.5*unit(cEI) + 0.5*unit(cEI with effective sigma .8/.1/.1)",
        "all_confirmation_policies_share_active_history": True,
        "joint_active_history_equals_sigma_90_05_05_history": active_equals_sigma90,
        "pure_cei_reference_mean_R_ohm": float(pure_values.mean()),
        "results": results,
        "interpretation": (
            "The joint acquisition collapses to the previously observed GP-heavy "
            "effective-sigma trajectory and is worse than pure cEI. Mild diversity "
            "changes one seed without resolving the gap; strong diversity overexplores."
        ),
    }
    (DATA/"wpt_8d_average_acquisition_diversity_analysis.json").write_text(
        json.dumps(report, indent=2)+"\n")

    short = {"original_top12": "original", "soft_distance_batch_p0p10": r"soft $p=.10$",
             "soft_distance_batch_p0p20": r"soft $p=.20$", "soft_distance_batch": r"soft $p=1$",
             "soft_distance_active_and_batch": "soft all",
             "hard_minsep_0p30": r"hard $d=.30$", "hard_minsep_0p40": r"hard $d=.40$"}
    offsets = {
        "original_top12": (-10, 17),
        "hard_minsep_0p30": (8, 8),
        "soft_distance_batch_p0p10": (-5, -20),
        "hard_minsep_0p40": (8, -3),
        "soft_distance_batch_p0p20": (5, 3),
        "soft_distance_batch": (-40, 13),
        "soft_distance_active_and_batch": (-50, -16),
    }
    x = [r["mean_pair_min_distance"] for r in results]
    y = [r["mean_R_ohm"] for r in results]
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    for item, xv, yv in zip(results, x, y):
        ax.scatter(xv, yv, s=28)
        ax.annotate(short[item["policy"]], (xv, yv),
                    xytext=offsets[item["policy"]],
                    textcoords="offset points", fontsize=6.5)
    ax.axhline(pure_values.mean(), color="#7F7F7F", ls="--", lw=1.2,
               label="pure cEI reference")
    ax.set(xlabel="Mean minimum pair distance in confirmation batch",
           ylabel=r"Mean qualified resistance ($\Omega$)")
    ax.grid(color="#D9D9D9", lw=.55); ax.legend(frameon=False, fontsize=7)
    ax.margins(x=.05, y=.08)
    fig.subplots_adjust(left=.15, right=.98, bottom=.20, top=.94)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES/f"fig09_joint_acquisition_diversity.{ext}",
                    dpi=300 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
