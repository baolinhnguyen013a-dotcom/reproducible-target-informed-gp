#!/usr/bin/env python3
"""Audit, analyze, and render the WPT effective-sigma experiments."""
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
DATA = ESSAY/"data"
FIGURES = ESSAY/"figures"
SEEDS = tuple(range(34261011, 34261021))


def load(path):
    return list(csv.DictReader(path.open(newline="")))


def truth(value):
    return value == "True"


def bootstrap_ci(values, rng, draws=200_000):
    values = np.asarray(values, float)
    sampled = values[rng.integers(0, len(values), (draws, len(values)))]
    return np.quantile(sampled.mean(axis=1), [.025, .975]).tolist()


def main():
    rng = np.random.default_rng(20261012)
    control = load(DATA/"wpt_8d_sigma_mixture_gp_control/results.csv")
    main_rows = load(DATA/"wpt_8d_sigma_mixture/results.csv")
    asym_rows = load(DATA/"wpt_8d_sigma_mixture_asymmetric/results.csv")
    additive = load(DATA/"wpt_8d_low_auxiliary/results.csv")
    original = load(DATA/"wpt_8d_mixed/results.csv")
    sweep_base = load(DATA/"wpt_8d_budget_sweep_b30_b80/results.csv")
    sweep_sigma = load(DATA/"wpt_8d_sigma_mixture_sweep_b30_b80/results.csv")

    all_sigma = control+main_rows+asym_rows
    by_policy = {}
    for row in all_sigma:
        by_policy.setdefault(row["policy"], {})[int(row["seed"])] = row
    c = by_policy["sigma_gp_only"]
    original_c = {int(r["seed"]): r for r in original if r["policy"] == "cei"}
    assert all(c[s]["design"] == original_c[s]["design"] for s in SEEDS)
    assert all(abs(float(c[s]["best_R_ohm"])-float(original_c[s]["best_R_ohm"])) < 1e-12
               for s in SEEDS)

    initial = None
    for policy, rows in by_policy.items():
        for seed in SEEDS:
            first_six = json.loads(rows[seed]["active_designs"])[:6]
            seed_ref = json.dumps(first_six)
            if policy == "sigma_gp_only":
                continue
            control_ref = json.dumps(json.loads(c[seed]["active_designs"])[:6])
            assert seed_ref == control_ref

    additive_map = {int(r["seed"]): r for r in additive if r["policy"] == "cei_all05"}
    b50 = []
    order = ["sigma_gp_only", "sigma_90_05_05", "sigma_80_10_10",
             "sigma_80_15_05", "sigma_80_05_15", "sigma_60_20_20",
             "sigma_equal"]
    for policy in order:
        rows = by_policy[policy]
        values = np.asarray([float(rows[s]["best_R_ohm"]) for s in SEEDS])
        base = np.asarray([float(c[s]["best_R_ohm"]) for s in SEEDS])
        diff = values-base
        item = {
            "policy": policy, "qualification": sum(truth(rows[s]["qualified"]) for s in SEEDS),
            "mean_R_ohm": float(values.mean()), "population_std_R_ohm": float(values.std()),
            "mean_difference_vs_cei_ohm": float(diff.mean()),
            "percent_difference_vs_cei": float(100*diff.mean()/base.mean()),
            "paired_bootstrap_95pct_ci_ohm": bootstrap_ci(diff, rng),
            "seed_wins_vs_cei": int(np.sum(diff < 0)),
            "seed_ties_vs_cei": int(np.sum(diff == 0)),
            "wilcoxon_p": (None if np.all(diff == 0) else float(wilcoxon(diff).pvalue)),
        }
        b50.append(item)
    add_values = np.asarray([float(additive_map[s]["best_R_ohm"]) for s in SEEDS])
    best_values = np.asarray([float(by_policy["sigma_80_10_10"][s]["best_R_ohm"]) for s in SEEDS])
    diff_add = best_values-add_values

    base_map = {(r["policy"], int(r["budget"]), int(r["seed"])): r for r in sweep_base}
    sigma_map = {(int(r["budget"]), int(r["seed"])): r for r in sweep_sigma}
    standalone_best = by_policy["sigma_80_10_10"]
    assert all(sigma_map[50, s]["design"] == standalone_best[s]["design"] for s in SEEDS)
    lower_budgets, resolved = [], []
    seed_average = {"cei": [], "cei_all05": []}
    for budget in range(36, 81):
        sigma_v = np.asarray([float(sigma_map[budget, s]["best_R_ohm"]) for s in SEEDS])
        cei_v = np.asarray([float(base_map["cei", budget, s]["best_R_ohm"]) for s in SEEDS])
        diff = sigma_v-cei_v
        ci = bootstrap_ci(diff, rng, draws=100_000)
        if diff.mean() < 0:
            lower_budgets.append(budget)
        if ci[1] < 0 or ci[0] > 0:
            resolved.append({"budget": budget, "mean_difference_ohm": float(diff.mean()),
                             "paired_bootstrap_95pct_ci_ohm": ci})
    for policy in seed_average:
        for seed in SEEDS:
            seed_average[policy].append(np.mean([
                float(sigma_map[b, seed]["best_R_ohm"])
                - float(base_map[policy, b, seed]["best_R_ohm"])
                for b in range(36, 81)]))
    sweep_report = {
        "qualification_counts_b30_b36": {
            policy: [sum(truth((sigma_map if policy == "sigma_80_10_10" else
                                base_map)[(b, s) if policy == "sigma_80_10_10" else
                                          (policy, b, s)]["qualified"]) for s in SEEDS)
                     for b in range(30, 37)]
            for policy in ("cei", "cei_all05", "sigma_80_10_10")
        },
        "budgets_compared_after_all_qualify": 45,
        "sigma_lower_mean_R_budgets": lower_budgets,
        "sigma_lower_count": len(lower_budgets),
        "cei_lower_or_equal_count": 45-len(lower_budgets),
        "resolved_budget_differences": resolved,
    }
    for policy, values in seed_average.items():
        values = np.asarray(values)
        sweep_report[f"mean_difference_vs_{policy}_across_b36_b80_ohm"] = float(values.mean())
        sweep_report[f"paired_bootstrap_95pct_ci_vs_{policy}_ohm"] = bootstrap_ci(values, rng)
        sweep_report[f"wilcoxon_p_vs_{policy}"] = float(wilcoxon(values).pvalue)
    report = {
        "method": "quantile-calibrated replacement of sigma in PoF and EI",
        "baseline_identity_verified": True,
        "matched_six_point_initial_designs_verified": True,
        "b50_standalone_equals_sweep_verified": True,
        "b50": b50,
        "best_sigma_policy_vs_additive_5pct": {
            "mean_difference_ohm": float(diff_add.mean()),
            "percent_difference": float(100*diff_add.mean()/add_values.mean()),
            "paired_bootstrap_95pct_ci_ohm": bootstrap_ci(diff_add, rng),
            "seed_wins": int(np.sum(diff_add < 0)),
            "wilcoxon_p": float(wilcoxon(diff_add).pvalue),
        },
        "sweep_b30_b80": sweep_report,
    }
    out = DATA/"wpt_8d_sigma_mixture_analysis.json"
    out.write_text(json.dumps(report, indent=2)+"\n")

    labels = {"sigma_gp_only": "GP sigma only", "sigma_90_05_05": "90/5/5",
              "sigma_80_10_10": "80/10/10", "sigma_80_15_05": "80/15/5",
              "sigma_80_05_15": "80/5/15", "sigma_60_20_20": "60/20/20",
              "sigma_equal": "equal thirds"}
    fig = plt.figure(figsize=(7.16, 3.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(.9, 1.25), hspace=.12, wspace=.34)
    ax0 = fig.add_subplot(grid[:, 0]); ax1 = fig.add_subplot(grid[0, 1]); ax2 = fig.add_subplot(grid[1, 1], sharex=ax1)
    y = np.arange(len(order)); means = [next(x for x in b50 if x["policy"] == p)["mean_R_ohm"] for p in order]
    stds = [next(x for x in b50 if x["policy"] == p)["population_std_R_ohm"] for p in order]
    ax0.errorbar(means, y, xerr=stds, fmt="o", capsize=3, color="#4C78A8")
    ax0.set(yticks=y, yticklabels=[labels[p] for p in order], xlabel=r"Qualified resistance at $B=50$ ($\Omega$)")
    ax0.invert_yaxis(); ax0.grid(axis="x", color="#D9D9D9", lw=.55)
    colors = {"cei": "#7F7F7F", "cei_all05": "#54A24B", "sigma_80_10_10": "#E45756"}
    names = {"cei": "cEI", "cei_all05": "additive 5%", "sigma_80_10_10": r"effective $\sigma$ 80/10/10"}
    for policy in names:
        qual, resistance = [], []
        for b in range(30, 81):
            rows = ([sigma_map[b, s] for s in SEEDS] if policy == "sigma_80_10_10"
                    else [base_map[policy, b, s] for s in SEEDS])
            q = [r for r in rows if truth(r["qualified"])]
            qual.append(100*len(q)/10); resistance.append(np.mean([float(r["best_R_ohm"]) for r in q]))
        ax1.plot(range(30, 81), qual, color=colors[policy], lw=1.35, label=names[policy])
        ax2.plot(range(30, 81), resistance, color=colors[policy], lw=1.35, label=names[policy])
    ax1.set(ylabel=r"Qualified runs (\%)", ylim=(65, 102)); ax1.tick_params(labelbottom=False)
    ax2.set(xlabel="Budget, $B$", ylabel=r"Mean qualified $R$ ($\Omega$)")
    for ax in (ax1, ax2): ax.grid(color="#D9D9D9", lw=.55)
    ax1.legend(frameon=False, ncol=3, fontsize=6.3)
    fig.subplots_adjust(left=.16, right=.99, bottom=.15, top=.97)
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIGURES/f"fig08_wpt_sigma_mixture.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
