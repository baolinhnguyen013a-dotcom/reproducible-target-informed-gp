"""Dense surrogate sweep using a ±2.5% relative inductance constraint and bounded regret."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from compare_surrogate_targets import ROOT, truth_grid, write_json
from sweep_budget_curves import fit

EPSILON = 0.025
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
BUDGETS = tuple(range(20, 41, 2))
VALIDATION_COUNT = 12
METHODS = ("baseline", "gp", "ensemble")
OUTPUT = ROOT / "results" / "01_current_relative_tolerance" / "01_surrogates_budget_20_to_40_10seeds"


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def choose(ids, grid, target):
    candidates = [grid[int(i)] for i in ids]
    valid = [point for point in candidates if feasible(point, target)]
    return min(valid, key=lambda point: point["R_ohm"]) if valid else min(
        candidates, key=lambda point: abs(point["L_mH"] - target) / target
    )


def decide(method, payload, grid, target):
    mean = np.asarray(payload["mean"])
    training = set(payload["training_ids"])
    unseen = np.array([index for index in range(len(grid)) if index not in training])
    relative_prediction_error = np.abs(mean[unseen] - target) / target
    if method == "baseline":
        resistance = np.asarray(payload["r_prediction"])
        preferred = unseen[relative_prediction_error <= EPSILON]
        preferred = preferred[np.argsort(resistance[preferred])]
        fallback = unseen[np.argsort(relative_prediction_error)]
        final_ids = np.array(list(preferred) + [i for i in fallback if i not in set(preferred)], dtype=int)[:VALIDATION_COUNT]
        return final_ids, choose(training | set(map(int, final_ids)), grid, target)
    final_ids = unseen[np.argsort(relative_prediction_error)[:VALIDATION_COUNT]]
    return final_ids, choose(final_ids, grid, target)


def bounded_loss(decision, optimum, target):
    if not feasible(decision, target):
        return False, None, 1.0
    regret = decision["R_ohm"] / optimum["R_ohm"] - 1
    return True, regret, regret / (1 + regret)


def append(path, rows, fields):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def summarise(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["budget"], row["target_mH"])].append(float(row["bounded_loss_u"]))
    summary = []
    for method in METHODS:
        for budget in BUDGETS:
            target_means = [sum(sorted(grouped[(method, budget, target)])[1:-1]) / 8 for target in TARGETS]
            summary.append({"method": method, "total_budget": budget,
                            "mean_trimmed_bounded_loss_u": sum(target_means) / len(target_means)})
    return summary


def plot(summary):
    labels = {"baseline": "Plain surrogate", "gp": "Gaussian process", "ensemble": "Deep ensemble (mean)"}
    figure, axis = plt.subplots(figsize=(10, 6))
    figure.subplots_adjust(bottom=.14)
    for method, label in labels.items():
        points = sorted((row["total_budget"], row["mean_trimmed_bounded_loss_u"]) for row in summary if row["method"] == method)
        x, y = map(np.array, zip(*points))
        fine_x = np.linspace(x.min(), x.max(), 300)
        axis.plot(fine_x, PchipInterpolator(x, y)(fine_x), linewidth=2, label=label)
        axis.scatter(x, y, zorder=3)
    axis.set(xlabel="Total true-evaluation budget", ylabel="Trimmed bounded resistance loss u (lower is better)",
             title="Surrogates, relative inductance tolerance ±2.5%", ylim=(-.02, 1.02), xticks=BUDGETS)
    axis.grid(alpha=.25); axis.legend()
    figure.text(.5, .025, "u = r/(1+r), r = R/R_exhaustive − 1; infeasible = 1. "
                         "Each point drops the highest and lowest of 10 seed losses.", ha="center", fontsize=8)
    figure.savefig(OUTPUT / "bounded_loss_by_budget.png", dpi=180)
    figure.savefig(OUTPUT / "bounded_loss_by_budget.svg")
    plt.close(figure)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config)
    fields = ["method", "budget", "seed", "target_mH", "qualified", "relative_resistance_regret",
              "resistance_increase_percent", "bounded_loss_u", "training_unique_evaluations",
              "validation_requests", "decision_T", "decision_n", "decision_L_mH", "decision_R_ohm"]
    all_rows = []
    write_json(OUTPUT / "manifest.json", {
        "relative_inductance_tolerance": EPSILON,
        "reference_calibration": "0.5 mH / 20 mH = 2.5%",
        "ensemble_rule": "mean prediction only",
        "loss": "r/(1+r), r=R/R_exhaustive-1; infeasible=1",
        "aggregation": "For each target/method/budget, drop highest and lowest of 10 losses, then average 8.",
    })
    for budget in BUDGETS:
        for seed in SEEDS:
            for method in METHODS:
                payload = fit(method, seed, budget, config, grid, EPSILON)
                rows = []
                for target in TARGETS:
                    optimum = min((point for point in grid if feasible(point, target)), key=lambda point: point["R_ohm"])
                    _, decision = decide(method, payload, grid, target)
                    qualified, regret, loss = bounded_loss(decision, optimum, target)
                    rows.append({
                        "method": method, "budget": budget, "seed": seed, "target_mH": target,
                        "qualified": qualified, "relative_resistance_regret": regret,
                        "resistance_increase_percent": None if regret is None else 100 * regret,
                        "bounded_loss_u": loss, "training_unique_evaluations": len(payload["training_ids"]),
                        "validation_requests": VALIDATION_COUNT, "decision_T": decision["T"], "decision_n": decision["n"],
                        "decision_L_mH": decision["L_mH"], "decision_R_ohm": decision["R_ohm"],
                    })
                append(OUTPUT / "results.csv", rows, fields)
                all_rows.extend(rows)
                print(f"checkpoint: {method}, budget={budget}, seed={seed}", flush=True)
    summary = summarise(all_rows)
    append(OUTPUT / "summary.csv", summary, list(summary[0]))
    plot(summary)


if __name__ == "__main__":
    main()
