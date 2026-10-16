"""Checkpointed 10-seed score curves for total evaluation budgets 20–80."""
from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_surrogate_targets import ROOT, truth_grid, worker, write_json
from sweep_integer_targets import score, selection_rules

SEEDS = tuple(range(20261015, 20261025))
BUDGETS = tuple(range(22, 40, 2))
TARGETS = tuple(range(1, 88))
VALIDATION_COUNT = 12  # ceil(1% of the 1,110-point grid)
METHODS = ("baseline", "gp", "ensemble")
OUTPUT = ROOT / "results" / "surrogate_total_budget_22_to_38_exponential_10seeds"


def append_rows(path, rows, fields):
    new_file = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def fit(method, seed, total_budget, config, grid, tolerance):
    with tempfile.TemporaryDirectory(prefix="coil-budget-curve-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(ROOT / "methods" / "surrogate_selection", workspace / "methods" / "surrogate_selection")
        (workspace / "results").mkdir()
        filenames = {"baseline": "neural_surrogate_scan.json", "gp": "gaussian_process.json", "ensemble": "deep_ensemble.json"}
        settings_path = workspace / "config" / "surrogates" / filenames[method]
        settings = json.loads(settings_path.read_text())
        settings["seed"] = seed
        if method == "baseline":
            settings["training_samples"] = total_budget - VALIDATION_COUNT
            settings["validation_fraction"] = VALIDATION_COUNT / len(grid)
        else:
            settings["initial_samples"] = 6
            settings["maximum_true_evaluations"] = total_budget - VALIDATION_COUNT
            settings["final_validation_count"] = VALIDATION_COUNT
            if method == "ensemble":
                settings["sigma_tolerance_mH"] = 0.0
        write_json(settings_path, settings)
        # Keep transient warnings out of the results directory; raw numerical
        # outputs are retained in the checkpoint CSV instead.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            worker(workspace, method)
        return json.loads((workspace / "telemetry.json").read_text())


def make_plot(rows):
    labels = {"baseline": "Plain surrogate", "gp": "Gaussian process", "ensemble": "Deep ensemble"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["budget"], row["target_mH"])].append(row["performance_score"])
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    curve_rows = []
    for method, label in labels.items():
        points = []
        for budget in BUDGETS:
            target_means = []
            for target in TARGETS:
                values = sorted(grouped[(method, budget, target)])
                if len(values) != len(SEEDS):
                    raise ValueError(f"Missing seeds for {method}, budget {budget}, target {target}")
                target_means.append(sum(values[1:-1]) / (len(values) - 2))
            mean_score = sum(target_means) / len(target_means)
            points.append((budget, mean_score))
            curve_rows.append({"method": method, "training_budget": budget, "mean_trimmed_score": mean_score})
        axis.plot(*zip(*points), marker="o", linewidth=2, label=label)
    axis.set(xlabel="Total true-evaluation budget", ylabel="Mean trimmed performance score (L = 1–87 mH)",
             title="Surrogate performance versus total true-evaluation budget", ylim=(0, 105), xticks=BUDGETS)
    axis.grid(alpha=.25); axis.legend()
    figure.savefig(OUTPUT / "score_by_training_budget.png", dpi=180)
    figure.savefig(OUTPUT / "score_by_training_budget.svg")
    plt.close(figure)
    append_rows(OUTPUT / "budget_curve_summary.csv", curve_rows, ["method", "training_budget", "mean_trimmed_score"])


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config); tolerance = config["inductance_target"]["tolerance_mH"]
    write_json(OUTPUT / "manifest.json", {
        "seeds": SEEDS, "budgets": BUDGETS, "targets_mH": [1, 87],
        "final_validation_count": VALIDATION_COUNT,
        "evaluation_policy": "B is total true evaluations. Plain: (B-12) random training + 12 distinct validations. GP/ensemble: 6 initial + (B-18) acquired + 12 distinct validations.",
        "score": "max(0, 100 - (exp(100 * (R_method / R_exhaustive - 1)) - 1)); infeasible selections score 0.",
        "aggregation": "Per target, discard the lowest and highest of 10 seed scores, average the other 8, then average targets 1–87 mH."
    })
    fields = ["method", "budget", "seed", "target_mH", "performance_score", "qualified", "relative_resistance_regret",
              "training_unique_evaluations", "validation_requests", "decision_T", "decision_n", "decision_L_mH", "decision_R_ohm"]
    progress_fields = ["method", "budget", "seed", "status", "training_unique_evaluations", "validation_count"]
    all_rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            for method in METHODS:
                payload = fit(method, seed, budget, config, grid, tolerance)
                rows = []
                for target in TARGETS:
                    feasible = [p for p in grid if abs(p["L_mH"] - target) <= tolerance]
                    optimum = min(feasible, key=lambda p: p["R_ohm"])
                    rule = "plain" if method == "baseline" else ("default" if method == "gp" else "mean")
                    _, decision = selection_rules(method, payload, grid, target, tolerance, VALIDATION_COUNT)[rule]
                    qualified, regret, performance = score(decision, optimum, target, tolerance)
                    rows.append({"method": method, "budget": budget, "seed": seed, "target_mH": target,
                        "performance_score": performance, "qualified": qualified, "relative_resistance_regret": regret,
                        "training_unique_evaluations": len(payload["training_ids"]), "validation_requests": VALIDATION_COUNT,
                        "decision_T": decision["T"], "decision_n": decision["n"], "decision_L_mH": decision["L_mH"], "decision_R_ohm": decision["R_ohm"]})
                append_rows(OUTPUT / "results.csv", rows, fields)
                append_rows(OUTPUT / "progress.csv", [{"method": method, "budget": budget, "seed": seed, "status": "complete",
                    "training_unique_evaluations": len(payload["training_ids"]), "validation_count": VALIDATION_COUNT}], progress_fields)
                all_rows.extend(rows)
                print(f"checkpoint: {method}, budget={budget}, seed={seed}", flush=True)
    make_plot(all_rows)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
