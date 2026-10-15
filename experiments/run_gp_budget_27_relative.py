"""Checkpoint the missing B=27 GP point under the current relative-L protocol."""
from __future__ import annotations

import csv
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from compare_surrogate_targets import ROOT, truth_grid, worker, write_json

EPSILON = 0.025
SEEDS = tuple(range(20261015, 20261025))
TARGETS = tuple(range(1, 88))
VALIDATION_COUNT = 12

BUDGET = 27  # 6 initial + 9 active + 12 target-specific validations
OUTPUT = ROOT / "results" / "01_current_relative_tolerance" / "01_surrogates_budget_20_to_40_10seeds"
RESULTS = OUTPUT / "gp_budget_27_results.csv"
MANIFEST = OUTPUT / "gp_budget_27_manifest.json"
FIELDS = ["method", "budget", "seed", "target_mH", "qualified", "relative_resistance_regret",
          "resistance_increase_percent", "bounded_loss_u", "training_unique_evaluations",
          "validation_requests", "decision_T", "decision_n", "decision_L_mH", "decision_R_ohm"]


def fit_gp(seed: int, config: dict, grid: list[dict]) -> dict:
    """Fit the maintained GP implementation with an exact B=27 budget."""
    with tempfile.TemporaryDirectory(prefix="coil-gp-27-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(
            ROOT / "methods" / "whole_range_surrogate_models",
            workspace / "methods" / "whole_range_surrogate_models",
        )
        (workspace / "results").mkdir()
        settings_path = workspace / "config" / "surrogates" / "gaussian_process.json"
        settings = json.loads(settings_path.read_text())
        settings.update({
            "seed": seed,
            "initial_samples": 6,
            "maximum_true_evaluations": BUDGET - VALIDATION_COUNT,
            "final_validation_count": VALIDATION_COUNT,
        })
        write_json(settings_path, settings)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            worker(workspace, "gp")
        return json.loads((workspace / "telemetry.json").read_text())


def feasible(point: dict, target: float) -> bool:
    return abs(point["L_mH"] - target) / target <= EPSILON


def choose(ids, grid: list[dict], target: float) -> dict:
    candidates = [grid[int(index)] for index in ids]
    valid = [point for point in candidates if feasible(point, target)]
    return min(valid, key=lambda point: point["R_ohm"]) if valid else min(
        candidates, key=lambda point: abs(point["L_mH"] - target) / target
    )


def decide(payload: dict, grid: list[dict], target: float) -> dict:
    mean = np.asarray(payload["mean"])
    training = set(payload["training_ids"])
    unseen = np.array([index for index in range(len(grid)) if index not in training])
    relative_prediction_error = np.abs(mean[unseen] - target) / target
    final_ids = unseen[np.argsort(relative_prediction_error)[:VALIDATION_COUNT]]
    return choose(final_ids, grid, target)


def bounded_loss(decision: dict, optimum: dict, target: float):
    if not feasible(decision, target):
        return False, None, 1.0
    regret = decision["R_ohm"] / optimum["R_ohm"] - 1
    return True, regret, regret / (1 + regret)


def completed_seeds() -> set[int]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open(newline="") as stream:
        return {int(row["seed"]) for row in csv.DictReader(stream)}


def append(rows: list[dict]) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config)
    write_json(MANIFEST, {
        "method": "gp", "budget": BUDGET, "seeds": list(SEEDS),
        "targets_mH": [1, 87], "relative_inductance_tolerance": EPSILON,
        "budget_composition": "6 initial + 9 maximum-uncertainty acquisitions + 12 validations",
        "decision_rule": "Same GP rule as the main relative-tolerance sweep; final choice is from 12 validations.",
    })
    done = completed_seeds()
    for seed in SEEDS:
        if seed in done:
            print(f"skip completed seed {seed}", flush=True)
            continue
        payload = fit_gp(seed, config, grid)
        rows = []
        for target in TARGETS:
            feasible = [point for point in grid if abs(point["L_mH"] - target) / target <= EPSILON]
            optimum = min(feasible, key=lambda point: point["R_ohm"])
            decision = decide(payload, grid, target)
            qualified, regret, loss = bounded_loss(decision, optimum, target)
            rows.append({
                "method": "gp", "budget": BUDGET, "seed": seed, "target_mH": target,
                "qualified": qualified, "relative_resistance_regret": regret,
                "resistance_increase_percent": None if regret is None else 100 * regret,
                "bounded_loss_u": loss, "training_unique_evaluations": len(payload["training_ids"]),
                "validation_requests": VALIDATION_COUNT, "decision_T": decision["T"],
                "decision_n": decision["n"], "decision_L_mH": decision["L_mH"],
                "decision_R_ohm": decision["R_ohm"],
            })
        append(rows)
        print(f"checkpoint seed {seed}", flush=True)


if __name__ == "__main__":
    main()
