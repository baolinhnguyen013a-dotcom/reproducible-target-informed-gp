"""Matched B=27 pilot: target-conditioned TC-FEI versus the recorded global GP.

The pilot is resume-safe and checkpoints each completed target/seed execution.
TC-FEI is target-specific; the existing GP builds one reusable global model.
Both have the same standalone-target nominal physical-evaluation count.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import runpy
import random
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compare_surrogate_targets import ROOT, truth_grid, write_json

SEEDS = tuple(range(20261015, 20261025))
PILOT_TARGETS = (1, 5, 10, 20, 30, 40, 50, 60, 75, 87)
EPSILON = 0.025
BUDGET = 27
OUTPUT = ROOT / "results" / "coil_2d_relative_tolerance" / "tcfei_budget27_pilot"
RESULTS = OUTPUT / "tcfei_results.csv"
FIELDS = ("method", "budget", "seed", "target_mH", "qualified",
          "relative_resistance_regret", "bounded_loss_u", "decision_T",
          "decision_n", "decision_L_mH", "decision_R_ohm", "decision_source")


def feasible(point, target):
    return abs(point["L_mH"] - target) / target <= EPSILON


def run_one(task):
    seed, target = task
    with tempfile.TemporaryDirectory(prefix="coil-tcfei-pilot-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "config", workspace / "config")
        shutil.copytree(ROOT / "methods", workspace / "methods")
        (workspace / "results").mkdir()
        settings_path = workspace / "config" / "surrogates" / "target_conditioned_gp.json"
        settings = json.loads(settings_path.read_text())
        settings.update({"seed": seed, "target_mH": target})
        write_json(settings_path, settings)
        script = workspace / "methods" / "whole_range_surrogate_models" / "target_conditioned_gp.py"
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            values = runpy.run_path(str(script), run_name="__main__")
        decision = values["decision"]
        config = json.loads((workspace / "config" / "coil_problem.json").read_text())
        grid = truth_grid(config)
        optimum = min((point for point in grid if feasible(point, target)), key=lambda point: point["R_ohm"])
        qualified = feasible(decision, target)
        regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if qualified else None
        loss = regret / (1 + regret) if qualified else 1.0
        return {
            "method": "tcfei", "budget": BUDGET, "seed": seed, "target_mH": target,
            "qualified": qualified, "relative_resistance_regret": regret,
            "bounded_loss_u": loss, "decision_T": decision["point"][0],
            "decision_n": decision["point"][1], "decision_L_mH": decision["L_mH"],
            "decision_R_ohm": decision["R_ohm"], "decision_source": decision["source"],
        }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_row(row):
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def metrics(rows, targets):
    losses = [float(row["bounded_loss_u"]) for row in rows]
    qualification = [str(row["qualified"]).lower() == "true" for row in rows]
    by_seed = defaultdict(list)
    by_target = defaultdict(list)
    for row in rows:
        value = float(row["bounded_loss_u"])
        by_seed[int(row["seed"])].append(value)
        by_target[int(float(row["target_mH"]))].append(value)
    trimmed = [sum(sorted(by_target[target])[1:-1]) / 8 for target in targets]
    return {
        "cases": len(rows),
        "qualification_rate": sum(qualification) / len(qualification),
        "all_case_mean_u": sum(losses) / len(losses),
        "targetwise_trimmed_mean_u": sum(trimmed) / len(trimmed),
        "worst_seed_mean_u": max(sum(values) / len(values) for values in by_seed.values()),
        "worst_seed": max(by_seed, key=lambda seed: sum(by_seed[seed]) / len(by_seed[seed])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--targets", nargs="+", type=int, default=PILOT_TARGETS)
    parser.add_argument("--full", action="store_true", help="run all integer targets from 1 to 87 mH")
    args = parser.parse_args()
    targets = tuple(range(1, 88)) if args.full else tuple(args.targets)
    if any(target < 1 or target > 87 for target in targets):
        parser.error("targets must be integer values from 1 to 87 mH")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "manifest.json", {
        "status": "pilot", "budget": BUDGET, "seeds": list(SEEDS),
        "targets_mH": list(targets), "relative_tolerance": EPSILON,
        "tcfei_budget": "6 initial + 9 target-conditioned acquisitions + 12 validations",
        "comparison_warning": "Equal standalone cost; TC-FEI is target-specific whereas the baseline GP model is reusable.",
    })
    completed = {(int(row["seed"]), int(float(row["target_mH"]))) for row in read_rows(RESULTS)}
    tasks = [(seed, target) for seed in SEEDS for target in targets if (seed, target) not in completed]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_row(row)
            print(f"checkpoint {index}/{len(tasks)}: seed={row['seed']} target={row['target_mH']}", flush=True)

    tcfei = read_rows(RESULTS)
    gp_path = ROOT / "essay" / "data" / "gp_budget_27_results.csv"
    gp = [row for row in read_rows(gp_path) if int(float(row["target_mH"])) in targets]
    if len(tcfei) != len(SEEDS) * len(targets) or len(gp) != len(SEEDS) * len(targets):
        raise RuntimeError("incomplete pilot data")
    summary = {"tcfei": metrics(tcfei, targets), "global_gp": metrics(gp, targets)}
    source_counts = defaultdict(int)
    for row in tcfei:
        source = row["decision_source"]
        if source == "model_building":  # Backward-compatible classification for completed checkpoints.
            points = [(T, n) for T in range(1, 11) for n in range(20, 131)]
            initial = {points[i] for i in random.Random(int(row["seed"])).sample(range(len(points)), 6)}
            point = (int(row["decision_T"]), int(row["decision_n"]))
            source = "initial" if point in initial else "active_acquisition"
        source_counts[source] += 1
    summary["tcfei"]["decision_sources"] = dict(source_counts)
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
