#!/usr/bin/env python3
"""Nested B=30..80 sweep for the selected effective-sigma mixture."""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from run_wpt_8d_budget_sweep import evaluate_budget
from run_wpt_8d_mixed_benchmark import (
    N_CONFIRM, N_INITIAL, NOMINAL_ROTATION, NOMINAL_TRANSLATION, TARGET_M,
    evaluate, gps, normalize, sample_points,
)
from run_wpt_8d_sigma_mixture import sigma_mixture_acquisition

POLICY = "sigma_80_10_10"
WEIGHTS = (0.8, 0.1, 0.1)


def run_trajectory(task):
    seed, budgets, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(point) for point in x_raw])
    max_active = max(budgets)-N_CONFIRM
    snapshots, step = [], 0
    while True:
        x = normalize(x_raw)
        models = gps(x, y, seed+step)
        budget = len(x_raw)+N_CONFIRM
        if budget in budgets:
            # evaluate_budget expects four additive-score weights only for
            # metadata; replace those fields below with sigma weights.
            row = evaluate_budget(POLICY, (0., 0., 0., 0.), seed, budget,
                                  x_raw, y, models, candidate_count, started)
            row.pop("w_cei"); row.pop("w_gp")
            row["w_gp_sigma"] = WEIGHTS[0]
            row["w_disagreement"] = WEIGHTS[1]
            row["w_ntk"] = WEIGHTS[2]
            snapshots.append(row)
        if len(x_raw) >= max_active:
            break
        candidates = sample_points(seed*100003+step*1009+17, candidate_count)
        c = normalize(candidates)
        score = sigma_mixture_acquisition(
            models, c, y, x, y, WEIGHTS, seed*1000003+step*37, epochs)
        selected = candidates[int(np.argmax(score))]
        x_raw = np.vstack([x_raw, selected])
        y = np.vstack([y, evaluate(selected)])
        step += 1
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-min", type=int, default=30)
    parser.add_argument("--budget-max", type=int, default=80)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=34261011)
    parser.add_argument("--candidate-count", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(range(args.budget_min, args.budget_max+1))
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, budgets, args.candidate_count, args.epochs)
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trajectory, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"checkpoint {i}/{len(tasks)} trajectories", flush=True)
    rows.sort(key=lambda row: (row["seed"], row["budget"]))
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for budget in budgets:
        group = [row for row in rows if row["budget"] == budget]
        qualified = [row for row in group if row["qualified"]]
        summary.append({
            "policy": POLICY, "sigma_weights_gp_disagreement_ntk": WEIGHTS,
            "budget": budget, "cases": len(group),
            "qualification_rate": len(qualified)/len(group),
            "mean_R_qualified_ohm": (float(np.mean([r["best_R_ohm"] for r in qualified]))
                                      if qualified else None),
            "std_R_qualified_ohm": (float(np.std([r["best_R_ohm"] for r in qualified]))
                                     if qualified else None),
            "mean_target_error_pct": float(np.mean([r["relative_target_error_pct"] for r in group])),
        })
    manifest = {
        "method": "nested effective-sigma mixture sweep",
        "policy": POLICY, "sigma_weights_gp_disagreement_ntk": WEIGHTS,
        "dimensions": 8, "target_M_nH": TARGET_M,
        "fixed_pose": {"translation_mm": NOMINAL_TRANSLATION,
                       "rotation_rad": NOMINAL_ROTATION},
        "budgets": list(budgets), "n_initial": N_INITIAL,
        "n_confirmation": N_CONFIRM, "nested_active_histories": True,
        "confirmation_is_standalone_at_each_budget": True,
        "confirmation_candidate_pool_shared_across_budgets": True,
        "candidate_count": args.candidate_count, "ensemble_members": 10,
        "epochs": args.epochs,
        "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
