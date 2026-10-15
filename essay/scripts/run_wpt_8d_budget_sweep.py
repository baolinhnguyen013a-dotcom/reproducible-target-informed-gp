#!/usr/bin/env python3
"""Nested B=30..80 WPT sweep for cEI and the 5%-per-source mixture.

Each standalone budget is B = 6 initial + N_seq active + 12 confirmation
oracle calls. Active histories are nested within a policy. For a matched seed,
both policies use the same initial design, sequential candidate streams, and
budget-specific confirmation candidate pool. GP and neural components within
the mixed policy are always fitted to the same physical observations.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from run_wpt_8d_mixed_benchmark import (
    EPSILON, N_CONFIRM, N_INITIAL, NOMINAL_ROTATION, NOMINAL_TRANSLATION,
    TARGET_M, evaluate, fit_nn, gp_components, gps, neural_signals, normalize,
    sample_points, unit,
)

POLICIES = {
    "cei": (1., 0., 0., 0.),
    "cei_all05": (.85, .05, .05, .05),
}


def evaluate_budget(policy, weights, seed, budget, x_raw, y, models,
                    candidate_count, started):
    # Reuse the original B=50 protocol's seeded confirmation candidate pool at
    # every budget. The fitted GP and selected 12 points still change with B,
    # while the B=50 snapshot exactly reproduces the standalone experiment.
    confirmation = sample_points(seed*100003 + 999983,
                                 max(candidate_count*2, 2000))
    _, _, probability_feasible, mu_r = gp_components(
        models, normalize(confirmation), y)
    score = probability_feasible*np.exp(-unit(mu_r))
    chosen = np.argsort(-score, kind="stable")[:N_CONFIRM]
    confirm_y = np.asarray([evaluate(point) for point in confirmation[chosen]])
    all_x = np.vstack([x_raw, confirmation[chosen]])
    all_y = np.vstack([y, confirm_y])
    feasible = np.where(np.abs(all_y[:, 0]-TARGET_M) <= EPSILON*TARGET_M)[0]
    qualified = bool(len(feasible))
    decision = (int(feasible[np.argmin(all_y[feasible, 1])]) if qualified
                else int(np.argmin(np.abs(all_y[:, 0]-TARGET_M))))
    return {
        "policy": policy, "w_cei": weights[0], "w_gp": weights[1],
        "w_disagreement": weights[2], "w_ntk": weights[3],
        "seed": seed, "budget": budget, "n_initial": N_INITIAL,
        "n_sequential": budget-N_INITIAL-N_CONFIRM,
        "n_confirmation": N_CONFIRM, "qualified": qualified,
        "best_M_nH": float(all_y[decision, 0]),
        "best_R_ohm": float(all_y[decision, 1]),
        "relative_target_error_pct": float(abs(all_y[decision, 0]-TARGET_M)/TARGET_M*100),
        "decision_stage": ("initial" if decision < N_INITIAL else
                           ("sequential" if decision < len(x_raw) else "confirmation")),
        "design": json.dumps(all_x[decision].tolist()),
        "active_designs": json.dumps(x_raw.tolist()),
        "runtime_sec": time.monotonic()-started,
    }


def run_trajectory(task):
    policy, weights, seed, budgets, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(point) for point in x_raw])
    max_active = max(budgets)-N_CONFIRM
    snapshots = []
    step = 0
    while True:
        x = normalize(x_raw)
        models = gps(x, y, seed+step)
        budget = len(x_raw)+N_CONFIRM
        if budget in budgets:
            snapshots.append(evaluate_budget(policy, weights, seed, budget,
                                             x_raw, y, models,
                                             candidate_count, started))
        if len(x_raw) >= max_active:
            break
        candidates = sample_points(seed*100003 + step*1009 + 17, candidate_count)
        c = normalize(candidates)
        cei, gp_global, _, _ = gp_components(models, c, y)
        score = weights[0]*cei + weights[1]*gp_global
        if weights[2] > 0 or weights[3] > 0:
            disagreement, ntk = [], []
            for output in range(2):
                model = fit_nn(x, y[:, output],
                               seed*1000003 + step*37 + output*100003,
                               epochs=epochs)
                d, n = neural_signals(model, c, x)
                disagreement.append(d); ntk.append(n)
            q_dis = .5*unit(disagreement[0]) + .5*unit(disagreement[1])
            q_ntk = .5*unit(ntk[0]) + .5*unit(ntk[1])
            score += weights[2]*unit(q_dis) + weights[3]*unit(q_ntk)
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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    budgets = tuple(range(args.budget_min, args.budget_max+1))
    if budgets[0] <= N_INITIAL+N_CONFIRM:
        raise ValueError("minimum budget must include at least one sequential acquisition")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(policy, weights, seed, budgets, args.candidate_count, args.epochs)
             for policy, weights in POLICIES.items()
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_trajectory, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"checkpoint {i}/{len(tasks)} trajectories", flush=True)
    rows.sort(key=lambda row: (row["policy"], row["seed"], row["budget"]))
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for policy, weights in POLICIES.items():
        for budget in budgets:
            group = [row for row in rows if row["policy"] == policy and row["budget"] == budget]
            feasible = [row for row in group if row["qualified"]]
            summary.append({
                "policy": policy, "weights": weights, "budget": budget,
                "cases": len(group), "qualification_rate": len(feasible)/len(group),
                "mean_R_qualified_ohm": float(np.mean([row["best_R_ohm"] for row in feasible])) if feasible else None,
                "std_R_qualified_ohm": float(np.std([row["best_R_ohm"] for row in feasible])) if feasible else None,
                "mean_target_error_pct": float(np.mean([row["relative_target_error_pct"] for row in group])),
            })
    manifest = {
        "task": "minimize resistance at M=1500 nH within 2.5 percent",
        "dimensions": 8,
        "fixed_pose": {"translation_mm": NOMINAL_TRANSLATION,
                       "rotation_rad": NOMINAL_ROTATION},
        "budgets": list(budgets), "n_initial": N_INITIAL,
        "n_confirmation": N_CONFIRM,
        "nested_active_histories": True,
        "confirmation_is_standalone_at_each_budget": True,
        "confirmation_candidate_pool_shared_across_budgets": True,
        "matched_initial_and_candidate_streams_across_policies": True,
        "same_samples_within_mixed_trajectory": True,
        "candidate_count": args.candidate_count, "epochs": args.epochs,
        "ensemble_members": 10,
        "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
        "policies": POLICIES,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
