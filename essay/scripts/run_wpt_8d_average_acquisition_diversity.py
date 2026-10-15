#!/usr/bin/env python3
"""Average pure cEI and effective-sigma cEI, then audit confirmation diversity.

All 32 active steps maximize the mean of two candidate-wise normalized scores:
ordinary GP-cEI and effective-sigma cEI with sigma weights (.8,.1,.1).  Both
use the same observations.  From the resulting common active history, five
12-point GP-only batch confirmation rules are compared.
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
    EPSILON, LOWER_M, N_CONFIRM, N_INITIAL, N_SEQUENTIAL, TARGET_M, UPPER_M,
    evaluate, gp_components, gps, normalize, sample_points, unit,
)
from run_wpt_8d_sigma_mixture import sigma_mixture_acquisition

SIGMA_WEIGHTS = (0.8, 0.1, 0.1)
CONFIRMATION_POLICIES = (
    "original_top12",
    "soft_distance_batch_p0p10",
    "soft_distance_batch_p0p20",
    "soft_distance_batch",
    "soft_distance_active_and_batch",
    "hard_minsep_0p30",
    "hard_minsep_0p40",
)


def pairwise_min_distance(points):
    if len(points) < 2:
        return None
    delta = points[:, None, :]-points[None, :, :]
    distance = np.sqrt(np.sum(delta*delta, axis=2))
    distance[np.eye(len(points), dtype=bool)] = np.inf
    return float(distance.min())


def greedy_soft(score, candidates, count, fixed=None, exponent=1.0):
    score = unit(score)
    selected = []
    fixed = np.empty((0, candidates.shape[1])) if fixed is None else np.asarray(fixed)
    available = np.ones(len(candidates), dtype=bool)
    for _ in range(count):
        reference = np.vstack([fixed, candidates[selected]]) if selected or len(fixed) else None
        if reference is None or not len(reference):
            penalized = score.copy()
        else:
            delta = candidates[:, None, :]-reference[None, :, :]
            nearest = np.sqrt(np.sum(delta*delta, axis=2)).min(axis=1)
            penalized = score*unit(nearest)**exponent
        penalized[~available] = -np.inf
        chosen = int(np.argmax(penalized))
        selected.append(chosen); available[chosen] = False
    return np.asarray(selected, dtype=int)


def greedy_hard(score, candidates, count, minsep):
    order = np.argsort(-score, kind="stable")
    selected = []
    for candidate in order:
        if not selected:
            selected.append(int(candidate))
        else:
            distance = np.sqrt(np.sum((candidates[selected]-candidates[candidate])**2, axis=1))
            if float(distance.min()) >= minsep:
                selected.append(int(candidate))
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(int(i) for i in order if int(i) not in selected_set
                        and len(selected) < count)
    return np.asarray(selected, dtype=int)


def confirmation_ids(policy, score, candidates, active):
    if policy == "original_top12":
        return np.argsort(-score, kind="stable")[:N_CONFIRM]
    if policy == "soft_distance_batch":
        return greedy_soft(score, candidates, N_CONFIRM)
    if policy == "soft_distance_batch_p0p10":
        return greedy_soft(score, candidates, N_CONFIRM, exponent=.10)
    if policy == "soft_distance_batch_p0p20":
        return greedy_soft(score, candidates, N_CONFIRM, exponent=.20)
    if policy == "soft_distance_active_and_batch":
        return greedy_soft(score, candidates, N_CONFIRM, fixed=active)
    if policy == "hard_minsep_0p30":
        return greedy_hard(score, candidates, N_CONFIRM, .30)
    if policy == "hard_minsep_0p40":
        return greedy_hard(score, candidates, N_CONFIRM, .40)
    raise KeyError(policy)


def evaluate_confirmation(policy, seed, x_active, y_active, candidates,
                          candidate_y, score, started):
    c_norm, active_norm = normalize(candidates), normalize(x_active)
    chosen = confirmation_ids(policy, score, c_norm, active_norm)
    x = np.vstack([x_active, candidates[chosen]])
    y = np.vstack([y_active, candidate_y[chosen]])
    feasible = np.where((y[:, 0] >= LOWER_M) & (y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible))
    decision = (int(feasible[np.argmin(y[feasible, 1])]) if qualified
                else int(np.argmin(np.abs(y[:, 0]-TARGET_M))))
    selected_norm = c_norm[chosen]
    active_distance = np.sqrt(np.sum(
        (selected_norm[:, None, :]-active_norm[None, :, :])**2, axis=2)).min(axis=1)
    return {
        "confirmation_policy": policy, "seed": seed, "budget": len(y),
        "n_initial": N_INITIAL, "n_sequential": N_SEQUENTIAL,
        "n_confirmation": N_CONFIRM, "qualified": qualified,
        "best_M_nH": float(y[decision, 0]), "best_R_ohm": float(y[decision, 1]),
        "relative_target_error_pct": float(abs(y[decision, 0]-TARGET_M)/TARGET_M*100),
        "decision_stage": ("initial" if decision < N_INITIAL else
                           "sequential" if decision < N_INITIAL+N_SEQUENTIAL
                           else "confirmation"),
        "design": json.dumps(x[decision].tolist()),
        "confirmation_designs": json.dumps(candidates[chosen].tolist()),
        "confirmation_pair_min_distance": pairwise_min_distance(selected_norm),
        "confirmation_median_nearest_active_distance": float(np.median(active_distance)),
        "confirmation_min_active_distance": float(active_distance.min()),
        "runtime_sec": time.monotonic()-started,
    }


def run_case(task):
    seed, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(point) for point in x_raw])
    for step in range(N_SEQUENTIAL):
        candidates = sample_points(seed*100003+step*1009+17, candidate_count)
        x, c = normalize(x_raw), normalize(candidates)
        models = gps(x, y, seed+step)
        cei, _, _, _ = gp_components(models, c, y)
        sigma_cei = sigma_mixture_acquisition(
            models, c, y, x, y, SIGMA_WEIGHTS,
            seed*1000003+step*37, epochs)
        joint = .5*unit(cei)+.5*unit(sigma_cei)
        selected = candidates[int(np.argmax(joint))]
        x_raw = np.vstack([x_raw, selected])
        y = np.vstack([y, evaluate(selected)])

    confirmation = sample_points(seed*100003+999983,
                                 max(candidate_count*2, 2000))
    confirmation_y = np.asarray([evaluate(point) for point in confirmation])
    models = gps(normalize(x_raw), y, seed+N_SEQUENTIAL)
    _, _, pf, mu_r = gp_components(models, normalize(confirmation), y)
    original_score = pf*np.exp(-unit(mu_r))
    rows = [evaluate_confirmation(policy, seed, x_raw, y, confirmation,
                                  confirmation_y, original_score, started)
            for policy in CONFIRMATION_POLICIES]
    for row in rows:
        row["active_designs"] = json.dumps(x_raw.tolist())
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=34261011)
    parser.add_argument("--candidate-count", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, args.candidate_count, args.epochs)
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"checkpoint {i}/{len(tasks)} trajectories", flush=True)
    rows.sort(key=lambda row: (row["confirmation_policy"], row["seed"]))
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for policy in CONFIRMATION_POLICIES:
        group = [row for row in rows if row["confirmation_policy"] == policy]
        qualified = [row for row in group if row["qualified"]]
        summary.append({
            "confirmation_policy": policy, "cases": len(group),
            "qualification_rate": len(qualified)/len(group),
            "mean_R_qualified_ohm": float(np.mean([r["best_R_ohm"] for r in qualified])),
            "std_R_qualified_ohm": float(np.std([r["best_R_ohm"] for r in qualified])),
            "mean_target_error_pct": float(np.mean([r["relative_target_error_pct"] for r in group])),
            "mean_pair_min_distance": float(np.mean([r["confirmation_pair_min_distance"] for r in group])),
            "mean_min_active_distance": float(np.mean([r["confirmation_min_active_distance"] for r in group])),
        })
    manifest = {
        "active_acquisition": "0.5 normalized pure cEI + 0.5 normalized effective-sigma cEI",
        "effective_sigma_weights_gp_disagreement_ntk": SIGMA_WEIGHTS,
        "same_samples_for_both_acquisitions": True,
        "confirmation_base_score": "p_feasible * exp(-unit(mu_R))",
        "distance_space": "Euclidean distance in normalized 8-D design coordinates",
        "budget": {"initial": N_INITIAL, "sequential": N_SEQUENTIAL,
                   "confirmation": N_CONFIRM, "total": 50},
        "candidate_count_active": args.candidate_count,
        "candidate_count_confirmation": max(args.candidate_count*2, 2000),
        "ensemble_members": 10, "epochs": args.epochs,
        "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
        "confirmation_policies": CONFIRMATION_POLICIES,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
