#!/usr/bin/env python3
"""Test uncertainty fusion *inside* constrained expected improvement.

The GP predictive means are retained.  For each response, neural-ensemble
disagreement and NTK posterior standard deviation are quantile-calibrated to
the physical-unit scale of the GP posterior standard deviation over the same
candidate pool.  Their convex mixture replaces sigma in both probability of
feasibility and resistance expected improvement.

All components within a trajectory use one shared physical history.  The
terminal 12-point confirmation remains GP-only, matching the main paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr

from run_wpt_8d_mixed_benchmark import (
    EPSILON, HIGH, LOWER_M, LOW, N_CONFIRM, N_INITIAL, N_SEQUENTIAL,
    TARGET_M, UPPER_M, evaluate, fit_nn, gp_components, gps, neural_signals,
    normalize, sample_points, unit,
)


POLICIES = {
    "sigma_gp_only": (1.0, 0.0, 0.0),
    "sigma_90_05_05": (0.90, 0.05, 0.05),
    "sigma_80_10_10": (0.80, 0.10, 0.10),
    "sigma_80_15_05": (0.80, 0.15, 0.05),
    "sigma_80_05_15": (0.80, 0.05, 0.15),
    "sigma_60_20_20": (0.60, 0.20, 0.20),
    "sigma_equal": (1/3, 1/3, 1/3),
}


def quantile_calibrate(auxiliary, gp_sigma):
    """Map auxiliary ranks onto the empirical GP-sigma distribution.

    This preserves the auxiliary ordering while giving it the response units
    and marginal scale required by Gaussian feasibility and EI equations.
    Stable sorting makes ties deterministic.
    """
    auxiliary = np.asarray(auxiliary, float)
    gp_sigma = np.maximum(np.asarray(gp_sigma, float), 1e-12)
    order = np.argsort(auxiliary, kind="stable")
    calibrated = np.empty_like(gp_sigma)
    calibrated[order] = np.sort(gp_sigma, kind="stable")
    return calibrated


def effective_sigma(gp_sigma, disagreement, ntk, weights):
    w_gp, w_dis, w_ntk = weights
    result = w_gp*np.asarray(gp_sigma, float)
    if w_dis:
        result = result + w_dis*quantile_calibrate(disagreement, gp_sigma)
    if w_ntk:
        result = result + w_ntk*quantile_calibrate(ntk, gp_sigma)
    return np.maximum(result, 1e-12)


def sigma_mixture_acquisition(models, candidates, observed, x_train, y_train,
                              weights, fit_seed, epochs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(v).all() for v in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite GP prediction in sigma-mixture acquisition")
    sd_m, sd_r = np.maximum(sd_m, 1e-12), np.maximum(sd_r, 1e-12)

    if weights[1] or weights[2]:
        dis, ntk = [], []
        for output in range(2):
            model = fit_nn(x_train, y_train[:, output],
                           fit_seed + output*100003, epochs=epochs)
            d, n = neural_signals(model, candidates, x_train)
            dis.append(d)
            ntk.append(n)
        eff_m = effective_sigma(sd_m, dis[0], ntk[0], weights)
        eff_r = effective_sigma(sd_r, dis[1], ntk[1], weights)
    else:
        eff_m, eff_r = sd_m, sd_r

    pf = ndtr((UPPER_M-mu_m)/eff_m)-ndtr((LOWER_M-mu_m)/eff_m)
    feasible = observed[(observed[:, 0] >= LOWER_M) &
                        (observed[:, 0] <= UPPER_M)]
    if len(feasible):
        incumbent = float(feasible[:, 1].min())
        z = (incumbent-mu_r)/eff_r
        ei = ((incumbent-mu_r)*ndtr(z)
              + eff_r*np.exp(-.5*z*z)/math.sqrt(2*math.pi))
        score = pf*np.maximum(ei, 0)/incumbent
    else:
        score = pf
    return unit(score)


def run_case(task):
    policy, weights, seed, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(p) for p in x_raw])

    for step in range(N_SEQUENTIAL):
        candidates = sample_points(seed*100003+step*1009+17, candidate_count)
        x_train, c = normalize(x_raw), normalize(candidates)
        models = gps(x_train, y, seed+step)
        fit_seed = seed*1000003+step*37
        score = sigma_mixture_acquisition(
            models, c, y, x_train, y, weights, fit_seed, epochs)
        chosen = candidates[int(np.argmax(score))]
        x_raw = np.vstack([x_raw, chosen])
        y = np.vstack([y, evaluate(chosen)])

    active_designs = x_raw.tolist()
    confirm_candidates = sample_points(seed*100003+999983,
                                      max(candidate_count*2, 2000))
    models = gps(normalize(x_raw), y, seed+N_SEQUENTIAL)
    _, _, pf, mu_r = gp_components(models, normalize(confirm_candidates), y)
    confirmation_score = pf*np.exp(-unit(mu_r))
    chosen = np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
    x_raw = np.vstack([x_raw, confirm_candidates[chosen]])
    y = np.vstack([y, np.asarray([evaluate(p) for p in confirm_candidates[chosen]])])

    feasible_ids = np.where((y[:, 0] >= LOWER_M) & (y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible_ids))
    decision = (int(feasible_ids[np.argmin(y[feasible_ids, 1])]) if qualified
                else int(np.argmin(np.abs(y[:, 0]-TARGET_M))))
    return {
        "policy": policy, "w_gp_sigma": weights[0],
        "w_disagreement": weights[1], "w_ntk": weights[2],
        "seed": seed, "budget": len(y), "n_initial": N_INITIAL,
        "n_sequential": N_SEQUENTIAL, "n_confirmation": N_CONFIRM,
        "qualified": qualified, "best_M_nH": float(y[decision, 0]),
        "best_R_ohm": float(y[decision, 1]),
        "relative_target_error_pct": float(abs(y[decision, 0]-TARGET_M)/TARGET_M*100),
        "decision_stage": ("initial" if decision < N_INITIAL else
                           "sequential" if decision < N_INITIAL+N_SEQUENTIAL
                           else "confirmation"),
        "design": json.dumps(x_raw[decision].tolist()),
        "active_designs": json.dumps(active_designs),
        "runtime_sec": time.monotonic()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=34261011)
    parser.add_argument("--candidate-count", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES),
                        default=tuple(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = {name: POLICIES[name] for name in args.policies}
    tasks = [(name, weights, seed, args.candidate_count, args.epochs)
             for name, weights in selected.items()
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            print(f"checkpoint {i}/{len(tasks)}", flush=True)
    rows.sort(key=lambda row: (row["policy"], row["seed"]))
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = []
    for name, weights in selected.items():
        group = [row for row in rows if row["policy"] == name]
        qualified = [row for row in group if row["qualified"]]
        summary.append({
            "policy": name, "sigma_weights_gp_disagreement_ntk": weights,
            "cases": len(group), "qualification_rate": len(qualified)/len(group),
            "mean_R_qualified_ohm": (float(np.mean([r["best_R_ohm"] for r in qualified]))
                                      if qualified else None),
            "std_R_qualified_ohm": (float(np.std([r["best_R_ohm"] for r in qualified]))
                                     if qualified else None),
            "mean_target_error_pct": float(np.mean([r["relative_target_error_pct"] for r in group])),
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in group])),
        })
    manifest = {
        "method": "replace GP sigma in PoF and EI with a quantile-calibrated convex mixture",
        "sigma_components": ["GP posterior standard deviation",
                             "neural ensemble disagreement", "NTK posterior standard deviation"],
        "calibration": "stable empirical quantile mapping to the corresponding GP-sigma distribution",
        "gp_predictive_means_retained": True,
        "same_samples_within_trajectory": True,
        "confirmation": "12-point GP-only terminal confirmation",
        "dimensions": 8, "target_M_nH": TARGET_M,
        "relative_tolerance": EPSILON,
        "budget": {"initial": N_INITIAL, "sequential": N_SEQUENTIAL,
                   "confirmation": N_CONFIRM, "total": N_INITIAL+N_SEQUENTIAL+N_CONFIRM},
        "candidate_count": args.candidate_count, "ensemble_members": 10,
        "epochs": args.epochs,
        "seeds": list(range(args.seed_start, args.seed_start+args.seeds)),
        "policies": selected,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
