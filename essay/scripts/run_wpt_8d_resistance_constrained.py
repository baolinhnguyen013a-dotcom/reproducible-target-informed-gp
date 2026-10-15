#!/usr/bin/env python3
"""Application-oriented 8-D WPT design: maximize M subject to R <= R_max.

The physical accounting and common-sample rule match the target-inductance
experiment: B = 6 initial + 32 sequential + 12 GP-only confirmation calls.
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
    N_CONFIRM, N_INITIAL, N_SEQUENTIAL, NOMINAL_ROTATION,
    NOMINAL_TRANSLATION, POLICIES, evaluate, gps, neural_signals, normalize,
    sample_points, unit,
)

R_MAX_OHM = 0.35


def acquisition(models, candidates, observed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(v).all() for v in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite WPT GP prediction")
    sd_m, sd_r = np.maximum(sd_m, 1e-12), np.maximum(sd_r, 1e-12)
    probability_feasible = ndtr((R_MAX_OHM - mu_r) / sd_r)
    feasible = observed[observed[:, 1] <= R_MAX_OHM]
    if len(feasible):
        incumbent = float(feasible[:, 0].max())
        z = (mu_m - incumbent) / sd_m
        improvement = ((mu_m - incumbent) * ndtr(z)
                       + sd_m * np.exp(-.5*z*z) / math.sqrt(2*math.pi))
        cei = probability_feasible * np.maximum(improvement, 0) / max(incumbent, 1e-12)
    else:
        cei = probability_feasible
    gp_global = .5*unit(sd_m) + .5*unit(sd_r)
    return unit(cei), unit(gp_global), probability_feasible, mu_m


def run_case(task):
    policy, weights, seed, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(p) for p in x_raw])
    for step in range(N_SEQUENTIAL):
        candidates = sample_points(seed*100003 + step*1009 + 17, candidate_count)
        x, c = normalize(x_raw), normalize(candidates)
        models = gps(x, y, seed+step)
        cei, gp_global, _, _ = acquisition(models, c, y)
        score = weights[0]*cei + weights[1]*gp_global
        if weights[2] > 0 or weights[3] > 0:
            disagreement, ntk = [], []
            for output in range(2):
                from run_wpt_8d_mixed_benchmark import fit_nn
                model = fit_nn(x, y[:, output],
                               seed*1000003 + step*37 + output*100003,
                               epochs=epochs)
                d, n = neural_signals(model, c, x)
                disagreement.append(d)
                ntk.append(n)
            q_dis = .5*unit(disagreement[0]) + .5*unit(disagreement[1])
            q_ntk = .5*unit(ntk[0]) + .5*unit(ntk[1])
            score += weights[2]*unit(q_dis) + weights[3]*unit(q_ntk)
        chosen = candidates[int(np.argmax(score))]
        x_raw = np.vstack([x_raw, chosen])
        y = np.vstack([y, evaluate(chosen)])

    confirm = sample_points(seed*100003 + 999983, max(candidate_count*2, 2000))
    models = gps(normalize(x_raw), y, seed+N_SEQUENTIAL)
    _, _, probability_feasible, mu_m = acquisition(models, normalize(confirm), y)
    confirmation_score = probability_feasible * np.exp(unit(mu_m))
    chosen = np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
    x_raw = np.vstack([x_raw, confirm[chosen]])
    y = np.vstack([y, np.asarray([evaluate(p) for p in confirm[chosen]])])

    feasible_ids = np.where(y[:, 1] <= R_MAX_OHM)[0]
    qualified = bool(len(feasible_ids))
    decision = (int(feasible_ids[np.argmax(y[feasible_ids, 0])]) if qualified
                else int(np.argmin(y[:, 1])))
    return {
        "policy": policy, "seed": seed,
        "budget": N_INITIAL+N_SEQUENTIAL+N_CONFIRM,
        "n_initial": N_INITIAL, "n_sequential": N_SEQUENTIAL,
        "n_confirmation": N_CONFIRM, "resistance_cap_ohm": R_MAX_OHM,
        "qualified": qualified, "best_M_nH": float(y[decision, 0]),
        "best_R_ohm": float(y[decision, 1]),
        "decision_stage": ("initial" if decision < N_INITIAL else
                           ("sequential" if decision < N_INITIAL+N_SEQUENTIAL
                            else "confirmation")),
        "design": json.dumps(x_raw[decision].tolist()),
        "runtime_sec": time.monotonic()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=36261011)
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
    fields = list(rows[0])
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for name, weights in selected.items():
        group = [r for r in rows if r["policy"] == name]
        feasible = [r for r in group if r["qualified"]]
        summary.append({
            "policy": name, "weights": weights, "cases": len(group),
            "qualification_rate": len(feasible)/len(group),
            "mean_best_M_nH": float(np.mean([r["best_M_nH"] for r in feasible])) if feasible else None,
            "std_best_M_nH": float(np.std([r["best_M_nH"] for r in feasible])) if feasible else None,
            "mean_R_ohm": float(np.mean([r["best_R_ohm"] for r in feasible])) if feasible else None,
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in group])),
        })
    manifest = {
        "task": "maximize mutual inductance subject to a fixed resistance cap",
        "source_model": "Asadi and Abazari (2025), doi:10.1049/elp2.70015",
        "dimensions": 8,
        "fixed_pose": {"translation_mm": NOMINAL_TRANSLATION,
                       "rotation_rad": NOMINAL_ROTATION},
        "resistance_cap_ohm": R_MAX_OHM,
        "budget": {"initial": N_INITIAL, "sequential": N_SEQUENTIAL,
                   "confirmation": N_CONFIRM, "total": 50},
        "same_samples_within_trajectory": True,
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
