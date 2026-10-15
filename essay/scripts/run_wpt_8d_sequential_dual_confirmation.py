#!/usr/bin/env python3
"""Sequential 12-point confirmation using cEI and effective-sigma proposals.

The first 38 observations are the established pure-cEI active histories
(6 initial + 32 sequential). Confirmation points are then selected one at a
time. After every physical evaluation both acquisitions are refitted to the
same enlarged history. Hybrid policies allocate exactly six selections to
ordinary cEI and six to effective-sigma cEI (.8/.1/.1).
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
    LOWER_M, N_CONFIRM, N_INITIAL, N_SEQUENTIAL, TARGET_M, UPPER_M,
    evaluate, gp_components, gps, normalize, sample_points,
)
from run_wpt_8d_sigma_mixture import sigma_mixture_acquisition

ESSAY = Path(__file__).resolve().parents[1]
BASELINE = ESSAY/"data/wpt_8d_sigma_mixture_gp_control/results.csv"
SIGMA_WEIGHTS = (0.8, 0.1, 0.1)
POLICIES = (
    "sequential_cei_only",
    "sequential_sigma_only",
    "alternate_cei_first",
    "alternate_sigma_first",
    "confidence_gate_6_6",
)


def score_margin(score, available):
    values = np.asarray(score)[available]
    if len(values) < 2:
        return float("inf")
    top = np.partition(values, -2)[-2:]
    return float(top.max()-top.min())


def choose_method(policy, step, used_cei, used_sigma, margin_cei, margin_sigma):
    if policy == "sequential_cei_only":
        return "cei"
    if policy == "sequential_sigma_only":
        return "sigma"
    if policy == "alternate_cei_first":
        return "cei" if step % 2 == 0 else "sigma"
    if policy == "alternate_sigma_first":
        return "sigma" if step % 2 == 0 else "cei"
    if used_cei >= 6:
        return "sigma"
    if used_sigma >= 6:
        return "cei"
    return "cei" if margin_cei >= margin_sigma else "sigma"


def run_case(task):
    policy, seed, active_designs, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = np.asarray(active_designs, float)
    if len(x_raw) != N_INITIAL+N_SEQUENTIAL:
        raise ValueError("base history must contain 38 active observations")
    y = np.asarray([evaluate(point) for point in x_raw])
    candidates = sample_points(seed*100003+999983,
                               max(candidate_count*2, 2000))
    available = np.ones(len(candidates), dtype=bool)
    methods, margins, proposals_agree = [], [], []
    used_cei = used_sigma = 0
    for step in range(N_CONFIRM):
        x = normalize(x_raw)
        c = normalize(candidates)
        models = gps(x, y, seed+N_SEQUENTIAL+step)
        cei_score, _, _, _ = gp_components(models, c, y)
        need_sigma = policy != "sequential_cei_only"
        if need_sigma:
            sigma_score = sigma_mixture_acquisition(
                models, c, y, x, y, SIGMA_WEIGHTS,
                seed*1000003+(N_SEQUENTIAL+step)*37, epochs)
        else:
            sigma_score = np.zeros_like(cei_score)
        ids = np.where(available)[0]
        cei_proposal = int(ids[np.argmax(cei_score[ids])])
        sigma_proposal = (int(ids[np.argmax(sigma_score[ids])])
                          if need_sigma else cei_proposal)
        margin_cei = score_margin(cei_score, available)
        margin_sigma = score_margin(sigma_score, available) if need_sigma else 0.
        method = choose_method(policy, step, used_cei, used_sigma,
                               margin_cei, margin_sigma)
        chosen = cei_proposal if method == "cei" else sigma_proposal
        available[chosen] = False
        x_raw = np.vstack([x_raw, candidates[chosen]])
        y = np.vstack([y, evaluate(candidates[chosen])])
        methods.append(method)
        margins.append({"cei": margin_cei, "sigma": margin_sigma})
        proposals_agree.append(cei_proposal == sigma_proposal)
        used_cei += method == "cei"
        used_sigma += method == "sigma"

    feasible = np.where((y[:, 0] >= LOWER_M) & (y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible))
    decision = (int(feasible[np.argmin(y[feasible, 1])]) if qualified
                else int(np.argmin(np.abs(y[:, 0]-TARGET_M))))
    return {
        "policy": policy, "seed": seed, "budget": len(y),
        "n_initial": N_INITIAL, "n_sequential": N_SEQUENTIAL,
        "n_confirmation": N_CONFIRM, "confirmation_cei_count": used_cei,
        "confirmation_sigma_count": used_sigma, "qualified": qualified,
        "best_M_nH": float(y[decision, 0]), "best_R_ohm": float(y[decision, 1]),
        "relative_target_error_pct": float(abs(y[decision, 0]-TARGET_M)/TARGET_M*100),
        "decision_stage": ("initial" if decision < N_INITIAL else
                           "sequential" if decision < N_INITIAL+N_SEQUENTIAL
                           else "confirmation"),
        "design": json.dumps(x_raw[decision].tolist()),
        "confirmation_methods": json.dumps(methods),
        "proposal_margins": json.dumps(margins),
        "proposal_agreement_count": int(sum(proposals_agree)),
        "all_designs": json.dumps(x_raw.tolist()),
        "runtime_sec": time.monotonic()-started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=34261011)
    parser.add_argument("--candidate-count", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=POLICIES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = list(csv.DictReader(BASELINE.open(newline="")))
    histories = {int(row["seed"]): json.loads(row["active_designs"])
                 for row in base if row["policy"] == "sigma_gp_only"}
    selected_policies = tuple(args.policies)
    seeds = tuple(range(args.seed_start, args.seed_start+args.seeds))
    if any(seed not in histories for seed in seeds):
        raise ValueError("missing retained pure-cEI active history")
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [(policy, seed, histories[seed], args.candidate_count, args.epochs)
             for policy in selected_policies for seed in seeds]
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
    for policy in selected_policies:
        group = [row for row in rows if row["policy"] == policy]
        qualified = [row for row in group if row["qualified"]]
        summary.append({
            "policy": policy, "cases": len(group),
            "qualification_rate": len(qualified)/len(group),
            "mean_R_qualified_ohm": float(np.mean([r["best_R_ohm"] for r in qualified])),
            "std_R_qualified_ohm": float(np.std([r["best_R_ohm"] for r in qualified])),
            "mean_target_error_pct": float(np.mean([r["relative_target_error_pct"] for r in group])),
            "mean_cei_confirmations": float(np.mean([r["confirmation_cei_count"] for r in group])),
            "mean_sigma_confirmations": float(np.mean([r["confirmation_sigma_count"] for r in group])),
            "mean_proposal_agreement_count": float(np.mean([r["proposal_agreement_count"] for r in group])),
            "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in group])),
        })
    manifest = {
        "method": "sequential dual-proposal confirmation",
        "base_active_history": "pure cEI, 6 initial + 32 sequential",
        "sigma_weights_gp_disagreement_ntk": SIGMA_WEIGHTS,
        "confidence_gate": "larger normalized top-versus-runner-up acquisition margin, with exact 6/6 quota",
        "same_history_for_both_proposers": True,
        "refit_after_each_confirmation": True,
        "budget": {"initial": N_INITIAL, "sequential": N_SEQUENTIAL,
                   "confirmation": N_CONFIRM, "total": 50},
        "candidate_count": max(args.candidate_count*2, 2000),
        "ensemble_members": 10, "epochs": args.epochs,
        "seeds": seeds, "policies": selected_policies,
    }
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
