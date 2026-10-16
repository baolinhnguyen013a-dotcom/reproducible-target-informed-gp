#!/usr/bin/env python3
"""Test deterministic-start cEI with an automatic stopping rule.

The default uses one deterministic center, the 75/25 feasibility-discovery
rule, ordinary cEI, a 1% relative-improvement stopping threshold, deterministic
five-start GP hyperparameter optimization, and a terminal 10-point simultaneous
GP confirmation batch.  Legacy stochastic GP optimization and the experimental
cosine-annealed xi-cEI policy remain available as command-line controls.  The
17-call model-building cap keeps the total safety budget at 27 evaluations.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.special import ndtr

import run_center_start_cei_lr as base

TARGETS = (1, 2, 3, 4, 5, 11, 20, 30, 39, 49, 58, 68, 77, 87)
POLICIES = ("ordinary_stop", "cosine_xi_stop")
DELTA_STOP = 1e-2
PATIENCE = 3
MAX_MODEL_CALLS = 17
MIN_MODEL_CALLS = 2 * base.X.shape[1] + 1
N_CONFIRM = 10
PF_WEIGHT = 0.75
OUT = base.ROOT/"essay"/"data"/"center_adaptive_stopping_default"
ROWS = OUT/"results.csv"
FIELDS = (
    "policy", "seed", "target_mH", "qualified", "exact", "bounded_loss_u",
    "relative_resistance_regret", "decision_T", "decision_n",
    "decision_L_mH", "decision_R_ohm", "decision_stage", "oracle_calls",
    "n_model_building", "n_confirmation", "stopped_naturally",
    "first_feasible_evaluation", "final_rho", "final_cosine_weight",
    "model_building_ids", "confirmation_ids",
)


def p_feasible(target, mean_l, std_l, indices):
    sigma = np.maximum(std_l[indices], 1e-12)
    lower, upper = target*(1-base.EPSILON), target*(1+base.EPSILON)
    return ndtr((upper-mean_l[indices])/sigma)-ndtr((lower-mean_l[indices])/sigma)


def run_one(task):
    policy, seed, target, delta_stop, gp_starts = task
    selected = [base.CENTER_ID]
    selected_set = {base.CENTER_ID}
    first_feasible = 1 if base.is_feasible(base.CENTER_ID, target) else None
    rho_reference = None
    xi_reference = None
    monotone_rho = None
    stop_streak = 0
    final_rho = None
    final_cosine = None
    stopped = False

    while len(selected) < MAX_MODEL_CALLS:
        ids = np.asarray(selected, int)
        mean_l, std_l = base.fit(base.L_ALL, ids, seed,
                                 deterministic_starts=gp_starts)
        mean_r, std_r = base.fit(base.R_ALL, ids, seed,
                                 deterministic_starts=gp_starts)
        unseen = np.asarray([i for i in range(len(base.POINTS))
                             if i not in selected_set], int)
        pf = p_feasible(target, mean_l, std_l, unseen)
        feasible_ids = [i for i in selected if base.is_feasible(i, target)]
        if not feasible_ids:
            score = PF_WEIGHT*base.unit(pf)+(1-PF_WEIGHT)*base.unit(std_l[unseen])
            final_rho, final_cosine = None, None
        else:
            incumbent = min(base.R_ALL[i] for i in feasible_ids)
            sigma_r = np.maximum(std_r[unseen], 1e-12)
            improvement = incumbent-mean_r[unseen]
            z0 = improvement/sigma_r
            ei0 = improvement*ndtr(z0)+sigma_r*np.exp(-.5*z0*z0)/np.sqrt(2*np.pi)
            cei0 = pf*np.maximum(ei0, 0)/incumbent
            rho = float(np.max(cei0))
            final_rho = rho
            if rho_reference is None:
                rho_reference = max(rho, delta_stop*(1+1e-12))
                relevant = sigma_r[pf >= .5*np.max(pf)]
                xi_reference = float(np.median(relevant if len(relevant) else sigma_r))
                monotone_rho = rho_reference
            monotone_rho = min(monotone_rho, max(rho, 1e-300))
            if rho_reference <= delta_stop:
                progress = 1.0
            else:
                progress = np.clip(
                    np.log(rho_reference/monotone_rho)
                    / np.log(rho_reference/delta_stop), 0, 1)
            cosine = .5*(1+np.cos(np.pi*progress))
            final_cosine = float(cosine)
            if rho <= delta_stop:
                stop_streak += 1
            else:
                stop_streak = 0
            if len(selected) >= MIN_MODEL_CALLS and stop_streak >= PATIENCE:
                stopped = True
                break
            xi = xi_reference*cosine if policy == "cosine_xi_stop" else 0.0
            shifted = improvement-xi
            z = shifted/sigma_r
            ei = shifted*ndtr(z)+sigma_r*np.exp(-.5*z*z)/np.sqrt(2*np.pi)
            score = pf*np.maximum(ei, 0)/incumbent
        chosen = int(unseen[int(np.argmax(score))])
        selected.append(chosen); selected_set.add(chosen)
        if first_feasible is None and base.is_feasible(chosen, target):
            first_feasible = len(selected)

    # Stopping ends adaptive model building, not physical evaluation.  Refit
    # once to the stopped history and purchase ten simultaneous confirmations
    # with the same GP-only terminal rule used by the earlier experiments.
    ids = np.asarray(selected, int)
    mean_l, std_l = base.fit(base.L_ALL, ids, seed, final=True,
                             deterministic_starts=gp_starts)
    mean_r, _ = base.fit(base.R_ALL, ids, seed, final=True,
                         deterministic_starts=gp_starts)
    unseen = np.asarray([i for i in range(len(base.POINTS))
                         if i not in selected_set], int)
    pf = p_feasible(target, mean_l, std_l, unseen)
    confirmation_score = pf*np.exp(-base.unit(mean_r[unseen]))
    confirmation = [int(i) for i in
                    unseen[np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]]]
    pool = selected + confirmation

    feasible_pool = [i for i in pool if base.is_feasible(i, target)]
    qualified = bool(feasible_pool)
    if qualified:
        decision = min(feasible_pool, key=lambda i: base.R_ALL[i])
        domain = np.flatnonzero(np.abs(base.L_ALL-target)/target <= base.EPSILON)
        optimum = int(domain[np.argmin(base.R_ALL[domain])])
        regret = float(base.R_ALL[decision]/base.R_ALL[optimum]-1)
        loss, exact = regret/(1+regret), regret <= 1e-12
    else:
        decision = min(pool, key=lambda i: abs(base.L_ALL[i]-target)/target)
        regret, loss, exact = None, 1.0, False
    layers, turns = base.POINTS[decision]
    return {
        "policy": policy, "seed": seed, "target_mH": target,
        "qualified": qualified, "exact": exact, "bounded_loss_u": loss,
        "relative_resistance_regret": regret, "decision_T": layers,
        "decision_n": turns, "decision_L_mH": float(base.L_ALL[decision]),
        "decision_R_ohm": float(base.R_ALL[decision]),
        "decision_stage": ("model_building" if decision in selected_set
                           else "confirmation"),
        "oracle_calls": len(pool), "n_model_building": len(selected),
        "n_confirmation": N_CONFIRM, "stopped_naturally": stopped,
        "first_feasible_evaluation": first_feasible, "final_rho": final_rho,
        "final_cosine_weight": final_cosine,
        "model_building_ids": json.dumps(selected),
        "confirmation_ids": json.dumps(confirmation),
    }


def read(path):
    if not path.exists(): return []
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def summarize(rows, policies):
    baseline = {(int(r["seed"]), int(float(r["target_mH"]))): r for r in read(
        base.ROOT/"results"/"coil_2d_relative_tolerance"/"cei_budget27"/"cei_results.csv")}
    output = []
    for policy in policies:
        group = [r for r in rows if r["policy"] == policy]
        values = np.asarray([float(r["bounded_loss_u"]) for r in group])
        calls = np.asarray([int(r["oracle_calls"]) for r in group])
        base_values = np.asarray([float(baseline[int(r["seed"]), int(r["target_mH"])]["bounded_loss_u"])
                                  for r in group])
        output.append({
            "policy": policy, "cases": len(group),
            "qualification_rate": float(np.mean([r["qualified"] == "True" for r in group])),
            "exact_optimum_cases": int(np.sum(values <= 1e-12)),
            "exact_optimum_rate": float(np.mean(values <= 1e-12)),
            "mean_bounded_loss_u": float(values.mean()),
            "mean_oracle_calls": float(calls.mean()),
            "median_oracle_calls": float(np.median(calls)),
            "maximum_oracle_calls": int(calls.max()),
            "natural_stop_cases": int(np.sum([r["stopped_naturally"] == "True" for r in group])),
            "wins_ties_losses_vs_random_six_B27": [int(np.sum(values < base_values)),
                                                    int(np.sum(values == base_values)),
                                                    int(np.sum(values > base_values))],
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta-stop", type=float, default=DELTA_STOP)
    parser.add_argument("--output-name", default=OUT.name)
    parser.add_argument("--policies", nargs="+", choices=POLICIES,
                        default=["ordinary_stop"])
    parser.add_argument("--deterministic-gp-starts", type=int, default=5,
                        help="total fixed optimizer starts; zero keeps seeded random restarts")
    args = parser.parse_args()
    if not 0 < args.delta_stop < 1:
        parser.error("--delta-stop must lie strictly between zero and one")
    if args.deterministic_gp_starts < 0:
        parser.error("--deterministic-gp-starts cannot be negative")
    out = base.ROOT/"essay"/"data"/args.output_name
    rows_path = out/"results.csv"
    policies = tuple(args.policies)
    out.mkdir(parents=True, exist_ok=True)
    existing = read(rows_path)
    done = {(r["policy"], int(r["seed"]), int(r["target_mH"])) for r in existing}
    tasks = [(p, s, t, args.delta_stop, args.deterministic_gp_starts)
             for p in policies for s in base.SEEDS for t in TARGETS
             if (p, s, t) not in done]
    new = not rows_path.exists()
    with rows_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new: writer.writeheader()
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                writer.writerow(future.result()); stream.flush()
                if count % 25 == 0 or count == len(tasks):
                    print(f"checkpoint {count}/{len(tasks)}", flush=True)
    rows = read(rows_path)
    expected = len(policies)*len(base.SEEDS)*len(TARGETS)
    if len(rows) != expected: raise RuntimeError(f"incomplete: {len(rows)}/{expected}")
    report = {
        "targets_mH": list(TARGETS), "seeds": list(base.SEEDS),
        "center": list(base.POINTS[base.CENTER_ID]),
        "pre_feasibility_score": ".75*unit(pF)+.25*unit(sigma_L)",
        "delta_stop_relative_improvement": args.delta_stop,
        "stop_patience": PATIENCE,
        "minimum_model_building": MIN_MODEL_CALLS,
        "budget": {"maximum_model_building": MAX_MODEL_CALLS,
                   "confirmation": N_CONFIRM,
                   "maximum_total": MAX_MODEL_CALLS+N_CONFIRM},
        "cosine_progress": "log contraction of max ordinary-cEI / incumbent",
        "cosine_xi_reference": "median sigma_R where pF >= half its current maximum at first feasibility",
        "confirmation": "simultaneous top-10 by pF*exp(-unit(mu_R)), fitted to the stopped history",
        "gp_optimizer": (f"deterministic Halton multistart, K={args.deterministic_gp_starts}"
                         if args.deterministic_gp_starts else
                         "seeded random restarts (legacy control)"),
        "results": summarize(rows, policies),
    }
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    (out/"manifest.json").write_text(json.dumps({
        "status": "complete", "script": str(__file__), "policies": list(policies),
        "delta_stop_relative_improvement": args.delta_stop,
        "minimum_model_building": MIN_MODEL_CALLS,
        "deterministic_gp_starts": args.deterministic_gp_starts,
        "targets_mH": list(TARGETS), "seeds": list(base.SEEDS),
    }, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
