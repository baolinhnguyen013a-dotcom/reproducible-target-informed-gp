#!/usr/bin/env python3
"""Deterministic, automatically stopped cEI for the nominal 8-D WPT task.

Protocol: one geometric center; 75/25 feasibility/inductance-uncertainty
discovery until the first feasible observation; ordinary cEI thereafter;
stop when maximum candidate cEI predicts at most 1% relative improvement for
three consecutive refits; then purchase ten simultaneous GP confirmations.
Sobol candidate blocks and K=5 GP hyperparameter multistart are deterministic.
"""
from __future__ import annotations

import csv
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from run_center_start_cei_lr import deterministic_multistart
from run_wpt_8d_mixed_benchmark import (
    EPSILON, HIGH, LOW, LOWER_M, TARGET_M, UPPER_M, evaluate, normalize, unit,
)

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT/"essay"/"data"/"wpt_8d_automated_cei"
N_CANDIDATES = 2048
N_CONFIRM_CANDIDATES = 4096
N_CONFIRM = 10
MAX_MODEL_CALLS = 40
STOP_DELTA = .01
STOP_PATIENCE = 3
MIN_MODEL_CALLS = 2 * len(LOW) + 1
FEASIBILITY_WEIGHT = .75
GP_STARTS = 5
CENTER = (LOW+HIGH)/2
CENTER[[0, 1, 4, 5]] = np.rint(CENTER[[0, 1, 4, 5]])


def sobol_stream():
    needed = 1+(MAX_MODEL_CALLS-1)*N_CANDIDATES+N_CONFIRM_CANDIDATES
    power = int(math.ceil(math.log2(needed)))
    return qmc.scale(qmc.Sobol(8, scramble=False).random_base2(power), LOW, HIGH)


def canonical(points):
    points = np.asarray(points, float).copy()
    points[:, [0, 1, 4, 5]] = np.rint(points[:, [0, 1, 4, 5]])
    return points


def unseen_unique(points, observed):
    seen = {tuple(row) for row in np.asarray(observed)}
    answer, local = [], set()
    for row in canonical(points):
        key = tuple(row)
        if key not in seen and key not in local:
            answer.append(row); local.add(key)
    return np.asarray(answer, float)


def gps(x, y):
    models = []
    optimize = len(x) >= 4
    for output in range(2):
        if optimize:
            kernel = (ConstantKernel(1., (1e-3, 1e3))
                      * Matern(.35, (.03, 3.), nu=2.5)
                      + WhiteKernel(1e-5, (1e-8, 1e-2)))
            optimizer = deterministic_multistart(GP_STARTS)
        else:
            kernel = (ConstantKernel(1., constant_value_bounds="fixed")
                      * Matern(.35, length_scale_bounds="fixed", nu=2.5)
                      + WhiteKernel(1e-5, noise_level_bounds="fixed"))
            optimizer = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, alpha=1e-6,
                optimizer=optimizer, n_restarts_optimizer=0,
            ).fit(x, y[:, output])
        models.append(model)
    return models


def components(models, candidates, observed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(value).all()
               for value in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite deterministic WPT GP prediction")
    sd_m, sd_r = np.maximum(sd_m, 1e-12), np.maximum(sd_r, 1e-12)
    pf = ndtr((UPPER_M-mu_m)/sd_m)-ndtr((LOWER_M-mu_m)/sd_m)
    feasible = observed[(observed[:, 0] >= LOWER_M)
                        & (observed[:, 0] <= UPPER_M)]
    if len(feasible):
        incumbent = float(feasible[:, 1].min())
        improvement = incumbent-mu_r
        z = improvement/sd_r
        ei = improvement*ndtr(z)+sd_r*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
        cei = pf*np.maximum(ei, 0)/incumbent
    else:
        incumbent, cei = None, None
    return pf, mu_r, sd_m, cei, incumbent


def main():
    started = time.monotonic()
    stream = sobol_stream()
    x_raw = np.asarray([CENTER], float)
    y = np.asarray([evaluate(CENTER)])
    trace = [{"model_call": 1, "stage": "center", "M_nH": y[0, 0],
              "R_ohm": y[0, 1], "rho": "", "below_threshold_streak": 0}]
    first_feasible = (1 if LOWER_M <= y[0, 0] <= UPPER_M else None)
    streak = 0
    stopped_naturally = False
    final_rho = None

    while len(x_raw) < MAX_MODEL_CALLS:
        step = len(x_raw)-1
        block = stream[1+step*N_CANDIDATES:1+(step+1)*N_CANDIDATES]
        candidates = unseen_unique(block, x_raw)
        models = gps(normalize(x_raw), y)
        pf, _, sd_m, cei, _ = components(models, normalize(candidates), y)
        if first_feasible is None:
            score = (FEASIBILITY_WEIGHT*unit(pf)
                     +(1-FEASIBILITY_WEIGHT)*unit(sd_m))
            rho, stage = None, "feasibility"
        else:
            rho = float(np.max(cei))
            final_rho = rho
            stage = "cei"
            streak = streak+1 if rho <= STOP_DELTA else 0
            if len(x_raw) >= MIN_MODEL_CALLS and streak >= STOP_PATIENCE:
                stopped_naturally = True
                break
            score = cei
        chosen = candidates[int(np.argmax(score))]
        value = evaluate(chosen)
        x_raw = np.vstack([x_raw, chosen])
        y = np.vstack([y, value])
        if first_feasible is None and LOWER_M <= value[0] <= UPPER_M:
            first_feasible = len(x_raw)
        trace.append({"model_call": len(x_raw), "stage": stage,
                      "M_nH": value[0], "R_ohm": value[1],
                      "rho": "" if rho is None else rho,
                      "below_threshold_streak": streak})

    # Fixed terminal batch: fit once to the stopped history, then evaluate ten.
    confirm_start = 1+(MAX_MODEL_CALLS-1)*N_CANDIDATES
    candidates = unseen_unique(
        stream[confirm_start:confirm_start+N_CONFIRM_CANDIDATES], x_raw)
    models = gps(normalize(x_raw), y)
    pf, mu_r, _, _, _ = components(models, normalize(candidates), y)
    confirmation_score = pf*np.exp(-unit(mu_r))
    chosen = np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
    confirm_x = candidates[chosen]
    confirm_y = np.asarray([evaluate(point) for point in confirm_x])
    all_x, all_y = np.vstack([x_raw, confirm_x]), np.vstack([y, confirm_y])
    feasible = np.where((all_y[:, 0] >= LOWER_M) & (all_y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible))
    decision = (int(feasible[np.argmin(all_y[feasible, 1])]) if qualified
                else int(np.argmin(np.abs(all_y[:, 0]-TARGET_M))))
    result = {
        "qualified": qualified, "exact_reproducible": True,
        "best_M_nH": float(all_y[decision, 0]),
        "best_R_ohm": float(all_y[decision, 1]),
        "relative_target_error_pct": float(abs(all_y[decision, 0]-TARGET_M)/TARGET_M*100),
        "design": all_x[decision].tolist(),
        "decision_stage": ("model_building" if decision < len(x_raw) else "confirmation"),
        "first_feasible_evaluation": first_feasible,
        "n_model_building": len(x_raw), "n_confirmation": N_CONFIRM,
        "total_evaluations": len(all_x), "stopped_naturally": stopped_naturally,
        "final_below_threshold_streak": streak, "final_rho": final_rho,
        "runtime_sec": time.monotonic()-started,
    }
    manifest = {
        "task": "minimize resistance at M=1500 nH within 2.5 percent",
        "dimensions": 8, "pose": "fixed nominal",
        "center": CENTER.tolist(), "pre_feasibility_weights": {"pF": .75, "sigma_M": .25},
        "acquisition_after_feasibility": "ordinary cEI",
        "stop": {"maximum_relative_cEI": STOP_DELTA, "patience": STOP_PATIENCE,
                 "minimum_model_building": MIN_MODEL_CALLS},
        "budget": {"maximum_model_building": MAX_MODEL_CALLS,
                   "confirmation": N_CONFIRM, "maximum_total": MAX_MODEL_CALLS+N_CONFIRM},
        "gp": {"kernel": "isotropic Matern-5/2", "deterministic_starts": GP_STARTS,
               "fixed_prior_for_observations_below": 4},
        "candidates": {"sequence": "unscrambled Sobol", "per_step": N_CANDIDATES,
                       "confirmation_pool": N_CONFIRM_CANDIDATES},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    with (OUT/"trace.csv").open("w", newline="") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=list(trace[0]))
        writer.writeheader(); writer.writerows(trace)
    (OUT/"model_building_designs.json").write_text(json.dumps(x_raw.tolist(), indent=2)+"\n")
    (OUT/"confirmation_designs.json").write_text(json.dumps(confirm_x.tolist(), indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
