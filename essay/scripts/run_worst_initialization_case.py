#!/usr/bin/env python3
"""Run the preselected worst random-six initialization through full GP-cEI.

The six physical designs are injected from the archived initialization screen;
the integer seed is used only for the baseline GP hyperparameter restarts.  The
comparison is an illustrative, preselected stress case rather than an estimate
of failure frequency.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_center_start_control as base  # noqa: E402
import run_center_cosine_stopping_smoke as stopping  # noqa: E402

SEED = 20261105
TARGET = 2
N_MODEL = 15
N_CONFIRM = 12
DATA = base.ROOT / "essay" / "data" / "initialization_screen_1000"
OUT = DATA / "worst_seed_end_to_end"


def recorded_initial_ids():
    with (DATA / "seed_initialization_points.csv").open(newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if int(r["seed"]) == SEED)
    points = [tuple(map(int, item.split(":")))
              for item in row["six_designs_T:n"].split(",")]
    return points, [base.POINTS.index(point) for point in points]


def run_injected_baseline(initial_ids):
    selected = list(initial_ids)
    selected_set = set(selected)
    first_feasible = next((position for position, index in enumerate(selected, 1)
                           if base.is_feasible(index, TARGET)), None)
    phases = []
    while len(selected) < N_MODEL:
        ids = np.asarray(selected, int)
        mean_l, std_l = base.fit(base.L_ALL, ids, SEED)
        mean_r, std_r = base.fit(base.R_ALL, ids, SEED)
        unseen = np.asarray([i for i in range(len(base.POINTS))
                             if i not in selected_set], int)
        sigma_l = np.maximum(std_l[unseen], 1e-12)
        lower, upper = TARGET * (1-base.EPSILON), TARGET * (1+base.EPSILON)
        p_f = (ndtr((upper-mean_l[unseen])/sigma_l)
               - ndtr((lower-mean_l[unseen])/sigma_l))
        feasible_ids = [i for i in selected if base.is_feasible(i, TARGET)]
        if feasible_ids:
            incumbent = min(base.R_ALL[i] for i in feasible_ids)
            sigma_r = np.maximum(std_r[unseen], 1e-12)
            z = (incumbent-mean_r[unseen])/sigma_r
            ei = ((incumbent-mean_r[unseen])*ndtr(z)
                  + sigma_r*np.exp(-.5*z*z)/np.sqrt(2*np.pi))
            score = p_f*np.maximum(ei, 0)/incumbent
            phase = "cEI"
        else:
            score = p_f
            phase = "pF"
        chosen = int(unseen[int(np.argmax(score))])
        selected.append(chosen)
        selected_set.add(chosen)
        phases.append(phase)
        if first_feasible is None and base.is_feasible(chosen, TARGET):
            first_feasible = len(selected)

    ids = np.asarray(selected, int)
    mean_l, std_l = base.fit(base.L_ALL, ids, SEED, final=True)
    mean_r, _ = base.fit(base.R_ALL, ids, SEED, final=True)
    unseen = np.asarray([i for i in range(len(base.POINTS))
                         if i not in selected_set], int)
    sigma_l = np.maximum(std_l[unseen], 1e-12)
    lower, upper = TARGET*(1-base.EPSILON), TARGET*(1+base.EPSILON)
    p_f = (ndtr((upper-mean_l[unseen])/sigma_l)
           - ndtr((lower-mean_l[unseen])/sigma_l))
    score = p_f*np.exp(-base.unit(mean_r[unseen]))
    confirmation = [int(i) for i in unseen[np.argsort(-score)[:N_CONFIRM]]]
    pool = selected + confirmation
    feasible_pool = [i for i in pool if base.is_feasible(i, TARGET)]
    domain = np.flatnonzero(np.abs(base.L_ALL-TARGET)/TARGET <= base.EPSILON)
    optimum = int(domain[np.argmin(base.R_ALL[domain])])
    qualified = bool(feasible_pool)
    decision = (min(feasible_pool, key=lambda i: base.R_ALL[i]) if qualified
                else min(pool, key=lambda i: abs(base.L_ALL[i]-TARGET)/TARGET))
    regret = float(base.R_ALL[decision]/base.R_ALL[optimum]-1) if qualified else None
    return {
        "method": "random6_pF_cEI_injected",
        "seed": SEED,
        "target_mH": TARGET,
        "budget": 27,
        "initial_points": [list(base.POINTS[i]) for i in initial_ids],
        "initial_L_mH": [float(base.L_ALL[i]) for i in initial_ids],
        "acquisition_phases": phases,
        "first_feasible_evaluation": first_feasible,
        "qualified": qualified,
        "exact": bool(qualified and regret <= 1e-12),
        "bounded_loss_u": (regret/(1+regret) if qualified else 1.0),
        "decision": {
            "point": list(base.POINTS[decision]),
            "L_mH": float(base.L_ALL[decision]),
            "R_ohm": float(base.R_ALL[decision]),
            "stage": ("initial" if decision in initial_ids else
                      "sequential" if decision in selected_set else "confirmation"),
        },
        "model_building_ids": selected,
        "confirmation_ids": confirmation,
    }


def main():
    points, initial_ids = recorded_initial_ids()
    baseline = run_injected_baseline(initial_ids)
    automatic = stopping.run_one(("ordinary_stop", 20261015, TARGET, .01, 5))
    report = {
        "status": "complete",
        "case_selection": "preselected by maximum initial mismatch before GP outcomes",
        "inference_scope": "illustrative single stress case; not population-level proof",
        "recorded_seed": SEED,
        "recorded_initial_points": [list(point) for point in points],
        "random6_baseline": baseline,
        "deterministic_automatic": automatic,
        "comparison": {
            "random6_exact": baseline["exact"],
            "deterministic_exact": bool(automatic["exact"]),
            "random6_calls": baseline["budget"],
            "deterministic_calls": int(automatic["oracle_calls"]),
            "bounded_loss_difference_random_minus_deterministic":
                float(baseline["bounded_loss_u"]-automatic["bounded_loss_u"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
