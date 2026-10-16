#!/usr/bin/env python3
"""Run the proposed automatic GP workflow on held-out 2-D coil targets.

This is the frozen paper method: deterministic center initialization, 75/25
feasibility/uncertainty discovery, ordinary cEI after feasibility,
deterministic GP fitting, the evidence guard, adaptive stopping, and final
confirmation accounting.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import run_center_cosine_stopping_smoke as stopping
import run_center_start_control as base


DEVELOPMENT_TARGETS = set(stopping.TARGETS)
TARGETS = tuple(t for t in range(1, 88) if t not in DEVELOPMENT_TARGETS)
SEED_LABEL = base.SEEDS[0]  # inert under deterministic K=5 fitting
OUT = base.ROOT / "essay" / "data" / "center_stopping_heldout_73"
ROWS = OUT / "results.csv"


def read_rows():
    if not ROWS.exists():
        return []
    with ROWS.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    existing = read_rows()
    done = {int(row["target_mH"]) for row in existing}
    tasks = [("ordinary_stop", SEED_LABEL, target, stopping.DELTA_STOP, 5)
             for target in TARGETS if target not in done]
    new_file = not ROWS.exists()
    with ROWS.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=stopping.FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(stopping.run_one, task): task for task in tasks}
            for count, future in enumerate(as_completed(futures), 1):
                writer.writerow(future.result())
                stream.flush()
                if count % 10 == 0 or count == len(tasks):
                    print("checkpoint %d/%d" % (count, len(tasks)), flush=True)

    rows = read_rows()
    if len(rows) != len(TARGETS):
        raise RuntimeError("incomplete held-out test: %d/%d" %
                           (len(rows), len(TARGETS)))
    losses = np.asarray([float(row["bounded_loss_u"]) for row in rows])
    calls = np.asarray([int(row["oracle_calls"]) for row in rows])
    failed = [int(row["target_mH"]) for row in rows
              if row["qualified"] != "True"]
    inexact = [int(row["target_mH"]) for row in rows
               if row["exact"] != "True"]
    report = {
        "split": {
            "development_targets_mH": sorted(DEVELOPMENT_TARGETS),
            "heldout_targets_mH": list(TARGETS),
            "heldout_targets": len(TARGETS),
        },
        "frozen_configuration": {
            "feasibility_weight": stopping.PF_WEIGHT,
            "delta_relative_improvement": stopping.DELTA_STOP,
            "patience": stopping.PATIENCE,
            "minimum_model_building": stopping.MIN_MODEL_CALLS,
            "deterministic_gp_starts": 5,
            "confirmation": stopping.N_CONFIRM,
            "maximum_model_building": stopping.MAX_MODEL_CALLS,
        },
        "results": {
            "qualified_targets": len(rows)-len(failed),
            "failed_targets_mH": failed,
            "exact_optimum_targets": int(np.sum(losses <= 1e-12)),
            "inexact_targets_mH": inexact,
            "mean_bounded_loss_u": float(losses.mean()),
            "mean_total_calls": float(calls.mean()),
            "median_total_calls": float(np.median(calls)),
            "minimum_total_calls": int(calls.min()),
            "maximum_total_calls": int(calls.max()),
            "natural_stop_targets": int(sum(row["stopped_naturally"] == "True"
                                            for row in rows)),
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps({
        "status": "complete",
        "script": str(__file__),
        "deterministic": True,
        "seed_label_is_inert": True,
        "development_data_excluded": True,
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
