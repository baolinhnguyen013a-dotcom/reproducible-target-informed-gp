"""Error and calibration analysis for the exact B=27 global GP benchmark."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from compare_surrogate_targets import ROOT, truth_grid, write_json
from run_gp_budget_27_relative import (
    BUDGET,
    EPSILON,
    SEEDS,
    TARGETS,
    VALIDATION_COUNT,
    bounded_loss,
    choose,
    feasible,
    fit_gp,
)

OUTPUT = ROOT / "results" / "01_current_relative_tolerance" / "01_surrogates_budget_20_to_40_10seeds"
GRID_METRICS = OUTPUT / "gp_budget_27_grid_error_by_seed.csv"
TARGET_DIAGNOSTICS = OUTPUT / "gp_budget_27_target_error_diagnostics.csv"
SUMMARY = OUTPUT / "gp_budget_27_error_analysis.json"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    grid = truth_grid(config)
    true_l = np.asarray([point["L_mH"] for point in grid])
    grid_rows, target_rows = [], []

    for seed in SEEDS:
        payload = fit_gp(seed, config, grid)
        mean = np.asarray(payload["mean"], dtype=float)
        sigma = np.asarray(payload["sigma"], dtype=float)
        training = set(map(int, payload["training_ids"]))
        held_out = np.asarray([i for i in range(len(grid)) if i not in training])
        errors = np.abs(mean[held_out] - true_l[held_out])
        safe_sigma = np.maximum(sigma[held_out], 1e-12)
        correlation = float(np.corrcoef(errors, sigma[held_out])[0, 1])
        grid_rows.append({
            "seed": seed,
            "training_evaluations": len(training),
            "held_out_mae_mH": float(np.mean(errors)),
            "held_out_rmse_mH": float(np.sqrt(np.mean(errors**2))),
            "held_out_max_error_mH": float(np.max(errors)),
            "coverage_1sigma": float(np.mean(errors <= safe_sigma)),
            "coverage_2sigma": float(np.mean(errors <= 2 * safe_sigma)),
            "mean_sigma_mH": float(np.mean(sigma[held_out])),
            "error_sigma_correlation": correlation,
        })

        unseen = np.asarray([i for i in range(len(grid)) if i not in training])
        for target in TARGETS:
            optimum = min((point for point in grid if feasible(point, target)), key=lambda p: p["R_ohm"])
            ranking = unseen[np.argsort(np.abs(mean[unseen] - target) / target)]
            validation = ranking[:VALIDATION_COUNT]
            current = choose(validation, grid, target)
            expanded = choose(training | set(map(int, validation)), grid, target)
            current_qualified, current_regret, current_loss = bounded_loss(current, optimum, target)
            expanded_qualified, expanded_regret, expanded_loss = bounded_loss(expanded, optimum, target)
            validation_feasible = [int(i) for i in validation if feasible(grid[int(i)], target)]
            target_rows.append({
                "seed": seed,
                "target_mH": target,
                "validation_feasible_count": len(validation_feasible),
                "qualified_current_pool": current_qualified,
                "current_relative_regret": current_regret,
                "current_bounded_loss_u": current_loss,
                "qualified_expanded_pool": expanded_qualified,
                "expanded_relative_regret": expanded_regret,
                "expanded_bounded_loss_u": expanded_loss,
                "loss_reduction_from_retaining_training_points": current_loss - expanded_loss,
                "oracle_optimum_in_training": any(
                    grid[i]["T"] == optimum["T"] and grid[i]["n"] == optimum["n"] for i in training
                ),
                "oracle_optimum_in_validation": any(
                    grid[int(i)]["T"] == optimum["T"] and grid[int(i)]["n"] == optimum["n"] for i in validation
                ),
                "decision_relative_L_error": abs(current["L_mH"] - target) / target,
            })
        print(f"analysed seed {seed}", flush=True)

    write_csv(GRID_METRICS, grid_rows)
    write_csv(TARGET_DIAGNOSTICS, target_rows)
    qualified = [r for r in target_rows if r["qualified_current_pool"]]
    expanded_qualified = [r for r in target_rows if r["qualified_expanded_pool"]]
    improved = [r for r in target_rows if r["loss_reduction_from_retaining_training_points"] > 1e-15]
    by_bin = defaultdict(list)
    for row in target_rows:
        target = int(row["target_mH"])
        label = "1--20" if target <= 20 else "21--40" if target <= 40 else "41--60" if target <= 60 else "61--87"
        by_bin[label].append(row)
    summary = {
        "budget": BUDGET,
        "seeds": list(SEEDS),
        "targets_mH": [min(TARGETS), max(TARGETS)],
        "cases": len(target_rows),
        "grid_prediction": {
            key: float(np.mean([row[key] for row in grid_rows]))
            for key in ("held_out_mae_mH", "held_out_rmse_mH", "held_out_max_error_mH",
                        "coverage_1sigma", "coverage_2sigma", "mean_sigma_mH", "error_sigma_correlation")
        },
        "current_decision_pool": {
            "qualification_rate": len(qualified) / len(target_rows),
            "mean_loss_all_cases": float(np.mean([r["current_bounded_loss_u"] for r in target_rows])),
            "mean_relative_regret_qualified": float(np.mean([r["current_relative_regret"] for r in qualified])),
            "oracle_optimum_validation_recall": float(np.mean([r["oracle_optimum_in_validation"] for r in target_rows])),
        },
        "expanded_decision_pool": {
            "qualification_rate": len(expanded_qualified) / len(target_rows),
            "mean_loss_all_cases": float(np.mean([r["expanded_bounded_loss_u"] for r in target_rows])),
            "cases_improved": len(improved),
            "mean_loss_reduction_all_cases": float(np.mean([r["loss_reduction_from_retaining_training_points"] for r in target_rows])),
        },
        "target_bins": {
            label: {
                "cases": len(rows),
                "qualification_rate": float(np.mean([r["qualified_current_pool"] for r in rows])),
                "mean_loss": float(np.mean([r["current_bounded_loss_u"] for r in rows])),
                "mean_relative_L_error": float(np.mean([r["decision_relative_L_error"] for r in rows])),
            }
            for label, rows in by_bin.items()
        },
    }
    write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
