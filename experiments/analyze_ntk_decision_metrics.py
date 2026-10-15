#!/usr/bin/env python3
"""Decision-focused audit of the NTK normalization-control experiment.

Reconstruct the final validation decision for every saved method/seed and
measure whether the returned design is the exact optimum or lies among the ten
lowest-resistance feasible designs.  This distinguishes global response MAE
from the ranking/feasibility quantities that determine constrained design.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from scipy.stats import spearmanr


HANDOFF_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(HANDOFF_SRC))
import coil_relu_100seed_benchmark as base  # noqa: E402


def decision_counts(y, pred, sig, selected, targets, validation_count, epsilon):
    mask = np.ones(len(y), dtype=bool)
    mask[selected] = False
    unseen = np.where(mask)[0]
    resistance_scale = max(float(np.std(y[selected, 1])), 1e-6)
    resistance_anchor = float(np.min(y[selected, 1]))
    effective_r = np.clip(
        (pred[unseen, 1] - resistance_anchor) / resistance_scale, -10, 10
    )
    resistance_utility = np.exp(-np.maximum(effective_r, 0))

    exact = 0
    top10 = 0
    feasible = 0
    losses = []
    for target in targets:
        actual_feasible = np.abs(y[:, 0] - target) <= epsilon * target
        feasible_indices = np.where(actual_feasible)[0]
        feasible_order = feasible_indices[
            np.lexsort((feasible_indices, y[feasible_indices, 1]))
        ]
        top10_indices = set(map(int, feasible_order[:10]))
        optimum = float(y[feasible_order[0], 1])

        s = np.maximum(sig[unseen, 0], 1e-9)
        probability = np.maximum(
            ndtr((target * (1 + epsilon) - pred[unseen, 0]) / s)
            - ndtr((target * (1 - epsilon) - pred[unseen, 0]) / s),
            0,
        )
        score = probability * resistance_utility
        order = np.lexsort((np.abs(pred[unseen, 0] - target), -score))
        validated = unseen[order[:validation_count]]
        eligible = validated[actual_feasible[validated]]
        if len(eligible) == 0:
            losses.append(1.0)
            continue

        feasible += 1
        returned = int(eligible[np.argmin(y[eligible, 1])])
        returned_r = float(y[returned, 1])
        exact += int(np.isclose(returned_r, optimum, rtol=1e-9, atol=1e-10))
        top10 += int(returned in top10_indices)
        losses.append(max(0.0, 1.0 - optimum / returned_r))
    return feasible, exact, top10, float(np.mean(losses))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--validations", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.025)
    args = parser.parse_args()

    torch.set_num_threads(1)
    rows = pd.concat([pd.read_csv(path) for path in args.rows], ignore_index=True)
    x, y = base.reconstructed_oracle()
    xnorm = (x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-12)
    targets = list(range(1, 88))
    records = []
    cache = {}
    for number, row in enumerate(rows.itertuples(index=False), start=1):
        selected = np.asarray([int(value) for value in row.indices.split(",")])
        cache_key = (int(row.seed), tuple(selected))
        if cache_key not in cache:
            cache[cache_key] = base.posterior_models(
                xnorm, xnorm[selected], y[selected],
                seed=int(row.seed) * 100 + len(selected), kind="ensemble",
                ensemble=args.ensemble, epochs=args.epochs, gates=False,
            )[:2]
        pred, sig = cache[cache_key]
        feasible, exact, top10, mean_loss = decision_counts(
            y, pred, sig, selected, targets, args.validations, args.epsilon
        )
        if not np.isclose(mean_loss, float(row.mean_loss), atol=1e-12, rtol=1e-10):
            raise AssertionError(
                f"decision reconstruction mismatch for {row.seed}/{row.method}: "
                f"{mean_loss} vs {row.mean_loss}"
            )
        records.append({
            "seed": int(row.seed), "method": row.method,
            "mae_L_mH": float(row.mae_L_mH), "mae_R_ohm": float(row.mae_R_ohm),
            "mean_loss": mean_loss, "feasible_targets": feasible,
            "exact_optima": exact, "top10_returns": top10,
            "exact_rate": exact / len(targets), "top10_rate": top10 / len(targets),
        })
        if number % 25 == 0:
            print(f"{number}/{len(rows)} records reconstructed", flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "decision_metrics_by_seed.csv", index=False)

    aggregate = {}
    correlations = {}
    for method, group in frame.groupby("method", sort=True):
        aggregate[method] = {
            column: float(group[column].mean())
            for column in (
                "mae_L_mH", "mae_R_ohm", "mean_loss", "feasible_targets",
                "exact_optima", "top10_returns", "exact_rate", "top10_rate",
            )
        }
        correlations[method] = {}
        for predictor in ("mae_L_mH", "mae_R_ohm", "exact_rate", "top10_rate"):
            statistic, pvalue = spearmanr(group[predictor], group["mean_loss"])
            correlations[method][predictor] = {
                "spearman_rho_with_mean_loss": float(statistic),
                "two_sided_p": float(pvalue),
            }

    result = {
        "definition": (
            "top10_return means that the final minimum-resistance feasible design "
            "among the 12 validations is one of the ten lowest-resistance feasible "
            "configurations in exhaustive search for that target"
        ),
        "aggregate": aggregate,
        "within_method_spearman_correlations": correlations,
    }
    (output / "decision_metric_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
