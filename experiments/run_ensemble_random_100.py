#!/usr/bin/env python3
"""Matched 100-seed random-sampling ensemble control for the NTK audit."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch


HANDOFF_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(HANDOFF_SRC))
import coil_relu_100seed_benchmark as base  # noqa: E402


FIELDS = [
    "seed", "method", "mean_loss", "feasible_targets", "total_targets",
    "exact_optima", "mae_L_mH", "rmse_L_mH", "mae_R_ohm",
    "rmse_R_ohm", "runtime_sec", "indices", "train_evals", "budget",
]


def run_one(x, y, seed, epochs, ensemble, validations, epsilon):
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    chosen = list(map(int, rng.choice(len(y), size=6, replace=False)))
    for _ in range(9):
        mask = np.ones(len(y), dtype=bool)
        mask[chosen] = False
        chosen.append(int(rng.choice(np.where(mask)[0])))
    selected = np.asarray(chosen, dtype=int)
    xnorm = (x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-12)
    mu, sig, _ = base.posterior_models(
        xnorm, xnorm[selected], y[selected], seed=seed * 100 + len(selected),
        kind="ensemble", ensemble=ensemble, epochs=epochs, gates=False,
    )
    metrics = base.metric_report(
        y, mu, sig, selected, list(range(1, 88)), validations, epsilon
    )
    return {
        "seed": seed, "method": "ensemble_random", **metrics,
        "runtime_sec": time.monotonic() - start,
        "indices": ",".join(map(str, selected)),
        "train_evals": len(selected), "budget": len(selected) + validations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=30261011)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--validations", type=int, default=12)
    parser.add_argument("--epsilon", type=float, default=0.025)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x, y = base.reconstructed_oracle()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.csv"
    existing = set()
    if rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as handle:
            existing = {int(row["seed"]) for row in csv.DictReader(handle)}
    for offset in range(args.seeds):
        seed = args.start_seed + offset
        if seed in existing:
            continue
        row = run_one(
            x, y, seed, args.epochs, args.ensemble,
            args.validations, args.epsilon,
        )
        with rows_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
        print(f"{offset + 1}/{args.seeds} seeds complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
