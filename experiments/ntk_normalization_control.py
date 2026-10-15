#!/usr/bin/env python3
"""Paired control for fixed versus round-wise L/R acquisition scaling.

This script intentionally imports the independently audited Antigravity NTK
implementation and changes only when the 90th-percentile L/R scales are
computed.  It compares:

* uncertainty_fixed   -- original ensemble disagreement implementation;
* uncertainty_dynamic -- recompute disagreement scales every active round;
* ntk_fixed           -- hold NTK scales from the first active round;
* ntk_dynamic         -- original full-parameter NTK implementation.

The final surrogate fit, validation-only decision pool, bounded loss, seeds,
and evaluation counts are common to every arm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HANDOFF_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(HANDOFF_SRC))
import coil_relu_100seed_benchmark as base  # noqa: E402
import coil_first_order_active_benchmark as first_order  # noqa: E402


METHODS = (
    "uncertainty_fixed",
    "uncertainty_dynamic",
    "ntk_fixed",
    "ntk_dynamic",
)
FIELDS = [
    "seed", "method", "mean_loss", "feasible_targets", "total_targets",
    "exact_optima", "mae_L_mH", "rmse_L_mH", "mae_R_ohm",
    "rmse_R_ohm", "runtime_sec", "indices", "train_evals", "budget",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(xnorm, selected, y, fit_seed, method, epochs, ensemble, alpha,
            fixed_scales):
    mask = np.ones(len(xnorm), dtype=bool)
    mask[selected] = False
    signals = []
    for output_index in range(2):
        seed = (int(fit_seed) * 1009 + 100003 * output_index) % (2**32)
        model, ym, ys = base.fit_nn(
            xnorm[selected], y[selected, output_index], ensemble, epochs, seed
        )
        if method.startswith("uncertainty"):
            _, std, _ = base.predict_nn(model, ym, ys, xnorm, False)
            signals.append(std)
        else:
            features = first_order.ntk_features(model, xnorm)
            variance = first_order.ntk_variance(
                features, selected, regularization=1e-3
            )
            signals.append(np.sqrt(np.maximum(variance, 0.0)))

    signal = np.column_stack(signals)
    if not np.isfinite(signal[mask]).all():
        bad = int(np.size(signal[mask]) - np.isfinite(signal[mask]).sum())
        raise FloatingPointError(
            f"{method} produced {bad} non-finite unobserved acquisition values "
            f"with {len(selected)} selected points"
        )
    dynamic = method.endswith("dynamic")
    if dynamic or fixed_scales is None:
        scales = np.maximum(np.percentile(signal[mask], 90, axis=0), 1e-9)
    else:
        scales = fixed_scales
    score = ((1.0 - alpha) * signal[:, 0] / scales[0]
             + alpha * signal[:, 1] / scales[1])
    score[~mask] = -np.inf
    return int(np.argmax(score)), scales


def run_one(x, y, seed, method, params):
    start = time.monotonic()
    xnorm = (x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-12)
    chosen = list(np.random.default_rng(seed).choice(len(y), size=6, replace=False))
    fixed_scales = None
    for round_index in range(params.train_count - 6):
        candidate, round_scales = acquire(
            xnorm, np.asarray(chosen, dtype=int), y, seed * 100 + round_index,
            method, params.epochs, params.ensemble, params.alpha, fixed_scales,
        )
        if fixed_scales is None:
            fixed_scales = round_scales.copy()
        chosen.append(candidate)

    selected = np.asarray(chosen, dtype=int)
    mu, sig, _ = base.posterior_models(
        xnorm, xnorm[selected], y[selected],
        seed=seed * 100 + params.train_count,
        kind="ensemble", ensemble=params.ensemble,
        epochs=params.epochs, gates=False,
    )
    metrics = base.metric_report(
        y, mu, sig, selected, params.targets, params.validations, params.epsilon
    )
    return {
        "seed": seed,
        "method": method,
        **metrics,
        "runtime_sec": time.monotonic() - start,
        "indices": ",".join(map(str, selected)),
        "train_evals": params.train_count,
        "budget": params.budget,
    }


def bootstrap_difference(pivot, challenger, baseline, seed=93737, draws=10000):
    differences = (
        pivot[challenger].to_numpy(dtype=float)
        - pivot[baseline].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(0, len(differences), size=(draws, len(differences)))
    ].mean(axis=1)
    return {
        "challenger": challenger,
        "baseline": baseline,
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "wins": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
        "losses": int(np.sum(differences > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--budget", type=int, default=27)
    parser.add_argument("--validations", type=int, default=12)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.025)
    args = parser.parse_args()
    if args.budget - args.validations <= 6:
        parser.error("budget-validations must exceed the six initial points")

    torch.set_num_threads(1)
    x, y = base.reconstructed_oracle()
    params = SimpleNamespace(
        epochs=args.epochs,
        ensemble=args.ensemble,
        budget=args.budget,
        validations=args.validations,
        train_count=args.budget - args.validations,
        alpha=args.alpha,
        epsilon=args.epsilon,
        targets=list(range(1, 88)),
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "rows.csv"
    settings = {
        **vars(args),
        "methods": list(METHODS),
        "oracle": "reconstructed analytical oracle",
        "source_sha256": {
            "coil_relu_100seed_benchmark.py": sha256(
                HANDOFF_SRC / "coil_relu_100seed_benchmark.py"
            ),
            "coil_first_order_active_benchmark.py": sha256(
                HANDOFF_SRC / "coil_first_order_active_benchmark.py"
            ),
        },
    }
    settings_path = output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise RuntimeError("Existing output directory has different settings")
    settings_path.write_text(json.dumps(settings, indent=2))

    existing = {}
    if rows_path.exists():
        with rows_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[(int(row["seed"]), row["method"])] = row

    rows = []
    for offset in range(args.seeds):
        seed = args.start_seed + offset
        for method in METHODS:
            key = seed, method
            if key in existing:
                row = existing[key]
            else:
                row = run_one(x, y, seed, method, params)
                with rows_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS)
                    if handle.tell() == 0:
                        writer.writeheader()
                    writer.writerow({field: row[field] for field in FIELDS})
                    handle.flush()
            rows.append(row)
        print(f"{offset + 1}/{args.seeds} paired seeds complete", flush=True)

    import pandas as pd
    frame = pd.DataFrame(rows)
    for column in FIELDS:
        if column not in ("method", "indices"):
            frame[column] = pd.to_numeric(frame[column])
    pivot = frame.pivot(index="seed", columns="method", values="mean_loss")
    summary = {
        "settings": settings,
        "means": {
            method: {
                metric: float(frame.loc[frame.method == method, metric].mean())
                for metric in (
                    "mean_loss", "feasible_targets", "exact_optima",
                    "mae_L_mH", "mae_R_ohm", "runtime_sec",
                )
            }
            for method in METHODS
        },
        "paired_controls": [
            bootstrap_difference(pivot, "uncertainty_dynamic", "uncertainty_fixed"),
            bootstrap_difference(pivot, "ntk_dynamic", "ntk_fixed"),
            bootstrap_difference(pivot, "ntk_fixed", "uncertainty_fixed"),
            bootstrap_difference(pivot, "ntk_dynamic", "uncertainty_dynamic"),
        ],
    }
    (output / "analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["means"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
