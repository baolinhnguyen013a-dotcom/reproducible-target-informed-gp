#!/usr/bin/env python3
"""Matched GP-cEI/GP/NN-disagreement/NN-NTK hybrid benchmark.

The physical observations are shared by the GP and neural models within each
trajectory.  Only the acquisition rule changes.  Final unseen-candidate
ranking is GP-only, and the returned design is chosen from physically measured
model-building and validation designs.  Results are checkpointed row by row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from coil_relu_100seed_benchmark import fit_nn

ROOT = Path(__file__).resolve().parents[1]
EPSILON = 0.025
VALIDATIONS = 12
TARGETS = tuple(range(1, 88))
METHODS = ("pure_cei", "cei_gp_global", "cei_nn_disagreement", "cei_nn_ntk")
FIELDS = (
    "stage", "method", "beta", "budget", "seed", "target_mH", "qualified",
    "relative_resistance_regret", "bounded_loss_u", "exact_optimum", "top10",
    "feasible_rank", "decision_T", "decision_n", "decision_L_mH",
    "decision_R_ohm", "decision_source", "selected_indices", "runtime_sec",
)


def oracle_grid():
    cfg = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    wire, geom = cfg["wire"], cfg["geometry"]
    pitch = wire["diameter_mm"] * wire["winding_factor"]
    points, values = [], []
    for layers in range(geom["layers"]["min"], geom["layers"]["max"] + 1):
        for turns in range(geom["turns_per_layer"]["min"], geom["turns_per_layer"]["max"] + 1):
            total = layers * turns
            height = turns * pitch
            thickness = layers * pitch
            diameter = geom["inner_diameter_mm"] + thickness
            inductance = .08 * total**2 * diameter**2 * 1e-4 / (
                3 * diameter + 9 * height + 10 * thickness
            )
            resistance = (4 * wire["resistivity_ohm_mm"] * total * diameter
                          / wire["diameter_mm"]**2)
            points.append((layers, turns))
            values.append((inductance, resistance))
    x = np.asarray(points, dtype=float)
    y = np.asarray(values, dtype=float)
    xnorm = (x - x.min(axis=0)) / np.maximum(x.max(axis=0) - x.min(axis=0), 1e-12)
    return x, xnorm, y


def unit_scale(values):
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    return (values - float(np.min(values))) / span if span > 0 else np.zeros_like(values)


def fit_gp(xnorm, y, selected, output, seed, restarts):
    kernel = (ConstantKernel(1., (1e-3, 1e3))
              * RBF([.25, .25], length_scale_bounds=(1e-2, 10.0))
              + WhiteKernel(1e-6, noise_level_bounds=(1e-8, 1e-2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10, normalize_y=True,
            random_state=int(seed), n_restarts_optimizer=int(restarts),
        ).fit(xnorm[selected], y[selected, output])
        mean, std = model.predict(xnorm, return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError("non-finite GP prediction")
    return mean, np.maximum(std, 1e-12)


@torch.no_grad()
def ntk_features(model, normalized):
    """Exact full-parameter finite-width features, 193 per ensemble member."""
    xt = torch.as_tensor(normalized, dtype=torch.float32)
    _, z1_t, z2_t = model.parts(xt)
    x = normalized.astype(np.float64)
    a1 = z1_t.clamp_min(0).numpy().astype(np.float64)
    a2 = z2_t.clamp_min(0).numpy().astype(np.float64)
    z1, z2 = z1_t.numpy(), z2_t.numpy()
    w2 = model.w2.detach().numpy().astype(np.float64)
    w3 = model.w3.detach().numpy()[:, :, 0].astype(np.float64)
    delta2 = (z2 > 0) * w3[:, None, :]
    delta1 = (z1 > 0) * np.einsum("mij,mnj->mni", w2, delta2, optimize=True)
    members, count = a1.shape[:2]
    grad_w1 = (x[None, :, :, None] * delta1[:, :, None, :]).reshape(members, count, -1)
    grad_w2 = (a1[:, :, :, None] * delta2[:, :, None, :]).reshape(members, count, -1)
    result = np.concatenate(
        [grad_w1, grad_w2, a2, delta1, delta2, np.ones((members, count, 1))], axis=-1
    )
    if result.shape[-1] != 193:
        raise AssertionError("unexpected neural parameter count")
    return result


def ntk_variance(features, selected, regularization=1e-3):
    members, count, parameters = features.shape
    phi = np.transpose(features, (1, 0, 2)).reshape(count, members * parameters) / math.sqrt(members)
    if not np.isfinite(phi).all():
        raise FloatingPointError("non-finite neural tangent features")
    # Rescale before any dot product.  The later unit-diagonal normalization
    # makes this common factor algebraically immaterial, while applying it here
    # prevents overflow for rare small-data neural fits.
    phi /= max(float(np.max(np.abs(phi))), 1e-12)
    scale = float(np.mean(np.einsum("ij,ij->i", phi, phi)))
    phi /= max(math.sqrt(scale), 1e-12)
    selected_phi = phi[np.asarray(selected, dtype=int)]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        ktt = selected_phi @ selected_phi.T
        kxt = phi @ selected_phi.T
        prior = np.einsum("ij,ij->i", phi, phi)
    if not np.isfinite(ktt).all() or not np.isfinite(kxt).all() or not np.isfinite(prior).all():
        raise FloatingPointError("non-finite neural tangent kernel")
    solved = np.linalg.solve(ktt + regularization * np.eye(len(selected)), kxt.T)
    variance = np.maximum(prior - np.einsum("ij,ji->i", kxt, solved), 0.0)
    if not np.isfinite(variance).all():
        raise FloatingPointError("non-finite neural tangent posterior variance")
    return variance


def neural_auxiliary(xnorm, y, selected, method, fit_seed, ensemble, epochs):
    signals = []
    for output in range(2):
        model, _, _ = fit_nn(
            xnorm[selected], y[selected, output], ensemble, epochs,
            (int(fit_seed) + 100003 * output) % (2**32),
        )
        if method == "cei_nn_disagreement":
            prediction = model(torch.as_tensor(xnorm, dtype=torch.float32)).detach().numpy()
            signals.append(prediction.std(axis=0, ddof=0))
        elif method == "cei_nn_ntk":
            signals.append(np.sqrt(ntk_variance(ntk_features(model, xnorm), selected)))
        else:
            raise ValueError(method)
    return .5 * unit_scale(signals[0]) + .5 * unit_scale(signals[1])


def run_case(task):
    stage, method, beta, budget, seed, target, ensemble, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    points, xnorm, y = oracle_grid()
    train_count = int(budget) - VALIDATIONS
    if train_count < 6:
        raise ValueError("budget must include at least six model-building observations and 12 validations")
    selected = list(random.Random(int(seed)).sample(range(len(y)), 6))
    initial = set(selected)
    for step in range(train_count - 6):
        ids = np.asarray(sorted(selected), dtype=int)
        unseen_mask = np.ones(len(y), dtype=bool)
        unseen_mask[ids] = False
        unseen = np.where(unseen_mask)[0]
        mean_l, std_l = fit_gp(xnorm, y, ids, 0, seed, 3)
        mean_r, std_r = fit_gp(xnorm, y, ids, 1, seed, 3)
        lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
        p_feasible = (ndtr((upper - mean_l[unseen]) / std_l[unseen])
                      - ndtr((lower - mean_l[unseen]) / std_l[unseen]))
        feasible_ids = ids[np.abs(y[ids, 0] - target) <= EPSILON * target]
        if len(feasible_ids):
            incumbent = float(np.min(y[feasible_ids, 1]))
            z = (incumbent - mean_r[unseen]) / std_r[unseen]
            ei = ((incumbent - mean_r[unseen]) * ndtr(z)
                  + std_r[unseen] * np.exp(-.5 * z**2) / math.sqrt(2 * math.pi))
            local = p_feasible * np.maximum(ei, 0) / incumbent
        else:
            local = p_feasible
        if method == "pure_cei":
            score = local
        else:
            if method == "cei_gp_global":
                auxiliary = .5 * unit_scale(std_l) + .5 * unit_scale(std_r)
            else:
                fit_seed = (int(seed) * 1000003 + int(target) * 1009 + step * 37) % (2**32)
                auxiliary = neural_auxiliary(
                    xnorm, y, ids, method, fit_seed, ensemble, epochs
                )
            score = ((1 - beta) * unit_scale(local)
                     + beta * unit_scale(auxiliary[unseen]))
        selected.append(int(unseen[int(np.argmax(score))]))

    ids = np.asarray(sorted(selected), dtype=int)
    mean_l, std_l = fit_gp(xnorm, y, ids, 0, seed, 5)
    mean_r, _ = fit_gp(xnorm, y, ids, 1, seed, 5)
    unseen_mask = np.ones(len(y), dtype=bool)
    unseen_mask[ids] = False
    unseen = np.where(unseen_mask)[0]
    lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
    p_feasible = (ndtr((upper - mean_l[unseen]) / std_l[unseen])
                  - ndtr((lower - mean_l[unseen]) / std_l[unseen]))
    validation_score = p_feasible * np.exp(-unit_scale(mean_r[unseen]))
    validation = unseen[np.argsort(-validation_score, kind="stable")[:VALIDATIONS]]
    pool = np.concatenate([ids, validation])
    feasible_pool = pool[np.abs(y[pool, 0] - target) <= EPSILON * target]
    if len(feasible_pool):
        decision_id = int(feasible_pool[np.argmin(y[feasible_pool, 1])])
        qualified = True
    else:
        decision_id = int(pool[np.argmin(np.abs(y[pool, 0] - target) / target)])
        qualified = False
    all_feasible = np.where(np.abs(y[:, 0] - target) <= EPSILON * target)[0]
    ordered = all_feasible[np.argsort(y[all_feasible, 1], kind="stable")]
    optimum_r = float(y[ordered[0], 1])
    regret = float(y[decision_id, 1] / optimum_r - 1) if qualified else None
    loss = regret / (1 + regret) if qualified else 1.0
    if qualified:
        rank = int(np.searchsorted(np.sort(y[all_feasible, 1]), y[decision_id, 1], side="left") + 1)
    else:
        rank = None
    source = "initial" if decision_id in initial else (
        "active_acquisition" if decision_id in set(ids) else "validation"
    )
    return {
        "stage": stage, "method": method, "beta": beta, "budget": budget,
        "seed": seed, "target_mH": target, "qualified": qualified,
        "relative_resistance_regret": regret, "bounded_loss_u": loss,
        "exact_optimum": bool(qualified and np.isclose(y[decision_id, 1], optimum_r, rtol=1e-9, atol=1e-10)),
        "top10": bool(qualified and rank <= 10), "feasible_rank": rank,
        "decision_T": int(points[decision_id, 0]), "decision_n": int(points[decision_id, 1]),
        "decision_L_mH": float(y[decision_id, 0]), "decision_R_ohm": float(y[decision_id, 1]),
        "decision_source": source, "selected_indices": ",".join(map(str, ids.tolist())),
        "runtime_sec": time.monotonic() - started,
    }


def read_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["method"], float(row["beta"]), int(row["budget"]))].append(row)
    output = []
    for (stage, method, beta, budget), items in sorted(groups.items()):
        loss = np.asarray([float(r["bounded_loss_u"]) for r in items])
        output.append({
            "stage": stage, "method": method, "beta": beta, "budget": budget,
            "cases": len(items), "mean_u": float(loss.mean()),
            "qualification_rate": float(np.mean([r["qualified"].lower() == "true" for r in items])),
            "exact_rate": float(np.mean([r["exact_optimum"].lower() == "true" for r in items])),
            "top10_rate": float(np.mean([r["top10"].lower() == "true" for r in items])),
            "mean_runtime_sec": float(np.mean([float(r["runtime_sec"]) for r in items])),
        })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--betas", nargs="+", type=float, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--targets", nargs="+", type=int, default=TARGETS)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = args.output / "results.csv"
    manifest = {
        "stage": args.stage, "methods": args.methods, "betas": args.betas,
        "budgets": args.budgets, "seed_start": args.seed_start, "seeds": args.seeds,
        "targets": args.targets, "ensemble": args.ensemble, "epochs": args.epochs,
        "validations": VALIDATIONS, "relative_tolerance": EPSILON,
        "normalization": "independent min-max scaling of cEI and auxiliary over unseen candidates",
        "final_ranking": "GP-only probability of feasibility times exp(-scaled mean resistance)",
        "status": "running",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    existing = read_rows(results)
    done = {(r["stage"], r["method"], round(float(r["beta"]), 8), int(r["budget"]),
             int(r["seed"]), int(float(r["target_mH"]))) for r in existing}
    tasks = []
    for method in args.methods:
        method_betas = [0.0] if method == "pure_cei" else args.betas
        for beta in method_betas:
            for budget in args.budgets:
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    for target in args.targets:
                        key = (args.stage, method, round(beta, 8), budget, seed, target)
                        if key not in done:
                            tasks.append((args.stage, method, beta, budget, seed, target,
                                          args.ensemble, args.epochs))
    new_file = not results.exists()
    with results.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_case, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                stream.flush()
                if index % 25 == 0 or index == len(tasks):
                    print(f"checkpoint {index}/{len(tasks)}", flush=True)
    rows = read_rows(results)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
