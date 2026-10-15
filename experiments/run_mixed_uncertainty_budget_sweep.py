#!/usr/bin/env python3
"""Shared-sample mixed-uncertainty acquisition benchmark.

Each trajectory uses B = 6 initial + x sequential + 12 confirmation
evaluations.  The GP and both neural uncertainty estimators are fitted to the
same physical observations.  A single trajectory is continued to the largest
requested budget and evaluated at every requested prefix, making the budget
sweep nested and directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
import warnings
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
N_INITIAL = 6
N_CONFIRM = 12
TARGETS = tuple(range(1, 88))
FIELDS = (
    "stage", "policy", "w_cei", "w_gp", "w_disagreement", "w_ntk",
    "budget", "n_initial", "n_sequential", "n_confirmation", "seed",
    "target_mH", "qualified", "bounded_loss_u", "exact_optimum", "top10",
    "decision_T", "decision_n", "decision_L_mH", "decision_R_ohm",
    "decision_source", "selected_indices", "runtime_sec",
)


def oracle_grid():
    cfg = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    wire, geom = cfg["wire"], cfg["geometry"]
    pitch = wire["diameter_mm"] * wire["winding_factor"]
    points, values = [], []
    for layers in range(geom["layers"]["min"], geom["layers"]["max"] + 1):
        for turns in range(geom["turns_per_layer"]["min"], geom["turns_per_layer"]["max"] + 1):
            total = layers * turns
            height, thickness = turns * pitch, layers * pitch
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
    xnorm = (x - x.min(0)) / np.maximum(x.max(0) - x.min(0), 1e-12)
    return x, xnorm, y


def unit(values):
    values = np.asarray(values, dtype=float)
    span = float(np.ptp(values))
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def fit_gp(x, y, ids, output, seed, restarts=0):
    kernel = (ConstantKernel(1., (1e-3, 1e3))
              * RBF([.25, .25], length_scale_bounds=(1e-2, 10.))
              + WhiteKernel(1e-6, noise_level_bounds=(1e-8, 1e-2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-10, normalize_y=True,
            random_state=int(seed), n_restarts_optimizer=int(restarts),
        ).fit(x[ids], y[ids, output])
        mean, std = model.predict(x, return_std=True)
    return mean, np.maximum(std, 1e-12)


@torch.no_grad()
def ntk_features(model, normalized):
    xt = torch.as_tensor(normalized, dtype=torch.float32)
    _, z1_t, z2_t = model.parts(xt)
    x = normalized.astype(np.float64)
    a1, a2 = z1_t.clamp_min(0).numpy().astype(np.float64), z2_t.clamp_min(0).numpy().astype(np.float64)
    z1, z2 = z1_t.numpy(), z2_t.numpy()
    w2 = model.w2.detach().numpy().astype(np.float64)
    w3 = model.w3.detach().numpy()[:, :, 0].astype(np.float64)
    delta2 = (z2 > 0) * w3[:, None, :]
    delta1 = (z1 > 0) * np.einsum("mij,mnj->mni", w2, delta2, optimize=True)
    members, count = a1.shape[:2]
    grad_w1 = (x[None, :, :, None] * delta1[:, :, None, :]).reshape(members, count, -1)
    grad_w2 = (a1[:, :, :, None] * delta2[:, :, None, :]).reshape(members, count, -1)
    return np.concatenate([grad_w1, grad_w2, a2, delta1, delta2,
                           np.ones((members, count, 1))], axis=-1)


def ntk_variance(features, selected, regularization=1e-3):
    members, count, parameters = features.shape
    phi = np.transpose(features, (1, 0, 2)).reshape(count, members * parameters) / math.sqrt(members)
    if not np.isfinite(phi).all():
        raise FloatingPointError("non-finite neural tangent features")
    phi /= max(float(np.max(np.abs(phi))), 1e-12)
    phi /= max(math.sqrt(float(np.mean(np.einsum("ij,ij->i", phi, phi)))), 1e-12)
    chosen = phi[np.asarray(selected, dtype=int)]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        ktt, kxt = chosen @ chosen.T, phi @ chosen.T
        prior = np.einsum("ij,ij->i", phi, phi)
    if not np.isfinite(ktt).all() or not np.isfinite(kxt).all() or not np.isfinite(prior).all():
        raise FloatingPointError("non-finite neural tangent kernel")
    solved = np.linalg.solve(ktt + regularization * np.eye(len(selected)), kxt.T)
    return np.maximum(prior - np.einsum("ij,ji->i", kxt, solved), 0.)


def neural_signals(x, y, ids, fit_seed, ensemble, epochs, need_disagreement, need_ntk):
    disagreements, ntks = [], []
    for output in range(2):
        model, _, _ = fit_nn(x[ids], y[ids, output], ensemble, epochs,
                             (int(fit_seed) + 100003 * output) % (2**32))
        if need_disagreement:
            pred = model(torch.as_tensor(x, dtype=torch.float32)).detach().numpy()
            disagreements.append(pred.std(axis=0, ddof=0))
        if need_ntk:
            ntks.append(np.sqrt(ntk_variance(ntk_features(model, x), ids)))
    disagreement = .5 * unit(disagreements[0]) + .5 * unit(disagreements[1]) if need_disagreement else None
    ntk = .5 * unit(ntks[0]) + .5 * unit(ntks[1]) if need_ntk else None
    return disagreement, ntk


def evaluate_prefix(stage, policy, weights, budget, seed, target, points, x, y,
                    selected, initial, started, predictions=None):
    ids = np.asarray(selected[:budget - N_CONFIRM], dtype=int)
    if predictions is None:
        mean_l, std_l = fit_gp(x, y, ids, 0, seed, 0)
        mean_r, _ = fit_gp(x, y, ids, 1, seed, 0)
    else:
        mean_l, std_l, mean_r = predictions
    unseen = np.setdiff1d(np.arange(len(y)), ids, assume_unique=False)
    lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
    pf = ndtr((upper - mean_l[unseen]) / std_l[unseen]) - ndtr((lower - mean_l[unseen]) / std_l[unseen])
    score = pf * np.exp(-unit(mean_r[unseen]))
    confirmation = unseen[np.argsort(-score, kind="stable")[:N_CONFIRM]]
    pool = np.concatenate([ids, confirmation])
    feasible = pool[np.abs(y[pool, 0] - target) <= EPSILON * target]
    qualified = bool(len(feasible))
    decision = int(feasible[np.argmin(y[feasible, 1])]) if qualified else int(pool[np.argmin(np.abs(y[pool, 0] - target))])
    all_feasible = np.where(np.abs(y[:, 0] - target) <= EPSILON * target)[0]
    ordered = all_feasible[np.argsort(y[all_feasible, 1], kind="stable")]
    optimum_r = float(y[ordered[0], 1])
    regret = float(y[decision, 1] / optimum_r - 1) if qualified else None
    loss = regret / (1 + regret) if qualified else 1.
    rank = int(np.searchsorted(np.sort(y[all_feasible, 1]), y[decision, 1], side="left") + 1) if qualified else 999999
    active_set = set(ids.tolist())
    source = "initial" if decision in initial else ("sequential" if decision in active_set else "confirmation")
    return {
        "stage": stage, "policy": policy,
        "w_cei": weights[0], "w_gp": weights[1], "w_disagreement": weights[2], "w_ntk": weights[3],
        "budget": budget, "n_initial": N_INITIAL, "n_sequential": budget - N_CONFIRM - N_INITIAL,
        "n_confirmation": N_CONFIRM, "seed": seed, "target_mH": target,
        "qualified": qualified, "bounded_loss_u": loss,
        "exact_optimum": bool(qualified and np.isclose(y[decision, 1], optimum_r, rtol=1e-9, atol=1e-10)),
        "top10": bool(qualified and rank <= 10), "decision_T": int(points[decision, 0]),
        "decision_n": int(points[decision, 1]), "decision_L_mH": float(y[decision, 0]),
        "decision_R_ohm": float(y[decision, 1]), "decision_source": source,
        "selected_indices": ",".join(map(str, ids.tolist())), "runtime_sec": time.monotonic() - started,
    }


def run_trajectory(task):
    stage, policy, weights, budgets, seed, target, ensemble, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    points, x, y = oracle_grid()
    selected = list(random.Random(int(seed)).sample(range(len(y)), N_INITIAL))
    initial = set(selected)
    max_train = max(budgets) - N_CONFIRM
    snapshots = {}
    while len(selected) < max_train:
        ids = np.asarray(selected, dtype=int)
        unseen = np.setdiff1d(np.arange(len(y)), ids, assume_unique=False)
        mean_l, std_l = fit_gp(x, y, ids, 0, seed)
        mean_r, std_r = fit_gp(x, y, ids, 1, seed)
        current_budget = len(selected) + N_CONFIRM
        if current_budget in budgets:
            snapshots[current_budget] = evaluate_prefix(
                stage, policy, weights, current_budget, seed, target, points,
                x, y, selected, initial, started, (mean_l, std_l, mean_r)
            )
        lower, upper = target * (1 - EPSILON), target * (1 + EPSILON)
        pf = ndtr((upper - mean_l[unseen]) / std_l[unseen]) - ndtr((lower - mean_l[unseen]) / std_l[unseen])
        feasible = ids[np.abs(y[ids, 0] - target) <= EPSILON * target]
        if len(feasible):
            incumbent = float(np.min(y[feasible, 1]))
            z = (incumbent - mean_r[unseen]) / std_r[unseen]
            ei = (incumbent - mean_r[unseen]) * ndtr(z) + std_r[unseen] * np.exp(-.5*z*z) / math.sqrt(2*math.pi)
            cei = pf * np.maximum(ei, 0.) / incumbent
        else:
            cei = pf
        components = [unit(cei), None, None, None]
        if weights[1] > 0:
            components[1] = unit(.5 * unit(std_l[unseen]) + .5 * unit(std_r[unseen]))
        if weights[2] > 0 or weights[3] > 0:
            fit_seed = (int(seed) * 1000003 + int(target) * 1009 + (len(ids)-N_INITIAL) * 37) % (2**32)
            disagreement, ntk = neural_signals(x, y, ids, fit_seed, ensemble, epochs,
                                                weights[2] > 0, weights[3] > 0)
            if weights[2] > 0:
                components[2] = unit(disagreement[unseen])
            if weights[3] > 0:
                components[3] = unit(ntk[unseen])
        score = sum(w * c for w, c in zip(weights, components) if w > 0)
        selected.append(int(unseen[int(np.argmax(score))]))
    final_budget = len(selected) + N_CONFIRM
    if final_budget in budgets:
        snapshots[final_budget] = evaluate_prefix(stage, policy, weights, final_budget,
                                                  seed, target, points, x, y, selected,
                                                  initial, started)
    return [snapshots[b] for b in budgets]


def parse_policy(spec):
    name, raw = spec.split("=", 1)
    weights = tuple(float(v) for v in raw.split(","))
    if len(weights) != 4 or any(v < 0 for v in weights) or not np.isclose(sum(weights), 1.):
        raise ValueError(f"policy must have four nonnegative weights summing to one: {spec}")
    return name, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--policies", nargs="+", required=True,
                        help="name=w_cei,w_gp,w_disagreement,w_ntk")
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--targets", nargs="+", type=int, default=TARGETS)
    parser.add_argument("--ensemble", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policies = [parse_policy(p) for p in args.policies]
    budgets = tuple(sorted(set(args.budgets)))
    if budgets[0] < N_INITIAL + 1 + N_CONFIRM:
        raise ValueError("each budget must contain at least one sequential acquisition")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest.update({"policies": [{"name": n, "weights": w} for n, w in policies],
                     "budgets": budgets, "n_initial": N_INITIAL, "n_confirmation": N_CONFIRM,
                     "shared_sample_rule": "all surrogates within a trajectory use identical physical observations",
                     "status": "running"})
    manifest["output"] = str(manifest["output"])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tasks = [(args.stage, name, weights, budgets, seed, target, args.ensemble, args.epochs)
             for name, weights in policies
             for seed in range(args.seed_start, args.seed_start + args.seeds)
             for target in args.targets]
    path = args.output / "results.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_trajectory, task) for task in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                writer.writerows(future.result())
                stream.flush()
                if i % 10 == 0 or i == len(tasks):
                    print(f"checkpoint {i}/{len(tasks)} trajectories", flush=True)
    rows = list(csv.DictReader(path.open(newline="")))
    summary = []
    for name, weights in policies:
        for budget in budgets:
            group = [r for r in rows if r["policy"] == name and int(r["budget"]) == budget]
            loss = np.asarray([float(r["bounded_loss_u"]) for r in group])
            summary.append({"policy": name, "weights": weights, "budget": budget, "cases": len(group),
                            "mean_u": float(loss.mean()),
                            "qualification_rate": float(np.mean([r["qualified"] == "True" for r in group])),
                            "exact_rate": float(np.mean([r["exact_optimum"] == "True" for r in group])),
                            "top10_rate": float(np.mean([r["top10"] == "True" for r in group]))})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest["status"] = "complete"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
