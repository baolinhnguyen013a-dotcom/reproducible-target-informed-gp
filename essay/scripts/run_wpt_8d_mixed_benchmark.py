#!/usr/bin/env python3
"""Nominal-alignment 8-D WPT optimization with shared-sample uncertainty fusion.

Budget: B = 6 initial + 32 sequential + 12 confirmation = 50 physical-model
evaluations.  The six relative-pose variables from Asadi and Abazari (2025)
are fixed; they are neither optimized nor interpreted as tolerances here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

PROJECT = Path(__file__).resolve().parents[2]
SIM_DIR = PROJECT / "mutual_inductance_simulator"
sys.path.insert(0, str(SIM_DIR))
from core import PartialMutualInductanceCalculator, PlanarPolygonCoil

TARGET_M = 1500.0
EPSILON = 0.025
LOWER_M, UPPER_M = TARGET_M * (1-EPSILON), TARGET_M * (1+EPSILON)
N_INITIAL, N_SEQUENTIAL, N_CONFIRM = 6, 32, 12
NOMINAL_TRANSLATION = (0.0, 0.0, 70.0)
NOMINAL_ROTATION = (0.0, 0.0, 0.0)
WIRE_D = 0.68
WIRE_A = math.pi * (WIRE_D/2)**2
RHO_CU = 1.851e-5
LOW = np.array([4, 5, 1, 30, 4, 5, 1, 30.], float)
HIGH = np.array([10, 15, 3, 80, 10, 15, 3, 80.], float)
POLICIES = {
    "cei": (1., 0., 0., 0.),
    "cei_gp": (.8, .2, 0., 0.),
    "cei_neural": (.8, 0., .1, .1),
    "cei_all": (.7, .1, .1, .1),
    "cei_all05": (.85, .05, .05, .05),
}


def unit(v):
    v = np.asarray(v, float)
    span = float(np.ptp(v))
    return (v-v.min())/span if span > 0 else np.zeros_like(v)


def sample_points(seed, count):
    rng = np.random.default_rng(seed)
    x = rng.uniform(LOW, HIGH, size=(count, 8))
    x[:, [0, 1, 4, 5]] = np.rint(x[:, [0, 1, 4, 5]])
    return x


def normalize(x):
    return (np.asarray(x)-LOW)/(HIGH-LOW)


def evaluate(pt):
    ns_t, nt_t, s_t, ri_t, ns_r, nt_r, s_r, ri_r = pt
    tx = PlanarPolygonCoil(int(round(ns_t)), int(round(nt_t)), float(s_t), float(ri_t), w=WIRE_D)
    rx = PlanarPolygonCoil(int(round(ns_r)), int(round(nt_r)), float(s_r), float(ri_r), w=WIRE_D)
    result = PartialMutualInductanceCalculator.calculate_mutual_inductance(
        tx, rx, NOMINAL_TRANSLATION, NOMINAL_ROTATION, n_quad=6)
    length_t = float(np.sum(2*tx.ns*tx.get_turn_radii()*1e3*np.sin(np.pi/tx.ns)))
    length_r = float(np.sum(2*rx.ns*rx.get_turn_radii()*1e3*np.sin(np.pi/rx.ns)))
    return float(result["M_nH"]), float(RHO_CU*(length_t+length_r)/WIRE_A)


class Ensemble(torch.nn.Module):
    def __init__(self, members, inputs, seed):
        super().__init__()
        rng = np.random.default_rng(seed % (2**32))
        def weight(i, o):
            return torch.nn.Parameter(torch.tensor(rng.normal(0, np.sqrt(2/(i+o)), (members, i, o)), dtype=torch.float32))
        self.w1, self.w2, self.w3 = weight(inputs, 16), weight(16, 8), weight(8, 1)
        self.b1 = torch.nn.Parameter(torch.zeros(members, 1, 16))
        self.b2 = torch.nn.Parameter(torch.zeros(members, 1, 8))
        self.b3 = torch.nn.Parameter(torch.zeros(members, 1, 1))
        self.members = members

    def parts(self, x):
        h = x.unsqueeze(0).expand(self.members, -1, -1)
        z1 = torch.bmm(h, self.w1)+self.b1
        z2 = torch.bmm(torch.relu(z1), self.w2)+self.b2
        pred = torch.bmm(torch.relu(z2), self.w3)+self.b3
        return pred.squeeze(-1), z1, z2

    def forward(self, x):
        return self.parts(x)[0]


def fit_nn(x, y, seed, members=10, epochs=120):
    mean, scale = float(np.mean(y)), max(float(np.std(y)), 1e-6)
    model = Ensemble(members, x.shape[1], seed)
    xt = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor((np.asarray(y)-mean)/scale, dtype=torch.float32)[None, :]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(xt)
        loss = ((pred-target)**2).mean()
        loss += .001*sum((p*p).sum() for p in (model.w1, model.w2, model.w3))/(members*100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    return model


@torch.no_grad()
def neural_signals(model, candidates, train):
    all_x = np.vstack([candidates, train])
    xt = torch.tensor(all_x, dtype=torch.float32)
    pred, z1t, z2t = model.parts(xt)
    disagreement = pred.std(0, unbiased=False).numpy()[:len(candidates)]
    x = all_x.astype(np.float64)
    a1, a2 = z1t.clamp_min(0).numpy().astype(np.float64), z2t.clamp_min(0).numpy().astype(np.float64)
    z1, z2 = z1t.numpy(), z2t.numpy()
    w2 = model.w2.detach().numpy().astype(np.float64)
    w3 = model.w3.detach().numpy()[:, :, 0].astype(np.float64)
    d2 = (z2 > 0)*w3[:, None, :]
    d1 = (z1 > 0)*np.einsum("mij,mnj->mni", w2, d2, optimize=True)
    members, count = a1.shape[:2]
    features = np.concatenate([
        (x[None, :, :, None]*d1[:, :, None, :]).reshape(members, count, -1),
        (a1[:, :, :, None]*d2[:, :, None, :]).reshape(members, count, -1),
        a2, d1, d2, np.ones((members, count, 1))], axis=-1)
    phi = np.transpose(features, (1, 0, 2)).reshape(count, -1)/math.sqrt(members)
    if not np.isfinite(phi).all():
        raise FloatingPointError("non-finite WPT neural tangent features")
    phi /= max(float(np.max(np.abs(phi))), 1e-12)
    phi /= max(math.sqrt(float(np.mean(np.einsum("ij,ij->i", phi, phi)))), 1e-12)
    cand_phi, train_phi = phi[:len(candidates)], phi[len(candidates):]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        ktt, kxt = train_phi@train_phi.T, cand_phi@train_phi.T
        prior = np.einsum("ij,ij->i", cand_phi, cand_phi)
    if not np.isfinite(ktt).all() or not np.isfinite(kxt).all() or not np.isfinite(prior).all():
        raise FloatingPointError("non-finite WPT neural tangent kernel")
    solved = np.linalg.solve(ktt+1e-3*np.eye(len(train)), kxt.T)
    ntk = np.sqrt(np.maximum(prior-np.einsum("ij,ji->i", kxt, solved), 0))
    return disagreement, ntk


def gps(x, y, seed):
    predictions = []
    for output in range(2):
        # Fixed hyperparameters are deliberate at n=6--38: repeatedly fitting
        # nine ARD scales from so few points is ill-conditioned and can drive
        # the likelihood optimizer to numerical extremes.
        kernel = ConstantKernel(1., constant_value_bounds="fixed")*Matern(
            np.full(8, .35), length_scale_bounds="fixed", nu=2.5
        )+WhiteKernel(1e-5, noise_level_bounds="fixed")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6,
                                             optimizer=None, random_state=seed).fit(x, y[:, output])
        predictions.append(model)
    return predictions


def gp_components(models, candidates, observed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu_m, sd_m = models[0].predict(candidates, return_std=True)
        mu_r, sd_r = models[1].predict(candidates, return_std=True)
    if not all(np.isfinite(v).all() for v in (mu_m, sd_m, mu_r, sd_r)):
        raise FloatingPointError("non-finite WPT GP prediction")
    sd_m, sd_r = np.maximum(sd_m, 1e-12), np.maximum(sd_r, 1e-12)
    pf = ndtr((UPPER_M-mu_m)/sd_m)-ndtr((LOWER_M-mu_m)/sd_m)
    feasible = observed[(observed[:, 0] >= LOWER_M) & (observed[:, 0] <= UPPER_M)]
    if len(feasible):
        incumbent = float(feasible[:, 1].min())
        z = (incumbent-mu_r)/sd_r
        ei = (incumbent-mu_r)*ndtr(z)+sd_r*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
        cei = pf*np.maximum(ei, 0)/incumbent
    else:
        cei = pf
    gp_global = .5*unit(sd_m)+.5*unit(sd_r)
    return unit(cei), unit(gp_global), pf, mu_r


def run_case(task):
    policy, weights, seed, candidate_count, epochs = task
    torch.set_num_threads(1)
    started = time.monotonic()
    x_raw = sample_points(seed, N_INITIAL)
    y = np.asarray([evaluate(p) for p in x_raw])
    for step in range(N_SEQUENTIAL):
        candidates = sample_points(seed*100003+step*1009+17, candidate_count)
        x, c = normalize(x_raw), normalize(candidates)
        models = gps(x, y, seed+step)
        cei, gp_global, _, _ = gp_components(models, c, y)
        score = weights[0]*cei+weights[1]*gp_global
        if weights[2] > 0 or weights[3] > 0:
            dis, ntk = [], []
            for output in range(2):
                model = fit_nn(x, y[:, output], seed*1000003+step*37+output*100003, epochs=epochs)
                d, n = neural_signals(model, c, x)
                dis.append(d); ntk.append(n)
            disagreement = .5*unit(dis[0])+.5*unit(dis[1])
            ntk_score = .5*unit(ntk[0])+.5*unit(ntk[1])
            score += weights[2]*unit(disagreement)+weights[3]*unit(ntk_score)
        chosen = candidates[int(np.argmax(score))]
        x_raw = np.vstack([x_raw, chosen])
        y = np.vstack([y, evaluate(chosen)])

    # GP-only batch confirmation, mirroring the low-dimensional protocol.
    confirm_candidates = sample_points(seed*100003+999983, max(candidate_count*2, 2000))
    models = gps(normalize(x_raw), y, seed+N_SEQUENTIAL)
    cei, _, pf, mu_r = gp_components(models, normalize(confirm_candidates), y)
    confirmation_score = pf*np.exp(-unit(mu_r))
    chosen = np.argsort(-confirmation_score, kind="stable")[:N_CONFIRM]
    x_raw = np.vstack([x_raw, confirm_candidates[chosen]])
    y = np.vstack([y, np.asarray([evaluate(p) for p in confirm_candidates[chosen]])])
    feasible_ids = np.where((y[:, 0] >= LOWER_M) & (y[:, 0] <= UPPER_M))[0]
    qualified = bool(len(feasible_ids))
    decision = int(feasible_ids[np.argmin(y[feasible_ids, 1])]) if qualified else int(np.argmin(np.abs(y[:, 0]-TARGET_M)))
    return {"policy": policy, "seed": seed, "budget": len(y), "n_initial": N_INITIAL,
            "n_sequential": N_SEQUENTIAL, "n_confirmation": N_CONFIRM, "qualified": qualified,
            "best_M_nH": float(y[decision, 0]), "best_R_ohm": float(y[decision, 1]),
            "relative_target_error_pct": float(abs(y[decision, 0]-TARGET_M)/TARGET_M*100),
            "decision_stage": "initial" if decision < N_INITIAL else ("sequential" if decision < N_INITIAL+N_SEQUENTIAL else "confirmation"),
            "design": json.dumps(x_raw[decision].tolist()), "runtime_sec": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=34261011)
    parser.add_argument("--candidate-count", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--policies", nargs="+", choices=tuple(POLICIES), default=tuple(POLICIES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected_policies = {name: POLICIES[name] for name in args.policies}
    tasks = [(name, weights, seed, args.candidate_count, args.epochs)
             for name, weights in selected_policies.items()
             for seed in range(args.seed_start, args.seed_start+args.seeds)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            print(f"checkpoint {i}/{len(tasks)}", flush=True)
    fields = list(rows[0])
    with (args.output/"results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = []
    for name, weights in selected_policies.items():
        group = [r for r in rows if r["policy"] == name]
        qualified = [r for r in group if r["qualified"]]
        summary.append({"policy": name, "weights": weights, "cases": len(group),
                        "qualification_rate": len(qualified)/len(group),
                        "mean_R_qualified_ohm": float(np.mean([r["best_R_ohm"] for r in qualified])) if qualified else None,
                        "std_R_qualified_ohm": float(np.std([r["best_R_ohm"] for r in qualified])) if qualified else None,
                        "mean_target_error_pct": float(np.mean([r["relative_target_error_pct"] for r in group])),
                        "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in group]))})
    manifest = {"source_model": "Asadi and Abazari (2025), doi:10.1049/elp2.70015",
                "dimensions": 8, "fixed_pose": {"translation_mm": NOMINAL_TRANSLATION, "rotation_rad": NOMINAL_ROTATION},
                "target_M_nH": TARGET_M, "relative_tolerance": EPSILON,
                "budget": {"initial": N_INITIAL, "sequential": N_SEQUENTIAL, "confirmation": N_CONFIRM, "total": 50},
                "same_samples_within_trajectory": True, "candidate_count": args.candidate_count,
                "ensemble_members": 10, "epochs": args.epochs, "wire_width_mm": WIRE_D,
                "seeds": list(range(args.seed_start, args.seed_start+args.seeds)), "policies": selected_policies}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
