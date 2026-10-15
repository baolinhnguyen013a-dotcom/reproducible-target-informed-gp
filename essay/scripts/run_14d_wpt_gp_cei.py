"""
Full 14-Feature Practice Run: Budget B = 100 and B = 30
Evaluates GP-cEI vs Random Search across 10 random seeds on the 14D Table 2 engine:
- Physics engine: mutual_inductance_simulator (Asadi & Abazari 2025)
- Budget B = 30 and B = 100 physical calls
- Target M_t = 1500 nH (+/- 2.5% tolerance -> [1462.5, 1537.5] nH)
- Objective: Minimize total copper resistance R_total (Pouillet's Law)
"""

import sys
import time
from pathlib import Path
import json
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', ConvergenceWarning)

# Dynamic import of mutual_inductance_simulator
SIM_DIR = Path(__file__).resolve().parent.parent.parent / "mutual_inductance_simulator"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from core import PlanarPolygonCoil, PartialMutualInductanceCalculator

RHO_CU = 1.851e-5  # Ohm * mm
WIRE_D = 0.68      # mm
WIRE_A = np.pi * (WIRE_D / 2.0)**2  # mm^2

TARGET_M = 1500.0  # nH
EPSILON = 0.025    # +/- 2.5% tolerance
LOWER_M = TARGET_M * (1.0 - EPSILON)
UPPER_M = TARGET_M * (1.0 + EPSILON)


def sample_random_14d_point(rng: np.random.Generator) -> np.ndarray:
    ns_T = int(rng.integers(4, 11))
    nt_T = int(rng.integers(5, 16))
    s_T = float(rng.uniform(1.0, 3.0))
    r_i_T = float(rng.uniform(30.0, 80.0))

    ns_R = int(rng.integers(4, 11))
    nt_R = int(rng.integers(5, 16))
    s_R = float(rng.uniform(1.0, 3.0))
    r_i_R = float(rng.uniform(30.0, 80.0))

    r_o_T = r_i_T + (nt_T - 1) * s_T
    lat_max = (np.sqrt(2.0) / 2.0) * r_o_T
    v1 = float(rng.uniform(-lat_max, lat_max))
    v2 = float(rng.uniform(-lat_max, lat_max))
    v3 = float(rng.uniform(50.0, 120.0))

    phi = float(rng.uniform(-np.pi / 12.0, np.pi / 12.0))
    theta = float(rng.uniform(-np.pi / 12.0, np.pi / 12.0))
    psi = float(rng.uniform(-2.0 * np.pi / ns_R, 2.0 * np.pi / ns_R))

    return np.array([
        ns_T, nt_T, s_T, r_i_T,
        ns_R, nt_R, s_R, r_i_R,
        v1, v2, v3, phi, theta, psi
    ], dtype=np.float64)


def evaluate_physical_engine(point_14d: np.ndarray) -> dict:
    ns_T, nt_T, s_T, r_i_T, ns_R, nt_R, s_R, r_i_R, v1, v2, v3, phi, theta, psi = point_14d

    tx = PlanarPolygonCoil(ns=int(round(ns_T)), nt=int(round(nt_T)), s=float(s_T), r_i=float(r_i_T))
    rx = PlanarPolygonCoil(ns=int(round(ns_R)), nt=int(round(nt_R)), s=float(s_R), r_i=float(r_i_R))

    res = PartialMutualInductanceCalculator.calculate_mutual_inductance(
        tx, rx,
        translation=(float(v1), float(v2), float(v3)),
        rotation=(float(phi), float(theta), float(psi)),
        n_quad=8
    )
    M_nH = float(res['M_nH'])

    l_tx_mm = float(np.sum(2.0 * tx.ns * (tx.get_turn_radii() * 1e3) * np.sin(np.pi / tx.ns)))
    l_rx_mm = float(np.sum(2.0 * rx.ns * (rx.get_turn_radii() * 1e3) * np.sin(np.pi / rx.ns)))
    total_wire_m = (l_tx_mm + l_rx_mm) * 1e-3
    R_ohm = float(RHO_CU * (l_tx_mm + l_rx_mm) / WIRE_A)

    feasible = LOWER_M <= M_nH <= UPPER_M
    return {
        'M_nH': M_nH,
        'R_ohm': R_ohm,
        'wire_m': total_wire_m,
        'feasible': feasible,
        'point': point_14d
    }


def run_random_trial(seed: int, total_budget: int = 100):
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(total_budget):
        pt = sample_random_14d_point(rng)
        ev = evaluate_physical_engine(pt)
        history.append(ev)

    feas = [h for h in history if h['feasible']]
    if feas:
        best = min(feas, key=lambda x: x['R_ohm'])
        return {
            'seed': seed,
            'budget': total_budget,
            'qualified': True,
            'best_M_nH': best['M_nH'],
            'best_R_ohm': best['R_ohm'],
            'best_wire_m': best['wire_m'],
            'error_pct': abs(best['M_nH'] - TARGET_M) / TARGET_M * 100.0
        }
    else:
        best_closest = min(history, key=lambda x: abs(x['M_nH'] - TARGET_M))
        return {
            'seed': seed,
            'budget': total_budget,
            'qualified': False,
            'best_M_nH': best_closest['M_nH'],
            'best_R_ohm': best_closest['R_ohm'],
            'best_wire_m': best_closest['wire_m'],
            'error_pct': abs(best_closest['M_nH'] - TARGET_M) / TARGET_M * 100.0
        }


def run_gp_cei_trial(seed: int, total_budget: int = 100, n_init: int = 15, n_candidates: int = 4000):
    rng = np.random.default_rng(seed)
    history = []
    X_train = []
    y_M = []
    y_R = []

    for _ in range(n_init):
        pt = sample_random_14d_point(rng)
        ev = evaluate_physical_engine(pt)
        history.append(ev)
        X_train.append(pt)
        y_M.append(ev['M_nH'])
        y_R.append(ev['R_ohm'])

    scaler = MinMaxScaler()
    X_train_arr = np.array(X_train)
    scaler.fit(X_train_arr)

    n_active = total_budget - n_init
    for step in range(n_active):
        X_norm = scaler.transform(np.array(X_train))
        y_M_arr = np.array(y_M)
        y_R_arr = np.array(y_R)

        gp_M = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(14), nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1)),
            alpha=1e-6, normalize_y=True, random_state=seed, n_restarts_optimizer=1
        ).fit(X_norm, y_M_arr)

        gp_R = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(14), nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1)),
            alpha=1e-6, normalize_y=True, random_state=seed, n_restarts_optimizer=1
        ).fit(X_norm, y_R_arr)

        candidates = np.array([sample_random_14d_point(rng) for _ in range(n_candidates)])
        cand_norm = scaler.transform(candidates)

        mu_M, sigma_M = gp_M.predict(cand_norm, return_std=True)
        mu_R, sigma_R = gp_R.predict(cand_norm, return_std=True)

        sigma_M = np.maximum(sigma_M, 1e-12)
        p_f = ndtr((UPPER_M - mu_M) / sigma_M) - ndtr((LOWER_M - mu_M) / sigma_M)

        feas_seen = [h for h in history if h['feasible']]
        if feas_seen:
            incumbent_R = min(h['R_ohm'] for h in feas_seen)
            sigma_R = np.maximum(sigma_R, 1e-12)
            z = (incumbent_R - mu_R) / sigma_R
            ei = (incumbent_R - mu_R) * ndtr(z) + sigma_R * np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
            acq = p_f * np.maximum(ei, 0.0) / incumbent_R
        else:
            acq = p_f

        # Hybrid exploration coverage
        scale_M = max(float(np.std(y_M_arr)), 1e-12)
        scale_R = max(float(np.std(y_R_arr)), 1e-12)
        cov = 0.5 * sigma_M / scale_M + 0.5 * sigma_R / scale_R

        span_acq = np.ptp(acq)
        norm_acq = (acq - np.min(acq)) / span_acq if span_acq > 0 else np.zeros_like(acq)
        span_cov = np.ptp(cov)
        norm_cov = (cov - np.min(cov)) / span_cov if span_cov > 0 else np.zeros_like(cov)

        score = 0.75 * norm_acq + 0.25 * norm_cov
        best_cand_idx = int(np.argmax(score))
        next_pt = candidates[best_cand_idx]

        ev = evaluate_physical_engine(next_pt)
        history.append(ev)
        X_train.append(next_pt)
        y_M.append(ev['M_nH'])
        y_R.append(ev['R_ohm'])

    feas = [h for h in history if h['feasible']]
    if feas:
        best = min(feas, key=lambda x: x['R_ohm'])
        return {
            'seed': seed,
            'budget': total_budget,
            'qualified': True,
            'best_M_nH': best['M_nH'],
            'best_R_ohm': best['R_ohm'],
            'best_wire_m': best['wire_m'],
            'error_pct': abs(best['M_nH'] - TARGET_M) / TARGET_M * 100.0
        }
    else:
        best_closest = min(history, key=lambda x: abs(x['M_nH'] - TARGET_M))
        return {
            'seed': seed,
            'budget': total_budget,
            'qualified': False,
            'best_M_nH': best_closest['M_nH'],
            'best_R_ohm': best_closest['R_ohm'],
            'best_wire_m': best_closest['wire_m'],
            'error_pct': abs(best_closest['M_nH'] - TARGET_M) / TARGET_M * 100.0
        }


if __name__ == '__main__':
    seeds = list(range(1, 11))
    print("=" * 80)
    print("RUNNING 14D WPT COIL BENCHMARK: GP-cEI vs RANDOM SEARCH (B=100)")
    print("=" * 80)

    with ProcessPoolExecutor(max_workers=5) as pool:
        gp_res = list(pool.map(lambda s: run_gp_cei_trial(s, 100), seeds))
        rnd_res = list(pool.map(lambda s: run_random_trial(s, 100), seeds))

    print("\n--- GP-cEI Results (B=100) ---")
    gp_qual = [r for r in gp_res if r['qualified']]
    print(f"Qualification: {len(gp_qual)}/10 ({len(gp_qual)*10}%)")
    if gp_qual:
        print(f"Mean R: {np.mean([r['best_R_ohm'] for r in gp_qual]):.4f} Ohm | "
              f"Best R: {np.min([r['best_R_ohm'] for r in gp_qual]):.4f} Ohm")

    print("\n--- Random Search Results (B=100) ---")
    rnd_qual = [r for r in rnd_res if r['qualified']]
    print(f"Qualification: {len(rnd_qual)}/10 ({len(rnd_qual)*10}%)")
    if rnd_qual:
        print(f"Mean R: {np.mean([r['best_R_ohm'] for r in rnd_qual]):.4f} Ohm | "
              f"Best R: {np.min([r['best_R_ohm'] for r in rnd_qual]):.4f} Ohm")
