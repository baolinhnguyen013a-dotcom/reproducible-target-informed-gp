"""HISTORICAL SCRIPT -- not evidence for the revised manuscript.

The revised scope fixes all six pose variables and uses
``run_wpt_8d_mixed_benchmark.py``.  This file is retained only so that the
older archived tolerance artifacts remain traceable.

WPT Planar Coil Synthesis: 8-D Geometric Design and 6-DOF Spatial Tolerance Analysis
Physics Engine: mutual_inductance_simulator (Asadi & Abazari 2025)

Decomposition:
1. Stage 1 (Geometric Synthesis): Optimize 8 intrinsic coil variables at nominal alignment (z=70mm, 0 misalignments)
   Tx: (ns_T, nt_T, s_T, r_i_T)
   Rx: (ns_R, nt_R, s_R, r_i_R)
   Target: M_t = 1500 nH (+/- 2.5% tolerance) | Objective: Minimize R_total

2. Stage 2 (Spatial Tolerance Evaluation): Evaluate sensitivity over the 6-DOF misalignment envelope
   Lateral (dx, dy), Axial gap (dz), Angular tilts (phi, theta, psi)
"""

import json
import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from wpt_partial_inductance_oracle import (
    PartialMutualInductanceCalculator,
    PlanarPolygonCoil,
)

RHO_CU = 1.851e-5  # Ohm * mm
WIRE_D = 0.68      # mm
WIRE_A = np.pi * (WIRE_D / 2.0)**2  # mm^2
TARGET_M = 1500.0  # nH
EPSILON = 0.025    # +/- 2.5%
LOWER_M = TARGET_M * (1.0 - EPSILON)
UPPER_M = TARGET_M * (1.0 + EPSILON)
NOMINAL_Z = 70.0   # mm


def sample_geom_point(rng: np.random.Generator) -> np.ndarray:
    ns_T = int(rng.integers(4, 11))
    nt_T = int(rng.integers(5, 16))
    s_T = float(rng.uniform(1.0, 3.0))
    r_i_T = float(rng.uniform(30.0, 80.0))

    ns_R = int(rng.integers(4, 11))
    nt_R = int(rng.integers(5, 16))
    s_R = float(rng.uniform(1.0, 3.0))
    r_i_R = float(rng.uniform(30.0, 80.0))

    return np.array([ns_T, nt_T, s_T, r_i_T, ns_R, nt_R, s_R, r_i_R], dtype=np.float64)


def evaluate_geom(pt_8d: np.ndarray, translation=(0.0, 0.0, NOMINAL_Z), rotation=(0.0, 0.0, 0.0)) -> dict:
    ns_T, nt_T, s_T, r_i_T, ns_R, nt_R, s_R, r_i_R = pt_8d

    tx = PlanarPolygonCoil(ns=int(round(ns_T)), nt=int(round(nt_T)), s=float(s_T), r_i=float(r_i_T))
    rx = PlanarPolygonCoil(ns=int(round(ns_R)), nt=int(round(nt_R)), s=float(s_R), r_i=float(r_i_R))

    res = PartialMutualInductanceCalculator.calculate_mutual_inductance(
        tx, rx,
        translation=translation,
        rotation=rotation,
        n_quad=8
    )
    M_nH = float(res['M_nH'])

    l_tx = float(np.sum(2.0 * tx.ns * (tx.get_turn_radii() * 1e3) * np.sin(np.pi / tx.ns)))
    l_rx = float(np.sum(2.0 * rx.ns * (rx.get_turn_radii() * 1e3) * np.sin(np.pi / rx.ns)))
    R_ohm = float(RHO_CU * (l_tx + l_rx) / WIRE_A)

    return {
        'M_nH': M_nH,
        'R_ohm': R_ohm,
        'wire_m': (l_tx + l_rx) * 1e-3,
        'feasible': LOWER_M <= M_nH <= UPPER_M,
        'pt': pt_8d
    }


def run_gp_cei_geom_optimization(seed: int, budget: int = 40, n_init: int = 8, n_cands: int = 3000):
    rng = np.random.default_rng(seed)
    history, X_train, y_M, y_R = [], [], [], []

    for _ in range(n_init):
        pt = sample_geom_point(rng)
        ev = evaluate_geom(pt)
        history.append(ev)
        X_train.append(pt)
        y_M.append(ev['M_nH'])
        y_R.append(ev['R_ohm'])

    scaler = MinMaxScaler()
    scaler.fit(np.array(X_train))

    for step in range(budget - n_init):
        X_norm = scaler.transform(np.array(X_train))
        y_M_arr, y_R_arr = np.array(y_M), np.array(y_R)

        gp_M = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(8), nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1)),
            alpha=1e-6, normalize_y=True, random_state=seed
        ).fit(X_norm, y_M_arr)

        gp_R = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(8), nu=2.5) + WhiteKernel(1e-4, (1e-6, 1e-1)),
            alpha=1e-6, normalize_y=True, random_state=seed
        ).fit(X_norm, y_R_arr)

        cands = np.array([sample_geom_point(rng) for _ in range(n_cands)])
        c_norm = scaler.transform(cands)

        mu_M, sig_M = gp_M.predict(c_norm, return_std=True)
        mu_R, sig_R = gp_R.predict(c_norm, return_std=True)

        sig_M = np.maximum(sig_M, 1e-12)
        p_f = ndtr((UPPER_M - mu_M) / sig_M) - ndtr((LOWER_M - mu_M) / sig_M)

        feas = [h for h in history if h['feasible']]
        if feas:
            inc_R = min(h['R_ohm'] for h in feas)
            sig_R = np.maximum(sig_R, 1e-12)
            z = (inc_R - mu_R) / sig_R
            ei = (inc_R - mu_R) * ndtr(z) + sig_R * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            acq = p_f * np.maximum(ei, 0.0) / inc_R
        else:
            acq = p_f

        scale_M = max(float(np.std(y_M_arr)), 1e-12)
        scale_R = max(float(np.std(y_R_arr)), 1e-12)
        cov = 0.5 * sig_M / scale_M + 0.5 * sig_R / scale_R

        span_acq = np.ptp(acq)
        norm_acq = (acq - np.min(acq)) / span_acq if span_acq > 0 else np.zeros_like(acq)
        span_cov = np.ptp(cov)
        norm_cov = (cov - np.min(cov)) / span_cov if span_cov > 0 else np.zeros_like(cov)

        score = 0.75 * norm_acq + 0.25 * norm_cov
        best_pt = cands[int(np.argmax(score))]

        ev = evaluate_geom(best_pt)
        history.append(ev)
        X_train.append(best_pt)
        y_M.append(ev['M_nH'])
        y_R.append(ev['R_ohm'])

    feas = [h for h in history if h['feasible']]
    if feas:
        best = min(feas, key=lambda x: x['R_ohm'])
        return {'qualified': True, 'best_R': best['R_ohm'], 'best_M': best['M_nH'], 'best_pt': best['pt']}
    else:
        best = min(history, key=lambda x: abs(x['M_nH'] - TARGET_M))
        return {'qualified': False, 'best_R': best['R_ohm'], 'best_M': best['M_nH'], 'best_pt': best['pt']}


def evaluate_spatial_tolerance_envelope(best_pt: np.ndarray):
    """Evaluates 6-DOF spatial tolerance sensitivity for the synthesized design."""
    nom = evaluate_geom(best_pt, translation=(0.0, 0.0, NOMINAL_Z), rotation=(0.0, 0.0, 0.0))
    M_nom = nom['M_nH']
    
    # 1. Lateral offset tolerance
    lat_results = []
    for r_lat in [0.0, 5.0, 10.0, 15.0, 20.0]:
        res = evaluate_geom(best_pt, translation=(r_lat, 0.0, NOMINAL_Z))
        lat_results.append({'r_lat_mm': r_lat, 'M_nH': res['M_nH'], 'dev_pct': (res['M_nH'] - M_nom) / M_nom * 100.0})

    # 2. Axial gap tolerance
    axial_results = []
    for dz in [-10.0, -5.0, 0.0, 5.0, 10.0]:
        z = NOMINAL_Z + dz
        res = evaluate_geom(best_pt, translation=(0.0, 0.0, z))
        axial_results.append({'dz_mm': dz, 'z_mm': z, 'M_nH': res['M_nH'], 'dev_pct': (res['M_nH'] - M_nom) / M_nom * 100.0})

    # 3. Angular tilt tolerance
    tilt_results = []
    for phi_deg in [0.0, 5.0, 10.0, 15.0]:
        res = evaluate_geom(best_pt, translation=(0.0, 0.0, NOMINAL_Z), rotation=(np.radians(phi_deg), 0.0, 0.0))
        tilt_results.append({'phi_deg': phi_deg, 'M_nH': res['M_nH'], 'dev_pct': (res['M_nH'] - M_nom) / M_nom * 100.0})

    return {
        'M_nominal_nH': M_nom,
        'R_nominal_ohm': nom['R_ohm'],
        'lateral_tolerance': lat_results,
        'axial_tolerance': axial_results,
        'tilt_tolerance': tilt_results
    }


if __name__ == '__main__':
    print("=" * 80)
    print("STAGE 1: 8-D GEOMETRIC OPTIMIZATION AT NOMINAL ALIGNMENT (B=40)")
    print("=" * 80)
    res = run_gp_cei_geom_optimization(seed=1, budget=40, n_init=8)
    print(f"Optimal Design Found: R = {res['best_R']:.4f} Ohm | M = {res['best_M']:.2f} nH (Qualified={res['qualified']})")

    print("\n" + "=" * 80)
    print("STAGE 2: 6-DOF SPATIAL TOLERANCE ENVELOPE EVALUATION")
    print("=" * 80)
    tol = evaluate_spatial_tolerance_envelope(res['best_pt'])
    print(f"Nominal: M = {tol['M_nominal_nH']:.2f} nH")
    print("\nLateral displacement sensitivity:")
    for r in tol['lateral_tolerance']:
        print(f"  r = {r['r_lat_mm']:4.1f} mm: M = {r['M_nH']:7.2f} nH ({r['dev_pct']:+6.2f}%)")
    print("\nAxial gap sensitivity:")
    for a in tol['axial_tolerance']:
        print(f"  dz = {a['dz_mm']:+5.1f} mm (z={a['z_mm']:4.1f} mm): M = {a['M_nH']:7.2f} nH ({a['dev_pct']:+6.2f}%)")
    print("\nAngular tilt sensitivity:")
    for t in tol['tilt_tolerance']:
        print(f"  tilt = {t['phi_deg']:4.1f} deg: M = {t['M_nH']:7.2f} nH ({t['dev_pct']:+6.2f}%)")
