"""Self-contained historical 16-initial/80-total constrained GP-EI run.

The paper's matched B=27 acquisition study uses ``acquisition_policies.py``.
This older reference is retained to document the route to that experiment and
does not depend on a hidden shared engine.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Result:
    T: int
    n: int
    N: int
    L_mH: float
    R_ohm: float
    feasible: bool
    violation_mH: float


class CoilProblem:
    def __init__(self) -> None:
        config = json.loads(
            (Path(__file__).parents[2] / "config/coil_problem.json").read_text()
        )
        wire, geometry, target = (
            config["wire"], config["geometry"], config["inductance_target"]
        )
        self.T_bounds = (geometry["layers"]["min"], geometry["layers"]["max"])
        self.n_bounds = (geometry["turns_per_layer"]["min"],
                         geometry["turns_per_layer"]["max"])
        self.d = wire["diameter_mm"]
        self.pitch = self.d * wire["winding_factor"]
        self.D1 = geometry["inner_diameter_mm"]
        self.rho = wire["resistivity_ohm_mm"]
        self.target = target["mH"]
        self.tolerance = target["tolerance_mH"]
        self.cache: dict[tuple[int, int], Result] = {}

    def evaluate(self, T: float, n: float) -> Result:
        T = min(self.T_bounds[1], max(self.T_bounds[0], round(T)))
        n = min(self.n_bounds[1], max(self.n_bounds[0], round(n)))
        if (T, n) not in self.cache:
            N, H, thickness = T * n, n * self.pitch, T * self.pitch
            diameter = self.D1 + thickness
            inductance = (.08 * N**2 * diameter**2 * 1e-4
                          / (3 * diameter + 9 * H + 10 * thickness))
            resistance = 4 * self.rho * N * diameter / self.d**2
            violation = max(0.0, abs(inductance - self.target) - self.tolerance)
            self.cache[(T, n)] = Result(
                T, n, N, inductance, resistance, violation == 0.0, violation
            )
        return self.cache[(T, n)]


def rank(result: Result) -> tuple[float, float]:
    return (0.0, result.R_ohm) if result.feasible else (1.0, result.violation_mH)


def gp_predict(x_train: np.ndarray, y_train: np.ndarray,
               x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_mean = float(np.mean(y_train))
    y_scale = max(float(np.std(y_train)), 1e-8)
    standardized = (y_train - y_mean) / y_scale
    length_scale = .12
    distances = np.sum((x_train[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    kernel = np.exp(-.5 * distances / length_scale**2) + 1e-3 * np.eye(len(x_train))
    alpha = np.linalg.solve(kernel, standardized)
    alpha = np.clip(np.nan_to_num(alpha, nan=0., posinf=1e12, neginf=-1e12),
                    -1e12, 1e12)
    cross_distances = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** 2,
                             axis=2)
    cross_kernel = np.exp(-.5 * cross_distances / length_scale**2)
    mean = np.sum(cross_kernel * alpha[None, :], axis=1)
    projection = np.linalg.solve(kernel, cross_kernel.T)
    variance = np.maximum(1. - np.sum(cross_kernel * projection.T, axis=1), 1e-12)
    return mean * y_scale + y_mean, np.sqrt(variance) * y_scale


def normal_cdf(values: np.ndarray) -> np.ndarray:
    return .5 * (1. + np.array([math.erf(value / math.sqrt(2.)) for value in values]))


def optimize(problem: CoilProblem, seed: int = 20261013) -> Result:
    rng = random.Random(seed)
    candidates = [
        (T, n)
        for T in range(problem.T_bounds[0], problem.T_bounds[1] + 1)
        for n in range(problem.n_bounds[0], problem.n_bounds[1] + 1)
    ]
    normalized = np.array([
        [(T - problem.T_bounds[0]) / (problem.T_bounds[1] - problem.T_bounds[0]),
         (n - problem.n_bounds[0]) / (problem.n_bounds[1] - problem.n_bounds[0])]
        for T, n in candidates
    ])
    observed = set(rng.sample(range(len(candidates)), 16))
    while len(observed) < 80:
        ids = sorted(observed)
        values = [problem.evaluate(*candidates[index]) for index in ids]
        x_train = normalized[ids]
        mean_L, std_L = gp_predict(
            x_train, np.array([value.L_mH for value in values]),
            normalized[[index for index in range(len(candidates)) if index not in observed]],
        )
        unseen = [index for index in range(len(candidates)) if index not in observed]
        mean_R, std_R = gp_predict(
            x_train, np.array([value.R_ohm for value in values]), normalized[unseen]
        )
        lower = (problem.target - problem.tolerance - mean_L) / std_L
        upper = (problem.target + problem.tolerance - mean_L) / std_L
        p_feasible = normal_cdf(upper) - normal_cdf(lower)
        feasible_R = [value.R_ohm for value in values if value.feasible]
        if feasible_R:
            improvement = min(feasible_R) - mean_R
            z = improvement / std_R
            density = np.exp(-.5 * z**2) / math.sqrt(2. * math.pi)
            acquisition = p_feasible * (
                improvement * normal_cdf(z) + std_R * density
            )
        else:
            acquisition = p_feasible + .01 * std_L
        observed.add(unseen[int(np.argmax(acquisition))])
    return min((problem.evaluate(*candidates[index]) for index in observed), key=rank)


if __name__ == "__main__":
    print(optimize(CoilProblem()))
