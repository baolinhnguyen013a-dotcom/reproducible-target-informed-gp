"""Self-contained genetic-algorithm baseline for the 2-D coil lattice."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


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


def optimize(problem: CoilProblem, seed: int = 20261011) -> Result:
    rng = random.Random(seed)
    population = [
        problem.evaluate(rng.randint(*problem.T_bounds), rng.randint(*problem.n_bounds))
        for _ in range(100)
    ]
    for _ in range(100):
        population.sort(key=rank)
        next_population = population[:10]
        next_population.extend(
            problem.evaluate(rng.randint(*problem.T_bounds), rng.randint(*problem.n_bounds))
            for _ in range(10)
        )
        while len(next_population) < 100:
            parent_a = min(rng.sample(population, 4), key=rank)
            parent_b = min(rng.sample(population, 4), key=rank)
            T = parent_a.T if rng.random() < .5 else parent_b.T
            n = parent_a.n if rng.random() < .5 else parent_b.n
            if rng.random() < .35:
                T += rng.choice((-3, -2, -1, 1, 2, 3))
            if rng.random() < .50:
                n += rng.choice((-12, -8, -4, -2, -1, 1, 2, 4, 8, 12))
            next_population.append(problem.evaluate(T, n))
        population = next_population
    return min(population, key=rank)


if __name__ == "__main__":
    print(optimize(CoilProblem()))
