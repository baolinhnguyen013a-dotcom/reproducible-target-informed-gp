"""Self-contained integer-repaired particle-swarm baseline."""
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


def optimize(problem: CoilProblem, seed: int = 20261012) -> Result:
    rng = random.Random(seed)
    particles = []
    for _ in range(45):
        position = [rng.uniform(*problem.T_bounds), rng.uniform(*problem.n_bounds)]
        velocity = [rng.uniform(-1, 1), rng.uniform(-8, 8)]
        value = problem.evaluate(*position)
        particles.append([position, velocity, value, position[:], value])
    global_best = min((particle[2] for particle in particles), key=rank)
    global_position = next(particle[0][:] for particle in particles
                           if particle[2] == global_best)
    for _ in range(120):
        for particle in particles:
            position, velocity, _, personal_position, personal_best = particle
            for dimension in range(2):
                r1, r2 = rng.random(), rng.random()
                velocity[dimension] = (
                    .65 * velocity[dimension]
                    + 1.5 * r1 * (personal_position[dimension] - position[dimension])
                    + 1.5 * r2 * (global_position[dimension] - position[dimension])
                )
                position[dimension] += velocity[dimension]
            value = problem.evaluate(*position)
            if rank(value) < rank(personal_best):
                personal_best, personal_position = value, position[:]
            if rank(value) < rank(global_best):
                global_best, global_position = value, position[:]
            particle[2], particle[3], particle[4] = value, personal_position, personal_best
    return global_best


if __name__ == "__main__":
    print(optimize(CoilProblem()))
