"""Compare four optimizers for the two-variable air-core coil benchmark.

Decision representation: (T, n), where T is the layer count and n=N/T is the
integer turns per layer.  N=T*n is derived.  Replace CoilProblem.evaluate()
with a finite-element or measurement call to use the same optimizers with an
expensive evaluator.
"""

from __future__ import annotations

import json
import math
import random
import warnings
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# scikit-learn 1.6 on the installed Python/BLAS stack emits spurious matmul
# overflow warnings during small MLP convergence checks. Candidate values are
# always independently re-evaluated by CoilProblem before being reported.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


@dataclass(frozen=True)
class Result:
    T: int
    n: int
    N: int
    H_mm: float
    t_mm: float
    D_avg_mm: float
    L_mH: float
    R_ohm: float
    feasible: bool
    violation_mH: float


class CoilProblem:
    """Bounded problem specified in optimization_problem_setting.md."""

    def __init__(self, config_path: Path | None = None) -> None:
        config_path = config_path or Path(__file__).parents[2] / "config" / "coil_problem.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config = config
        wire = config["wire"]
        geometry = config["geometry"]
        target = config["inductance_target"]
        self.T_bounds = (geometry["layers"]["min"], geometry["layers"]["max"])
        self.n_bounds = (
            geometry["turns_per_layer"]["min"],
            geometry["turns_per_layer"]["max"],
        )
        self.d = wire["diameter_mm"]
        self.Krx = wire["winding_factor"]
        self.D1 = geometry["inner_diameter_mm"]
        self.rho = wire["resistivity_ohm_mm"]
        self.L_target = target["mH"]
        self.L_tolerance = target["tolerance_mH"]
        self.cache: dict[tuple[int, int], Result] = {}
        self.requests = 0
        self.unique_evaluations = 0

    def repair(self, T: float, n: float) -> tuple[int, int]:
        """Round a proposed point and project it into the integer domain."""
        T_int = min(self.T_bounds[1], max(self.T_bounds[0], round(T)))
        n_int = min(self.n_bounds[1], max(self.n_bounds[0], round(n)))
        return T_int, n_int

    def evaluate(self, T: float, n: float) -> Result:
        """Evaluate a candidate, caching repeated calls for expensive models."""
        self.requests += 1
        T_int, n_int = self.repair(T, n)
        key = (T_int, n_int)
        if key in self.cache:
            return self.cache[key]

        self.unique_evaluations += 1
        pitch = self.d * self.Krx
        N = T_int * n_int
        H = n_int * pitch
        thickness = T_int * pitch
        D_avg = self.D1 + thickness
        L = (0.08 * N**2 * D_avg**2 * 1e-4) / (
            3 * D_avg + 9 * H + 10 * thickness
        )
        R = 4 * self.rho * N * D_avg / self.d**2
        violation = max(0.0, abs(L - self.L_target) - self.L_tolerance)
        answer = Result(
            T=T_int,
            n=n_int,
            N=N,
            H_mm=H,
            t_mm=thickness,
            D_avg_mm=D_avg,
            L_mH=L,
            R_ohm=R,
            feasible=violation == 0.0,
            violation_mH=violation,
        )
        self.cache[key] = answer
        return answer


def rank(result: Result) -> tuple[float, float]:
    """Feasible first; then minimize R. Infeasible points minimize violation."""
    if result.feasible:
        return (0.0, result.R_ohm)
    return (1.0, result.violation_mH)


def best_of(results: list[Result]) -> Result:
    return min(results, key=rank)


def exhaustive_search(problem: CoilProblem) -> Result:
    return best_of(
        [
            problem.evaluate(T, n)
            for T in range(problem.T_bounds[0], problem.T_bounds[1] + 1)
            for n in range(problem.n_bounds[0], problem.n_bounds[1] + 1)
        ]
    )


def simulated_annealing(problem: CoilProblem, seed: int = 20261010) -> Result:
    rng = random.Random(seed)
    current = problem.evaluate(
        rng.randint(*problem.T_bounds), rng.randint(*problem.n_bounds)
    )
    best = current
    iterations = 5000
    for step in range(iterations):
        temperature = 5.0 * (0.002 / 5.0) ** (step / (iterations - 1))
        candidate = problem.evaluate(
            current.T + rng.choice((-1, 0, 1)),
            current.n + rng.choice((-4, -2, -1, 1, 2, 4)),
        )
        # Compare a scalar only for the Metropolis acceptance step. Feasibility
        # itself still has a large priority, consistent with rank().
        energy = (0 if current.feasible else 1e5 + current.violation_mH * 1e3) + current.R_ohm
        trial_energy = (
            (0 if candidate.feasible else 1e5 + candidate.violation_mH * 1e3)
            + candidate.R_ohm
        )
        if trial_energy <= energy or rng.random() < math.exp((energy - trial_energy) / temperature):
            current = candidate
        if rank(current) < rank(best):
            best = current
    return best


def genetic_algorithm(problem: CoilProblem, seed: int = 20261011) -> Result:
    rng = random.Random(seed)
    population_size = 100
    generations = 100
    population = [
        problem.evaluate(rng.randint(*problem.T_bounds), rng.randint(*problem.n_bounds))
        for _ in range(population_size)
    ]
    for _ in range(generations):
        population.sort(key=rank)
        next_population = population[:10]  # elitism
        # Random immigrants prevent a small integer population from collapsing
        # onto one feasible plateau before it samples neighboring turn counts.
        next_population.extend(
            problem.evaluate(
                rng.randint(*problem.T_bounds), rng.randint(*problem.n_bounds)
            )
            for _ in range(10)
        )
        while len(next_population) < population_size:
            parent_a = min(rng.sample(population, 4), key=rank)
            parent_b = min(rng.sample(population, 4), key=rank)
            T = parent_a.T if rng.random() < 0.5 else parent_b.T
            n = parent_a.n if rng.random() < 0.5 else parent_b.n
            if rng.random() < 0.35:
                T += rng.choice((-3, -2, -1, 1, 2, 3))
            if rng.random() < 0.50:
                n += rng.choice((-12, -8, -4, -2, -1, 1, 2, 4, 8, 12))
            next_population.append(problem.evaluate(T, n))
        population = next_population
    return best_of(population)


def particle_swarm(problem: CoilProblem, seed: int = 20261012) -> Result:
    rng = random.Random(seed)
    particles = []
    for _ in range(45):
        position = [rng.uniform(*problem.T_bounds), rng.uniform(*problem.n_bounds)]
        velocity = [rng.uniform(-1, 1), rng.uniform(-8, 8)]
        value = problem.evaluate(*position)
        particles.append([position, velocity, value, position[:], value])
    global_best = best_of([particle[2] for particle in particles])
    global_position = next(p[0][:] for p in particles if p[2] == global_best)
    for _ in range(120):
        for position, velocity, value, personal_position, personal_best in particles:
            for dimension in range(2):
                r1, r2 = rng.random(), rng.random()
                velocity[dimension] = (
                    0.65 * velocity[dimension]
                    + 1.5 * r1 * (personal_position[dimension] - position[dimension])
                    + 1.5 * r2 * (global_position[dimension] - position[dimension])
                )
                position[dimension] += velocity[dimension]
            value = problem.evaluate(*position)
            if rank(value) < rank(personal_best):
                personal_best = value
                personal_position[:] = position
            if rank(value) < rank(global_best):
                global_best = value
                global_position = position[:]
            # Persist updates after the unpacked loop variables are changed.
            for particle in particles:
                if particle[0] is position:
                    particle[2] = value
                    particle[3] = personal_position
                    particle[4] = personal_best
                    break
    return global_best


def _gp_predict(x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Small RBF Gaussian-process regressor for the two normalized variables."""
    y_mean = float(np.mean(y_train))
    y_scale = max(float(np.std(y_train)), 1e-8)
    y = (y_train - y_mean) / y_scale
    # Both decision variables are normalized to [0, 1].
    length_scale = 0.12
    squared_distance = np.sum((x_train[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    # A small nugget is necessary because nearby integer candidates can have
    # nearly identical kernel rows as the observation set grows.
    kernel = np.exp(-0.5 * squared_distance / length_scale**2) + 1e-3 * np.eye(len(x_train))
    # Direct solves are more stable than hand-composed triangular solves here
    # on the small, discrete data sets used by this benchmark.
    alpha = np.linalg.solve(kernel, y)
    # Guard the acquisition loop from a pathological surrogate fit; this is
    # preferable to allowing one unstable fit to select NaN candidates.
    alpha = np.clip(np.nan_to_num(alpha, nan=0.0, posinf=1e12, neginf=-1e12), -1e12, 1e12)
    cross_distance = np.sum((x_query[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    cross_kernel = np.exp(-0.5 * cross_distance / length_scale**2)
    mean = np.sum(cross_kernel * alpha[None, :], axis=1)
    projection = np.linalg.solve(kernel, cross_kernel.T)
    variance = np.maximum(1.0 - np.sum(cross_kernel * projection.T, axis=1), 1e-12)
    return mean * y_scale + y_mean, np.sqrt(variance) * y_scale


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.array([math.erf(value / math.sqrt(2.0)) for value in values]))


def bayesian_optimization(problem: CoilProblem, seed: int = 20261013) -> Result:
    """Constrained BO over the discrete domain, with 80 expensive evaluations.

    A separate GP predicts L and R.  The acquisition is expected improvement
    in resistance multiplied by the probability that L is in tolerance.
    """
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
    observed_indices = set(rng.sample(range(len(candidates)), 16))

    while len(observed_indices) < 80:
        observed = sorted(observed_indices)
        observations = [problem.evaluate(*candidates[index]) for index in observed]
        x_train = normalized[observed]
        L_train = np.array([item.L_mH for item in observations])
        R_train = np.array([item.R_ohm for item in observations])
        unobserved = [index for index in range(len(candidates)) if index not in observed_indices]
        x_query = normalized[unobserved]
        L_mean, L_std = _gp_predict(x_train, L_train, x_query)
        R_mean, R_std = _gp_predict(x_train, R_train, x_query)
        lower = (problem.L_target - problem.L_tolerance - L_mean) / L_std
        upper = (problem.L_target + problem.L_tolerance - L_mean) / L_std
        p_feasible = _normal_cdf(upper) - _normal_cdf(lower)
        feasible_resistances = [item.R_ohm for item in observations if item.feasible]

        if feasible_resistances:
            best_resistance = min(feasible_resistances)
            improvement = best_resistance - R_mean
            z = improvement / R_std
            pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
            expected_improvement = improvement * _normal_cdf(z) + R_std * pdf
            acquisition = p_feasible * expected_improvement
        else:
            # Before observing a feasible point, first learn where the
            # inductance acceptance band lies, with uncertainty as a tiebreak.
            acquisition = p_feasible + 0.01 * L_std
        observed_indices.add(unobserved[int(np.argmax(acquisition))])

    return best_of([problem.evaluate(*candidates[index]) for index in observed_indices])


def neural_surrogate_scan(problem: CoilProblem, seed: int = 20261014) -> Result:
    """Twenty uniform samples, neural surrogates, a full predicted scan, then validation.

    The last validation step is essential: twenty samples are not enough to
    certify a neural-network prediction in a constrained design problem.
    """
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
    settings_path = Path(__file__).parents[2] / "config" / "surrogates" / "neural_surrogate_scan.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    sample_count = settings["training_samples"]
    sample_indices = rng.sample(range(len(candidates)), sample_count)
    samples = [problem.evaluate(*candidates[index]) for index in sample_indices]
    x_train = normalized[sample_indices]
    x_scaler = StandardScaler().fit(x_train)
    x_scaled = x_scaler.transform(x_train)

    def predict(values: list[float], offset: int) -> np.ndarray:
        y = np.array(values)
        y_scaler = StandardScaler().fit(y.reshape(-1, 1))
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.01,
            alpha=1e-3,
            max_iter=1500,
            random_state=seed + offset,
        )
        model.fit(x_scaled, y_scaler.transform(y.reshape(-1, 1)).ravel())
        return y_scaler.inverse_transform(
            model.predict(x_scaler.transform(normalized)).reshape(-1, 1)
        ).ravel()

    L_prediction = predict([item.L_mH for item in samples], 1)
    R_prediction = predict([item.R_ohm for item in samples], 2)
    predicted_feasible = np.where(
        (L_prediction >= problem.L_target - problem.L_tolerance)
        & (L_prediction <= problem.L_target + problem.L_tolerance)
    )[0]
    if len(predicted_feasible):
        # Validate the top 1% (12) by predicted resistance rather than
        # declaring a neural prediction to be the physical answer.
        validation_count = math.ceil(
            settings["validation_fraction"] * len(candidates)
        )
        validation_indices = predicted_feasible[np.argsort(R_prediction[predicted_feasible])[:validation_count]]
    else:
        # If the NN predicts no feasible region, validate points nearest to
        # the required inductance instead.
        validation_indices = np.argsort(np.abs(L_prediction - problem.L_target))[:validation_count]
    validated = [problem.evaluate(*candidates[index]) for index in validation_indices]
    return best_of(samples + validated)


def serializable(result: Result) -> dict[str, float | int | bool]:
    return {key: round(value, 10) if isinstance(value, float) else value for key, value in asdict(result).items()}


def main() -> None:
    methods = {
        "exhaustive": exhaustive_search,
        "simulated_annealing": simulated_annealing,
        "genetic_algorithm": genetic_algorithm,
        "particle_swarm": particle_swarm,
        "bayesian_optimization": bayesian_optimization,
        "neural_surrogate_scan": neural_surrogate_scan,
    }
    outputs = {}
    for name, method in methods.items():
        problem = CoilProblem()
        result = method(problem)
        outputs[name] = {
            "best": serializable(result),
            "requested_evaluations": problem.requests,
            "unique_evaluations": problem.unique_evaluations,
        }

    destination = Path(__file__).parents[2] / "results" / "optimizer_results.json"
    destination.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
