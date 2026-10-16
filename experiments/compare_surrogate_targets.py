"""Compare the four existing surrogate scripts in isolated temporary workspaces.

Does nothing on import. Run explicitly; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "baseline": ("neural_surrogate_scan.py", "neural_surrogate_scan.json"),
    "gp": ("gp_uncertainty_sampling.py", "gaussian_process.json"),
    "ensemble": ("ensemble_uncertainty_sampling.py", "deep_ensemble.json"),
    "bnn": ("mc_dropout_uncertainty_sampling.py", "mc_dropout_bnn.json"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def truth_grid(config):
    """Independent exhaustive reference, never supplied to model training."""
    w, g = config["wire"], config["geometry"]
    pitch = w["diameter_mm"] * w["winding_factor"]
    grid = []
    for T in range(g["layers"]["min"], g["layers"]["max"] + 1):
        for n in range(g["turns_per_layer"]["min"], g["turns_per_layer"]["max"] + 1):
            N, H, thickness = T * n, n * pitch, T * pitch
            diameter = g["inner_diameter_mm"] + thickness
            L = .08 * N**2 * diameter**2 * 1e-4 / (3 * diameter + 9 * H + 10 * thickness)
            R = 4 * w["resistivity_ohm_mm"] * N * diameter / w["diameter_mm"]**2
            grid.append({"T": T, "n": n, "N": N, "L_mH": L, "R_ohm": R})
    return grid


def worker(workspace, method):
    """Run original code, then collect predictions and sample IDs from its globals."""
    import numpy as np

    script = workspace / "methods" / "surrogate_selection" / METHODS[method][0]
    with (workspace / "stdout.log").open("w") as log, contextlib.redirect_stdout(log):
        values = runpy.run_path(str(script), run_name="__main__")
    if method == "baseline":
        mean, sigma = values["L_hat"], None
        training = set(values["train_ids"])
        extras = {"r_prediction": np.asarray(values["R_hat"], dtype=float).tolist()}
        rules = {"default": (values["chosen"], values["result"])}
    elif method == "ensemble":
        mean = values["mean_L"]
        sigma = np.std(np.asarray(values["final_preds"]), axis=0)
        training = set(values["observed"])
        extras = {}
        rules = {"mean": (values["final_ids"], values["decision"])}
    else:
        mean, sigma = values["mean"], values["std"]
        training = set(values["seen"])
        extras = {}
        rules = {"default": (values["final_ids"], values["decision"])}
    mean = np.asarray(mean, dtype=float)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Non-finite predictions")
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Invalid uncertainty estimates")
    payload = {
        "mean": mean.tolist(), "sigma": None if sigma is None else sigma.tolist(),
        "training_ids": sorted(int(i) for i in training), "rules": {},
    }
    payload.update(extras)
    for name, (ids, decision) in rules.items():
        point = decision.get("point", [decision.get("T"), decision.get("n")])
        payload["rules"][name] = {
            "validation_ids": [int(i) for i in ids],
            "point": [int(i) for i in point],
        }
    write_json(workspace / "telemetry.json", payload)


def score(payload, grid, target, tolerance):
    feasible = [p for p in grid if abs(p["L_mH"] - target) <= tolerance]
    optimum = min(feasible, key=lambda p: p["R_ohm"]) if feasible else None
    lookup = {(p["T"], p["n"]): p for p in grid}
    means, sigmas = payload["mean"], payload["sigma"]
    if len(means) != len(grid) or (sigmas is not None and len(sigmas) != len(grid)):
        raise ValueError("Prediction grid length mismatch")
    training = set(payload["training_ids"])
    unseen = [i for i in range(len(grid)) if i not in training]
    errors = [abs(means[i] - grid[i]["L_mH"]) for i in unseen]
    diagnostics = {
        "held_out_grid_count": len(unseen),
        "held_out_mae_mH": sum(errors) / len(errors) if errors else None,
        "held_out_rmse_mH": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
        "held_out_max_error_mH": max(errors) if errors else None,
        "held_out_2sigma_coverage": (
            sum(abs(means[i] - grid[i]["L_mH"]) <= 2 * sigmas[i] for i in unseen) / len(unseen)
            if sigmas is not None and unseen else None
        ),
        "held_out_mean_sigma_mH": (
            sum(sigmas[i] for i in unseen) / len(unseen) if sigmas is not None and unseen else None
        ),
    }
    all_validated = set()
    for rule in payload["rules"].values():
        all_validated.update(rule["validation_ids"])
    rows = []
    for name, rule in payload["rules"].items():
        decision = lookup[tuple(rule["point"])]
        success = abs(decision["L_mH"] - target) <= tolerance
        validation = set(rule["validation_ids"])
        relative_regret = decision["R_ohm"] / optimum["R_ohm"] - 1 if success and optimum else None
        resistance_penalty = math.exp(100 * relative_regret) - 1 if relative_regret is not None else None
        raw_score = 100 - resistance_penalty if resistance_penalty is not None else 0
        rows.append({
            "rule": name, "status": "ok", "target_feasible": optimum is not None,
            "feasible_design_count": len(feasible), "success": success,
            "absolute_target_error_mH": abs(decision["L_mH"] - target),
            "resistance_regret_ohm": decision["R_ohm"] - optimum["R_ohm"] if success and optimum else None,
            "relative_resistance_regret": relative_regret,
            "resistance_penalty_points": resistance_penalty,
            "performance_score": max(0, raw_score),
            "decision": decision, "oracle_optimum": optimum,
            "training_unique_evaluations": len(training),
            "validation_requests": len(rule["validation_ids"]),
            "rule_unique_evaluations": len(training | validation),
            "whole_script_unique_evaluations": len(training | all_validated),
            **diagnostics,
        })
    return rows


def write_plots(rows, output, trim_extremes=0):
    """Plot per-target mean scores, optionally trimming high and low seed scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        ("baseline", "default"): "Plain surrogate",
        ("gp", "default"): "Gaussian process",
        ("ensemble", "mean"): "Deep ensemble (mean)",
        ("ensemble", "individual"): "Deep ensemble (individual)",
        ("bnn", "default"): "MC-dropout BNN",
    }
    grouped = {}
    for row in rows:
        if row.get("rule") is not None:
            grouped.setdefault((row["method"], row["rule"], row["target_mH"]), []).append(row["performance_score"])
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for (method, rule), label in labels.items():
        points = []
        for (m, r, target), values in grouped.items():
            if (m, r) != (method, rule):
                continue
            ordered = sorted(values)
            retained = ordered[trim_extremes:len(ordered) - trim_extremes] if trim_extremes else ordered
            if not retained:
                raise ValueError("Trimming removed every score; use more seeds or fewer trimmed extremes")
            points.append((target, sum(retained) / len(retained)))
        points.sort()
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", linewidth=2, label=label)
    axis.axhline(100, color="black", linestyle="--", linewidth=1, label="Exhaustive optimum")
    axis.set_ylim(0, 105)
    retained = "mean across seeds" if not trim_extremes else f"trimmed mean (remove {trim_extremes} low + {trim_extremes} high seed)"
    axis.set(xlabel="Target inductance (mH)", ylabel=f"Performance score: {retained}",
             title=f"Surrogate performance by target inductance ({retained})")
    axis.grid(alpha=.25)
    axis.legend()
    figure.savefig(output / "performance_by_target.png", dpi=180)
    figure.savefig(output / "performance_by_target.svg")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=float, default=[5, 10, 15, 20, 30, 40, 60, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261015, 20261016, 20261017])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tolerance", type=float, help="Absolute mH tolerance; default from coil_problem.json")
    parser.add_argument("--training-budget", type=int, help="Override all training totals (initial + acquired)")
    parser.add_argument("--validation-count", type=int, help="Override final validation budget for every method")
    parser.add_argument("--trim-extremes", type=int, default=0,
                        help="Remove this many lowest and highest seed scores before plotting each model/target")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per script")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args()
    if any(not math.isfinite(t) or t <= 0 for t in args.targets):
        parser.error("Targets must be finite positive mH values")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    if args.trim_extremes < 0 or 2 * args.trim_extremes >= len(args.seeds):
        parser.error("trim-extremes requires at least one seed remaining")
    config = json.loads((ROOT / "config" / "coil_problem.json").read_text())
    tolerance = args.tolerance if args.tolerance is not None else config["inductance_target"]["tolerance_mH"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        parser.error("Tolerance must be finite and positive")
    grid = truth_grid(config)
    for count in (args.training_budget, args.validation_count):
        if count is not None and not 1 <= count <= len(grid):
            parser.error("Budgets must be between 1 and the number of grid points")
    output = args.output or ROOT / "results" / ("target_sweep_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "config", output / "config_snapshot")
    source = ROOT / "methods" / "surrogate_selection"
    shutil.copytree(source, output / "scripts_snapshot", ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
    write_json(output / "manifest.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "executable": sys.executable,
        "tolerance_mH": tolerance,
        "source_sha256": {name: hashlib.sha256((source / spec[0]).read_bytes()).hexdigest() for name, spec in METHODS.items()},
        "protocol": "Fresh script per target and seed; oracle scored afterwards; original selection rules retained.",
        "performance_score": "100 if a feasible decision matches the exhaustive minimum resistance; otherwise 100 - (exp(100 * relative_resistance_increase) - 1), where relative increase is a decimal, not a percent integer. Infeasible, failed, and unreachable cases score 0.",
    })
    rows = []
    for target in args.targets:
        for seed in args.seeds:
            for method in args.methods:
                base = {"method": method, "target_mH": target, "seed": seed}
                case = output / f"case_{len(list(output.glob('case_*'))):04d}_{method}"
                case.mkdir()
                started = time.monotonic()
                try:
                    with tempfile.TemporaryDirectory(prefix="coil-sweep-") as temporary:
                        workspace = Path(temporary)
                        shutil.copytree(output / "config_snapshot", workspace / "config")
                        shutil.copytree(output / "scripts_snapshot", workspace / "methods" / "surrogate_selection")
                        (workspace / "results").mkdir()
                        problem = json.loads(json.dumps(config))
                        problem["inductance_target"] = {"mH": target, "tolerance_mH": tolerance}
                        write_json(workspace / "config" / "coil_problem.json", problem)
                        settings_path = workspace / "config" / "surrogates" / METHODS[method][1]
                        settings = json.loads(settings_path.read_text())
                        settings["seed"] = seed
                        if args.training_budget is not None:
                            settings["training_samples" if method == "baseline" else "maximum_true_evaluations"] = args.training_budget
                        if args.validation_count is not None:
                            if method == "baseline":
                                # ceil(fraction * grid_size) must equal the requested count.
                                settings["validation_fraction"] = (args.validation_count - 0.5) / len(grid)
                            else:
                                settings["final_validation_count"] = args.validation_count
                        write_json(case / "effective_settings.json", settings)
                        if method != "baseline" and not 1 <= settings["initial_samples"] <= settings["maximum_true_evaluations"] <= len(grid):
                            raise ValueError("Invalid config: initial_samples <= maximum_true_evaluations <= grid size required")
                        write_json(settings_path, settings)
                        env = dict(os.environ, PYTHONHASHSEED=str(seed % (2**32)))
                        with (case / "stderr.log").open("w") as log:
                            try:
                                subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(workspace), method],
                                               stdout=log, stderr=log, timeout=args.timeout, check=True, env=env)
                            finally:
                                if (workspace / "stdout.log").exists():
                                    shutil.copy2(workspace / "stdout.log", case / "stdout.log")
                        shutil.copytree(workspace / "results", case / "raw_results")
                        payload = json.loads((workspace / "telemetry.json").read_text())
                        write_json(case / "telemetry.json", payload)
                        rows.extend(dict(base, **row, elapsed_seconds=time.monotonic() - started) for row in score(payload, grid, target, tolerance))
                except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
                    rows.append(dict(base, status="error", error=str(error), elapsed_seconds=time.monotonic() - started))
                write_json(output / "results.json", rows)
                print(f"{method}: target={target:g}, seed={seed}, status={rows[-1]['status']}", flush=True)
    fields = ["method", "rule", "target_mH", "seed", "status", "target_feasible", "success",
              "absolute_target_error_mH", "resistance_regret_ohm", "relative_resistance_regret",
              "resistance_penalty_points", "performance_score",
              "training_unique_evaluations", "validation_requests", "rule_unique_evaluations",
              "whole_script_unique_evaluations", "held_out_rmse_mH", "held_out_2sigma_coverage", "elapsed_seconds", "error"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for method in args.methods:
        for target in args.targets:
            reachable = any(abs(p["L_mH"] - target) <= tolerance for p in grid)
            for rule in (["mean", "individual"] if method == "ensemble" else ["default"]):
                cases = [r for r in rows if r["method"] == method and r["target_mH"] == target
                         and (r.get("rule") == rule or r["status"] == "error")]
                completed = [r for r in cases if r["status"] == "ok"]
                successes = [r for r in completed if r["success"]]
                aggregates.append({
                    "method": method, "rule": rule, "target_mH": target,
                    "target_feasible": reachable, "attempts": len(cases),
                    "errors": len(cases) - len(completed), "successes": len(successes),
                    "success_rate_including_errors": len(successes) / len(cases) if reachable and cases else None,
                    "mean_resistance_regret_ohm_on_success": (
                        sum(r["resistance_regret_ohm"] for r in successes) / len(successes) if successes else None
                    ),
                })
    write_json(output / "aggregate.json", aggregates)
    write_plots(rows, output, trim_extremes=args.trim_extremes)
    print(f"Saved {output}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(Path(sys.argv[2]), sys.argv[3])
    else:
        main()
