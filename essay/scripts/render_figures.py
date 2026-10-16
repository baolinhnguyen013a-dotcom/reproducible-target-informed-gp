"""Render publication-style figures from the archived benchmark CSV files.

The plotting code intentionally keeps interpretation out of the graphics.  The
definitions of u, the tolerance, seed treatment, and conclusions belong in the
paper captions (see ../figures/captions.md).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

METHODS = ("baseline", "gp", "ensemble")
SURROGATE_LABELS = {
    "baseline": "Plain MLP surrogate",
    "gp": "Gaussian process",
    "ensemble": "Deep ensemble",
}
DIRECT_LABELS = {
    "simulated_annealing": "Simulated annealing",
    "genetic_algorithm": "Genetic algorithm",
    "particle_swarm": "Particle swarm",
}
COLORS = {"baseline": "#4C78A8", "gp": "#F58518", "ensemble": "#54A24B",
          "simulated_annealing": "#4C78A8", "genetic_algorithm": "#F58518",
          "particle_swarm": "#54A24B"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def surrogate_rows() -> list[dict[str, str]]:
    rows = read_csv("surrogate_results_raw_10seeds.csv")
    extra = DATA / "gp_budget_27_results.csv"
    if extra.exists():
        with extra.open(newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def configure() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def smooth(ax, x, y, *, label, color) -> None:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dense_x = np.linspace(x.min(), x.max(), 400)
    ax.plot(dense_x, PchipInterpolator(x, y)(dense_x), lw=1.2, color=color, label=label)
    ax.scatter(x, y, s=10, color=color, zorder=3)


def surrogate_seed_summary(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    losses: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        losses[(row["method"], int(row["budget"]), int(row["seed"]))].append(float(row["bounded_loss_u"]))
    summary = []
    for method in METHODS:
        budgets = sorted({budget for name, budget, _ in losses if name == method})
        for budget in budgets:
            per_seed = sorted((np.mean(values), seed) for (name, b, seed), values in losses.items()
                              if name == method and b == budget)
            summary.append({"method": method, "budget": budget,
                            "mean_u": float(np.mean([x[0] for x in per_seed])),
                            "worst_seed": per_seed[-1][1], "worst_seed_u": per_seed[-1][0]})
    return summary


def render_surrogate_curves() -> None:
    rows = benchmark_rows(
        "results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv"
    )
    labels = {
        "gp_random": "GP, random sampling",
        "gp_active": "GP, uncertainty sampling",
        "ensemble_random": "10-NN ensemble, random sampling",
        "ensemble_active": "10-NN ensemble, uncertainty sampling",
    }
    colors = {"gp_random": "#9ECAE9", "gp_active": "#2171B5",
              "ensemble_random": "#FDD0A2", "ensemble_active": "#D94801"}
    styles = {"gp_random": "--", "gp_active": "-",
              "ensemble_random": "--", "ensemble_active": "-"}
    target_means = {}
    for method in labels:
        grouped = defaultdict(list)
        for row in rows:
            if row["method"] == method:
                grouped[int(float(row["target_mH"]))].append(float(row["bounded_loss_u"]))
        target_means[method] = [(target, float(np.mean(values)))
                                for target, values in sorted(grouped.items())]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.8))
    fig.subplots_adjust(left=.18, right=.98, bottom=.13, top=.98, hspace=.40)
    for method, points in target_means.items():
        x, y = map(np.asarray, zip(*points))
        axes[0].plot(x, y, lw=1.2, marker="o", markersize=2.0,
                     linestyle=styles[method], color=colors[method], label=labels[method])
    axes[0].set(ylabel="Mean bounded loss, $u$", ylim=(-.02, 1.02))
    # The panels have different horizontal variables.  Put the upper panel's
    # label above its axis so it is not visually mistaken for a shared label.
    axes[0].set_xlabel("Target inductance (mH)", labelpad=3)
    axes[0].xaxis.set_label_position("top")
    axes[0].grid(color="#D9D9D9", lw=.55)
    axes[0].legend(frameon=False, ncol=2, fontsize=6.2)

    aggregate_labels = ("All-case mean", "Trimmed mean", "Worst-seed mean")
    positions = np.arange(len(aggregate_labels))
    width = .19
    offsets = np.arange(len(labels)) - (len(labels) - 1) / 2
    for offset, method in zip(offsets, labels):
        method_rows = [row for row in rows if row["method"] == method]
        axes[1].bar(positions + offset * width, loss_summary(method_rows), width=width,
                    color=colors[method], label=labels[method])
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, aggregate_labels)
    axes[1].set_ylabel("Bounded loss, $u$ (log scale)")
    axes[1].grid(axis="y", color="#D9D9D9", lw=.55, which="both")
    axes[1].legend(frameon=False, ncol=2, fontsize=6.2)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"fig01_global_surrogate_2x2.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def direct_rows_by_target() -> dict[str, list[dict[str, float]]]:
    rows = read_csv("direct_optimizer_summary_trimmed.csv")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        grouped[r["method"]].append({"target": float(r["target_mH"]),
                                      "u": float(r["trimmed_bounded_loss_u"]),
                                      "cost": float(r["trimmed_true_evaluations"])})
    return grouped


def render_direct_optimizers() -> None:
    grouped = direct_rows_by_target()
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.14, top=.90, hspace=.16)
    for method, label in DIRECT_LABELS.items():
        values = grouped[method]
        x = [v["target"] for v in values]
        smooth(axes[0], x, [v["u"] for v in values], label=label, color=COLORS[method])
        smooth(axes[1], x, [v["cost"] for v in values], label=label, color=COLORS[method])
    axes[0].set(ylabel="Bounded resistance loss, $u$", ylim=(-.01, .20))
    axes[1].set(xlabel="Target inductance (mH)", ylabel="Physical-model evaluations")
    for axis in axes:
        axis.grid(color="#D9D9D9", lw=.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"fig02_direct_optimizers.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def gp_trimmed_by_target(rows: list[dict[str, str]], budget: int = 27) -> list[dict[str, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == "gp" and int(row["budget"]) == budget:
            grouped[int(row["target_mH"])].append(float(row["bounded_loss_u"]))
    return [{"target": target, "u": float(np.mean(sorted(values)[1:-1])), "cost": float(budget)}
            for target, values in sorted(grouped.items())]


def render_gaussian_vs_sa() -> None:
    rows = benchmark_rows(
        "results/coil_2d_relative_tolerance/surrogate_family_comparison_budget27/results.csv"
    )
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] == "gp_active":
            grouped[int(float(row["target_mH"]))].append(float(row["bounded_loss_u"]))
    gp = [{"target": target, "u": float(np.mean(sorted(values)[1:-1])), "cost": 27.0}
          for target, values in sorted(grouped.items())]
    sa = direct_rows_by_target()["simulated_annealing"]
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True)
    fig.subplots_adjust(left=.18, right=.98, bottom=.14, top=.90, hspace=.16)
    for values, label, color in ((gp, "Reusable active GP ($B=27$)", COLORS["gp"]),
                                 (sa, "Simulated annealing", COLORS["simulated_annealing"])):
        x = [v["target"] for v in values]
        smooth(axes[0], x, [v["u"] for v in values], label=label, color=color)
    smooth(axes[1], [v["target"] for v in sa], [v["cost"] for v in sa],
           label="SA: independent search", color=COLORS["simulated_annealing"])
    gp_targets = [v["target"] for v in gp]
    axes[1].plot(gp_targets, [27] * len(gp_targets), color=COLORS["gp"], lw=1.2,
                 label="Reusable active GP ($B=27$)")
    axes[1].scatter(gp_targets, [27] * len(gp_targets), s=7,
                    color=COLORS["gp"], zorder=3)
    axes[0].set(ylabel="Bounded resistance loss, $u$", ylim=(-.01, .20))
    axes[1].set(xlabel="Target inductance (mH)", ylabel="Physical-model evaluations", ylim=(0, 160))
    for axis in axes:
        axis.grid(color="#D9D9D9", lw=.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.5, .99), frameon=False)
    axes[1].legend(loc="upper right", frameon=False, fontsize=7.5)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"fig03_gaussian_process_vs_sa.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def benchmark_rows(relative_path: str) -> list[dict[str, str]]:
    with (PROJECT / relative_path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def loss_summary(rows: list[dict[str, str]]) -> tuple[float, float, float]:
    by_target: dict[int, list[float]] = defaultdict(list)
    by_seed: dict[int, list[float]] = defaultdict(list)
    all_losses = []
    for row in rows:
        loss = float(row["bounded_loss_u"])
        all_losses.append(loss)
        by_target[int(float(row["target_mH"]))].append(loss)
        by_seed[int(row["seed"])].append(loss)
    trimmed = [np.mean(sorted(values)[1:-1]) for values in by_target.values()]
    worst_seed = max(np.mean(values) for values in by_seed.values())
    return float(np.mean(all_losses)), float(np.mean(trimmed)), float(worst_seed)


def render_gp_acquisition_comparison() -> None:
    methods = {
        "Reusable active GP": benchmark_rows(
            "results/coil_2d_relative_tolerance/global_dual_gp_budget27/alpha_0p5/results.csv"),
        "GP-cEI": benchmark_rows(
            "results/coil_2d_relative_tolerance/cei_budget27/cei_results.csv"),
        "GP-cLCB": benchmark_rows(
            "results/coil_2d_relative_tolerance/standard_bo_budget27/clcb_wl1_wg0_a0p5/results.csv"),
        "GP-IVR": benchmark_rows(
            "results/coil_2d_relative_tolerance/standard_bo_budget27/ivr_wl1_wg0_a0p5/results.csv"),
    }
    target_means = {}
    for name, rows in methods.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[int(float(row["target_mH"]))].append(float(row["bounded_loss_u"]))
        target_means[name] = [(target, float(np.mean(values)))
                              for target, values in sorted(grouped.items())]

    summary = {name: loss_summary(rows) for name, rows in methods.items()}
    colors = {
        "Reusable active GP": "#E45756",
        "GP-cEI": "#4C78A8", "GP-cLCB": "#B279A2", "GP-IVR": "#54A24B",
    }
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.8))
    fig.subplots_adjust(left=.18, right=.98, bottom=.13, top=.98, hspace=.40)
    for name, points in target_means.items():
        x, y = map(np.asarray, zip(*points))
        axes[0].plot(x, y, lw=1.2, marker="o", markersize=2.0, color=colors[name], label=name)
    largest_target_mean = max(value for points in target_means.values() for _, value in points)
    axes[0].set(xlabel="Target inductance (mH)", ylabel="Mean bounded loss, $u$",
                ylim=(-.005, min(1.02, max(.125, 1.08 * largest_target_mean))))
    axes[0].grid(color="#D9D9D9", lw=.55)
    axes[0].legend(frameon=False, ncol=2, fontsize=6.4)

    labels = ("All-case mean", "Trimmed mean", "Worst-seed mean")
    positions = np.arange(len(labels))
    width = .155
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2
    for offset, name in zip(offsets, methods):
        axes[1].bar(positions + offset * width, summary[name], width=width,
                    color=colors[name], label=name)
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Bounded loss, $u$ (log scale)")
    axes[1].grid(axis="y", color="#D9D9D9", lw=.55, which="both")
    axes[1].legend(frameon=False, ncol=2, fontsize=6.4)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(FIGURES / f"fig04_gp_acquisition_comparison.{extension}",
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight", pad_inches=.02)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    configure()
    render_surrogate_curves()
    render_direct_optimizers()
    render_gaussian_vs_sa()
    render_gp_acquisition_comparison()


if __name__ == "__main__":
    main()
