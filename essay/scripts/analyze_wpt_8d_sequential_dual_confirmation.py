#!/usr/bin/env python3
"""Audit and summarize sequential dual-proposal confirmation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ESSAY = Path(__file__).resolve().parents[1]
DATA = ESSAY/"data"
SEEDS = tuple(range(34261011, 34261021))


def load(path):
    return list(csv.DictReader(path.open(newline="")))


def bootstrap(values, rng, draws=200_000):
    values = np.asarray(values, float)
    sampled = values[rng.integers(0, len(values), (draws, len(values)))]
    return np.quantile(sampled.mean(1), [.025, .975]).tolist()


def main():
    rows = load(DATA/"wpt_8d_sequential_dual_confirmation/results.csv")
    batch = load(DATA/"wpt_8d_sigma_mixture_gp_control/results.csv")
    policies = sorted({row["policy"] for row in rows})
    table = {(row["policy"], int(row["seed"])): row for row in rows}
    baseline = {int(row["seed"]): row for row in batch
                if row["policy"] == "sigma_gp_only"}
    assert len(rows) == 50
    assert all(int(row["budget"]) == 50 for row in rows)
    assert all(row["qualified"] == "True" for row in rows)
    hybrid = [p for p in policies if p not in
              ("sequential_cei_only", "sequential_sigma_only")]
    assert all(int(table[p, s]["confirmation_cei_count"]) == 6 and
               int(table[p, s]["confirmation_sigma_count"]) == 6
               for p in hybrid for s in SEEDS)
    identical_decisions = all(
        len({table[p, seed]["design"] for p in policies}) == 1
        for seed in SEEDS)
    jaccard = {}
    reference = "alternate_cei_first"
    for policy in policies:
        values = []
        for seed in SEEDS:
            a = {tuple(x) for x in json.loads(table[reference, seed]["all_designs"])[38:]}
            b = {tuple(x) for x in json.loads(table[policy, seed]["all_designs"])[38:]}
            values.append(len(a & b)/len(a | b))
        jaccard[policy] = float(np.mean(values))
    sequential = np.asarray([float(table["sequential_cei_only", s]["best_R_ohm"])
                             for s in SEEDS])
    batched = np.asarray([float(baseline[s]["best_R_ohm"]) for s in SEEDS])
    difference = sequential-batched
    rng = np.random.default_rng(20261012)
    report = {
        "all_five_policies_qualify": "10/10",
        "all_five_policies_have_identical_final_decision_per_seed": identical_decisions,
        "mean_R_all_sequential_policies_ohm": float(sequential.mean()),
        "population_std_R_all_sequential_policies_ohm": float(sequential.std()),
        "mean_proposal_agreement_out_of_12": {
            policy: float(np.mean([int(table[policy, s]["proposal_agreement_count"])
                                   for s in SEEDS])) for policy in policies},
        "mean_confirmation_set_jaccard_vs_alternate_cei_first": jaccard,
        "decision_stage_counts": {
            stage: sum(table["sequential_cei_only", s]["decision_stage"] == stage
                       for s in SEEDS)
            for stage in ("initial", "sequential", "confirmation")},
        "comparison_with_original_batch_confirmation": {
            "batch_mean_R_ohm": float(batched.mean()),
            "sequential_mean_R_ohm": float(sequential.mean()),
            "mean_difference_ohm": float(difference.mean()),
            "percent_difference": float(100*difference.mean()/batched.mean()),
            "paired_bootstrap_95pct_ci_ohm": bootstrap(difference, rng),
            "seed_wins": int(np.sum(difference < 0)),
            "seed_ties": int(np.sum(difference == 0)),
            "seed_losses": int(np.sum(difference > 0)),
            "wilcoxon_p": float(wilcoxon(difference).pvalue),
        },
        "interpretation": (
            "The two proposers often choose different confirmation sets, but "
            "the final best design is identical under every sequential policy. "
            "Proposal gating therefore adds no decision value on these ten histories."
        ),
    }
    (DATA/"wpt_8d_sequential_dual_confirmation_analysis.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
