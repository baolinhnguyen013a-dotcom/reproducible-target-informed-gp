#!/usr/bin/env python3
"""Prefix ablation of the frozen terminal-confirmation ranking on development.

No GP is refit and no target is retuned: each row reuses its stored stopped
model-building history and the first n points of its frozen ranked confirmation
list.  The ten deterministic seed replays are retained in the counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_center_start_control as base


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "essay/data/center_stopping_1pct_deterministic_gp_k5/results.csv"
OUT = ROOT / "essay/data/confirmation_count_ablation"
COUNTS = (0, 1, 2, 5, 10)


def main() -> int:
    data = pd.read_csv(SOURCE)
    rows = []
    for count in COUNTS:
        qualified = exact = 0
        for record in data.itertuples(index=False):
            evaluated = (json.loads(record.model_building_ids)
                         + json.loads(record.confirmation_ids)[:count])
            feasible = [candidate for candidate in evaluated
                        if base.is_feasible(candidate, record.target_mH)]
            if not feasible:
                continue
            qualified += 1
            decision = min(feasible, key=lambda candidate: base.R_ALL[candidate])
            domain = np.flatnonzero(
                np.abs(base.L_ALL-record.target_mH)/record.target_mH <= base.EPSILON)
            optimum = int(domain[np.argmin(base.R_ALL[domain])])
            exact += decision == optimum
        rows.append({
            "confirmations": count,
            "cases": len(data),
            "qualified_cases": qualified,
            "exact_optimum_cases": exact,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "results.csv", index=False)
    summary = {
        "status": "complete",
        "source": str(SOURCE),
        "scope": "14 development targets x 10 deterministic replay labels",
        "method": "prefixes of each stored ranked confirmation list; no refit or retuning",
        "results": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
