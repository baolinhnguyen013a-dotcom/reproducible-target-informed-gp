# Maintained experiment runners

- `sweep_relative_tolerance_surrogates.py`: current surrogate-budget experiment.
- `adaptive_relative_tolerance_direct.py`: current direct-optimizer experiment.
- `compare_surrogate_targets.py`: full-pipeline comparison runner, retained to reproduce the initial comparison across the four surrogate families.
- `sweep_budget_curves.py`: shared model-fitting helper required by the current surrogate runner.
- `uncertainty_review.md`: uncertainty-estimation design notes for the proposed next method.
- `run_gp_budget_27_relative.py`: exact ten-seed checkpoint for the selected GP knee budget.
- `analyze_gp_errors_budget27.py`: reproducible held-out calibration, target-bin, candidate-recall, and decision-pool error analysis for that checkpoint.

Current result locations are defined in the two current runners and correspond to
`results/coil_2d_relative_tolerance/`. The scripts are not run automatically.
