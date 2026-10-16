# Experimental results

The retained two-dimensional coil results are under
`coil_2d_relative_tolerance/`. The directory name describes the benchmark;
it is not a release or experiment number.

The child directories use descriptive names rather than an ambiguous second
numbering system:

## Paper benchmarks

- `surrogate_budget_sweep/` — surrogate learning-budget comparison.
- `direct_optimizer_baselines/` — simulated annealing, genetic algorithm, and
  particle-swarm controls.
- `cei_budget27/` — target-conditioned constrained expected improvement.
- `standard_bo_budget27/` — matched cEI, cLCB, and IVR controls.
- `global_dual_gp_budget27/` — reusable global-GP control.
- `surrogate_family_comparison_budget27/` — matched GP/ensemble comparison.

## Supporting neural and uncertainty studies

- `relu_gate_acquisition/`
- `ntk_normalization_control/`
- `cei_neural_hybrid/`
- `mixed_uncertainty/`

## Development-only records

- `tcfei_smoke_test/`
- `tcfei_budget27_pilot/`
- `global_dual_gp_smoke_test/`

Development-only records are retained for traceability but are not substituted
for the frozen confirmation evidence cited by the manuscript.
