# Configuration guide

`coil_problem.json` defines the shared wire properties, geometry bounds and
inductance target only.

`surrogates/neural_surrogate_scan.json` configures the original fixed-sample NN
scan: 16 training samples and a validation fraction yielding 10 grid validations.
It is read by `methods/surrogate_selection/nn_fixed_sample.py`. The
script uses the declared `seed`; there is no second implementation hidden in a
shared engine.

The other files in `surrogates/` configure the standalone active-learning methods:

| Configuration | Method script | Initial samples | Additional samples (maximum) | Final validation |
|---|---|---:|---:|---:|
| `surrogates/gaussian_process.json` | `surrogate_selection/gp_single_response.py` | 6 | 10 (16 total) | 10 |
| `surrogates/deep_ensemble.json` | `surrogate_selection/nn_deep_ensemble.py` | 6 | 10 (16 total) | 10 |
| `surrogates/mc_dropout_bnn.json` | `surrogate_selection/nn_mc_dropout.py` | 6 | 10 | 10 |
| `surrogates/target_conditioned_gp.json` | `target_informed_gp/gp_conventional_acquisition_baselines.py` | 6 | 9 (15 total) | 12 |
| `surrogates/global_dual_gp.json` | `surrogate_selection/gp_dual_response.py` | 6 | 9 (15 total) | 12 |

`maximum_true_evaluations` includes the initial samples, but excludes final
validation. A `sigma_tolerance_mH` of zero disables ensemble early stopping; it also checks
each member's nominated design separately. `ensemble_size` is the number of
networks, while `mc_dropout_samples` controls stochastic prediction passes.
`seed` controls reproducibility.

`target_conditioned_gp.json` configures the shared target-conditioned and
mixed local/global GP implementation. Its `tcfei` mode is an archived research
prototype and is not the proposed method in the current paper. The tested
representatives are conventional GP-cEI, probability-weighted GP-cLCB, and
weighted GP-IVR. The obsolete 16-initial/80-total development implementation
is retained only in Git history and is not part of the maintained comparison.

The same target-conditioned script also implements the standard paper baselines
`cei`, `clcb`, and `ivr`. For these modes, `local_weight` and `global_weight`
must be nonnegative and sum to one. `global_response_alpha` balances scaled
inductance and resistance uncertainty inside the global component. Thus the
configuration exposes both research axes without changing code:

- `local_weight = 1`: pure target-conditioned acquisition;
- `global_weight = 1`, `global_response_alpha = 0`: original global L-only GP;
- `global_weight = 1`, `global_response_alpha = 0.5`: balanced global L/R GP.

At `global_weight = 1`, the target-conditioned term is exactly zero, so the
acquisition sequence is independent of the requested target and may be reused.
Intermediate local/global weights and alpha are intended for a later two-axis
sensitivity sweep; the benchmark runner exposes them as command-line options.

`global_dual_gp.json` configures the reusable target-independent implementation
of the last case. Its response scales are fixed from the six initial samples so
the meaning of alpha does not drift during acquisition.

The BNN's existing `run_status: "failed"` is preserved historical metadata;
the script does not use it to control execution.

The former filenames were `gp_6_4_10.json`, `deep_ensemble_6_8_10.json`, and
`mc_dropout_bnn_6_10_10.json`. Budgets now live in the JSON contents rather than
filenames, so changing a budget does not require renaming files or editing paths.
All scripts load configuration relative to their own location.
