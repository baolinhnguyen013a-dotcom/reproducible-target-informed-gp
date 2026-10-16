# S3 GP–Neural Acquisition Confirmation

## Protocol provenance

The warning attached to the standalone `COIL_ANTIGRAVITY_HANDOFF` is valid:
that reconstructed neural benchmark does not contain the historical manuscript
GP-cEI implementation or its final decision protocol.  Its cEI-like numbers
must not be substituted for the manuscript GP results.

The S3 experiment instead implements the cEI and final-ranking equations in the
manuscript repository's current
`methods/whole_range_surrogate_models/target_conditioned_gp.py`.  Before fresh
confirmation, three archived manuscript cases were replayed with the new S3
runner:

| Seed | Target | Archived/new loss | Returned design | Source |
|---:|---:|---:|---:|---|
| 20261019 | 3 mH | 0.050718359858 | (5, 33) | active acquisition |
| 20261015 | 3 mH | 0.004457761244 | (6, 26) | validation |
| 20261023 | 27 mH | 0 | (10, 55) | active acquisition |

All displayed values reproduced exactly.  This establishes implementation
continuity for the checked rows, not independent external reproduction of the
whole manuscript.

## Development and frozen weight

Ten development seeds (31261011--31261020), all 87 targets, and B=20 were used
to compare beta = 0.05, 0.10, and 0.20.  The prespecified lowest-mean-loss rule
selected beta=0.20.  See
`results/coil_2d_relative_tolerance/cei_neural_hybrid/development_b20/FROZEN_WEIGHT_SELECTION.md`.

## Fresh confirmation

Confirmation used untouched seeds 32261011--32261020, all 87 targets, budgets
B=20 and B=27, and four paired acquisitions.  Every trajectory had the same six
initial observations.  B=20 therefore contained two acquisitions and B=27
contained nine; both used 12 GP-only ranked validations.  GP and NN models saw
the same measured data.  NN information affected acquisition only.

| Acquisition | B | Mean loss | Qualification | Exact | Top ten |
|---|---:|---:|---:|---:|---:|
| Pure GP-cEI | 20 | 0.193830 | 84.83% | 38.85% | 82.41% |
| cEI + GP uncertainty | 20 | 0.157394 | 88.51% | 41.15% | 85.63% |
| cEI + NN disagreement | 20 | **0.150014** | **89.20%** | 42.99% | **87.01%** |
| cEI + NN NTK | 20 | 0.153323 | 88.85% | **43.33%** | 86.55% |
| Pure GP-cEI | 27 | 0.046960 | 95.40% | 93.68% | 95.40% |
| cEI + GP uncertainty | 27 | **0.000991** | **100%** | 98.28% | **100%** |
| cEI + NN disagreement | 27 | 0.001028 | **100%** | **98.39%** | **100%** |
| cEI + NN NTK | 27 | 0.001090 | **100%** | 98.28% | **100%** |

All supplements improve the aggregate fresh-seed result over pure cEI.  At
B=20, neural disagreement gives the lowest mean loss, while NTK gives the
highest exact-optimum rate.  At B=27, ordinary GP uncertainty gives the lowest
mean loss and disagreement gives one additional exact case.  NTK is competitive
but is not the overall winner.

Ten-seed paired-bootstrap intervals versus pure cEI include or touch zero.  The
large aggregate improvement is concentrated in avoiding failures in a minority
of seeds.  These data support the claim that auxiliary exploration can improve
cEI robustness, but do not establish that NN internal NTK information is
superior to GP uncertainty or NN output disagreement.

Direct paired comparisons among the supplements are likewise inconclusive.  At
B=20, NTK minus disagreement has mean difference +0.003310 with 95% seed-
bootstrap interval [-0.006543, 0.016489]; NTK minus GP uncertainty is -0.004071
with interval [-0.041167, 0.025426].  At B=27, all three pairwise intervals
also include zero.  Accordingly, numerical ordering in the table is descriptive
rather than a statistically confirmed ranking.

## Reproducibility files

- Runner: `experiments/run_cei_neural_hybrid.py`
- Analyzer: `experiments/analyze_cei_neural_hybrid.py`
- Development data: `results/coil_2d_relative_tolerance/cei_neural_hybrid/development_b20/`
- Confirmation data and analysis: `results/coil_2d_relative_tolerance/cei_neural_hybrid/confirmation_b20_b27/`
