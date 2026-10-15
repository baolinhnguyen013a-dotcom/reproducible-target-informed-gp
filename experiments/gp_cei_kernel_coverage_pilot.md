# GP-cEI plus kernel-coverage pilot

Date: 2026-10-11

## Motivation

The neural audit suggests that target-independent representation coverage can
improve downstream constrained decisions even when it does not minimize global
response MAE. For an RBF Gaussian process, the direct analogue is not an NTK:
the GP already has its own kernel feature space. The tested hybrid therefore
mixes target-local constrained expected improvement with balanced L/R GP
posterior-variance coverage.

## Acquisition

`a = 0.75 * unit_scale(cEI) + 0.25 * unit_scale(global GP coverage)`

The coverage term is the alpha=0.5 scaled combination of L and R posterior
standard deviations. All other settings match pure cEI: ten seeds, 87 integer
targets, six initial samples, nine acquisitions, 12 validations, and relative
L tolerance +/-2.5%.

## Results

| Metric | Pure cEI | 75% cEI + 25% coverage |
|---|---:|---:|
| Qualification rate | 100% | 100% |
| Exact optima / 870 | 860 | 863 |
| Exact-optimum rate | 98.85% | 99.20% |
| All-case mean loss | 0.0004678 | 0.0004024 |
| Targetwise-trimmed mean loss | 0.0002171 | 0.0001353 |
| Worst-seed mean loss | 0.002487 | 0.002487 |

The all-case mean reduction is 13.99%. At the paired case level, the hybrid
improves three cases, ties 867, and worsens none. The corrected cases are target
3 mH for seeds 20261015 and 20261019, and target 5 mH for seed 20261019.

The improvement is sparse: only two of ten seed means change. A 20,000-sample
paired seed bootstrap gives a mean-difference percentile interval of
[-0.000186, 0]. Because the upper endpoint reaches zero and the 25% weight has
not been independently confirmed, this is exploratory pilot evidence. The
next valid test is to freeze the 0.75/0.25 weights and run fresh seeds; sweeping
more weights on the same ten seeds would turn the comparison into tuning rather
than confirmation.

Artifacts:

- `results/01_current_relative_tolerance/06_standard_bo_budget27/cei_wl0p75_wg0p25_a0p5/results.csv`
- `results/01_current_relative_tolerance/06_standard_bo_budget27/cei_wl0p75_wg0p25_a0p5/summary.json`
