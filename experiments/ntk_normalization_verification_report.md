# NTK normalization-control verification

Date: 2026-10-11

## Question

The independently audited experiment compared a neural-ensemble disagreement
baseline whose L/R scales were fixed after the first active round with an NTK
criterion whose L/R scales were recomputed every round. This control asks
whether that scaling asymmetry, rather than the acquisition geometry, explains
the reported NTK improvement.

## Controlled protocol

Four paired arms were run:

1. `uncertainty_fixed`: ensemble disagreement, first-round scales retained;
2. `uncertainty_dynamic`: ensemble disagreement, scales recomputed each round;
3. `ntk_fixed`: full finite-width NTK conditional variance, first-round scales retained;
4. `ntk_dynamic`: full finite-width NTK conditional variance, scales recomputed each round.

Everything else is shared: the same 100 seeds (30261011--30261110), initial six
designs, separate 10-member 2-16-8-1 ReLU ensembles for L and R, 180 Adam
epochs, alpha=0.5, nine active acquisitions, 12 target-specific validations,
87 integer targets, relative L tolerance +/-2.5%, validation-only final decision
pool, and bounded resistance loss. Thus each arm uses 15 shared model-building
evaluations plus 12 validation evaluations for each target; `B=27` must not be
read as 27 total evaluations for the complete 87-target sweep.

## Fresh 100-seed results

| Acquisition | Mean loss | Feasible / 87 | Exact optima / 87 | Exact-optimum rate | L MAE (mH) | R MAE (ohm) | Loss Q95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ensemble disagreement, fixed | 0.07356 | 81.91 | 61.94 | 71.20% | 0.8305 | 0.1742 | 0.21575 |
| Ensemble disagreement, dynamic | 0.05818 | 83.00 | 66.36 | 76.28% | 0.8104 | 0.1692 | 0.15965 |
| Full NTK, fixed | 0.05187 | 83.33 | 70.74 | 81.31% | 0.8697 | 0.2584 | 0.09241 |
| Full NTK, dynamic | 0.04929 | 83.57 | 70.07 | 80.54% | 0.9114 | 0.2626 | 0.08543 |

Paired bootstrap comparisons use 10,000 resamples of seed-level mean-loss
differences; negative values favour the challenger.

| Challenger minus baseline | Mean difference | 95% bootstrap CI | Wins / ties / losses |
|---|---:|---:|---:|
| Dynamic disagreement - fixed disagreement | -0.01537 | [-0.03186, -0.00012] | 56 / 12 / 32 |
| Dynamic NTK - fixed NTK | -0.00258 | [-0.00681, 0.00207] | 34 / 44 / 22 |
| Fixed NTK - fixed disagreement | -0.02168 | [-0.03900, -0.00647] | 52 / 0 / 48 |
| Dynamic NTK - dynamic disagreement | -0.00889 | [-0.02380, 0.00399] | 46 / 0 / 54 |

## Interpretation

The original full-NTK improvement over the original fixed-scale disagreement
baseline is not created by giving NTK dynamic normalization: when both methods
use fixed scales, NTK retains a significant mean-loss advantage. Its benefit is
again concentrated in avoiding severe failures, as shown by the much smaller
95th-percentile loss, rather than by winning a large majority of seeds.

However, normalization is consequential. Dynamic rescaling significantly
improves ensemble disagreement in this new seed set. Once both methods receive
dynamic rescaling, NTK still has the lower mean and tail loss, but the paired
95% interval crosses zero and it wins only 46 of 100 seeds. Therefore the data
support the following restricted claims:

- full NTK is better than the original fixed-scale disagreement implementation;
- full NTK remains better under a matched fixed-scale comparison;
- this experiment does **not** establish that NTK is better than a strengthened,
  dynamically normalized disagreement baseline;
- GP remains the strongest method measured on this smooth two-dimensional
  analytical benchmark.

The higher L/R global MAE under NTK confirms that its lower optimization loss
does not arise from uniformly better response prediction. The evidence is
consistent with better acquisition coverage and lower catastrophic tail risk,
but it does not prove the proposed iso-inductance mechanism.

## Random-to-disagreement-to-NTK decision audit

A subsequently added 100-seed random-ensemble control uses the same six initial
samples, nine further model-building evaluations, final architecture, and 12
target-specific validations. Its mean loss is 0.21071, with 34.16 exact optima
and 69.97 top-ten returns per 87 targets. Dynamically normalized disagreement
improves these values to 0.05818, 66.36, and 82.63; dynamically normalized full
NTK reaches 0.04929, 70.07, and 83.34. Thus the target-independent NTK rule does
not need to know L_t to allocate a model-building set that is more useful to the
later constrained decisions. This is an optimization-performance observation,
not evidence that NTK produces uniformly better response predictions.

## Relationship to Holzmuller et al. (JMLR 2023)

The implementation uses the same core base-kernel definition as the JMLR
framework: phi(x)=grad_theta f_theta(x), with the finite-width NTK given by the
inner product of two full parameter-gradient features. It also uses the same
Gaussian-process posterior covariance transformation and a maximum-diagonal
style sequential choice.

It is not the same complete algorithm. The coil implementation averages ten
member kernels, uses separate L and R networks, combines their normalized
conditional variances, computes exact 193-parameter features without
sketching, retrains after every acquired label, and normalizes the kernel on
the complete candidate grid. The JMLR experiments use their own neural-tangent
parameterization, training-set scaling, optional sketching, and a general
batch-selection framework including LCMD and other selectors. Accordingly,
the coil result is an application and adaptation of established finite-width
NTK active-learning ideas, not a claim of NTK acquisition novelty.

## Numerical and reproducibility checks

The confirmation was replayed from scratch with an assertion that every
unobserved acquisition value must be finite. All 400 method-seed runs passed.
The strict replay reproduced every selected-index sequence and all five checked
metrics exactly (maximum absolute difference 0). The inherited NTK matrix
calculation emits overflow warnings for some rows that are already selected and
therefore excluded from acquisition; no non-finite unobserved score entered a
query decision. This warning should nevertheless be removed in a future
numerically stabilized implementation.

Artifacts:

- implementation: `experiments/ntk_normalization_control.py`;
- development: `results/coil_2d_relative_tolerance/ntk_normalization_control/development_20/`;
- confirmation: `results/coil_2d_relative_tolerance/ntk_normalization_control/confirmation_100/`;
- strict replay: `results/coil_2d_relative_tolerance/ntk_normalization_control/confirmation_100_strict_numerics/`.

These results have not yet been inserted into the IEEE manuscript.
