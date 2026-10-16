# ReLU-switch acquisition: 100-seed development and confirmation report

## Decision

Do **not** add the neuron-switch term to the manuscript as an improved method on
the basis of this experiment. The development-selected setting did not
replicate on the independent confirmation seeds and was worse on the main
optimization metrics.

## Tested acquisition

The ten-network dual-response ensemble baseline uses

\[
U(x)=(1-\alpha)\frac{\sigma_L(x)}{s_L}
     +\alpha\frac{\sigma_R(x)}{s_R},\qquad \alpha=0.5.
\]

The proposed rule multiplies this by a downstream-weighted measure of local
ReLU activation switches:

\[
A(x)=U(x)\left[1+\lambda\widetilde G(x)\right].
\]

Here, \(\widetilde G\) counts activation-state changes under small perturbations
of the two input coordinates, weights those changes by absolute downstream
paths to the output, normalizes by the unseen-set 90th percentile, and clips
the result to \([0,2]\). This is a heuristic sensitivity indicator, not a
calibrated posterior uncertainty.

## Protocol

- Exact repository analytical oracle: 1,110 configurations; the reconstructed
  script oracle differs from the repository grid by zero for both L and R.
- Relative inductance tolerance: ±2.5%; targets: 1--87 mH.
- Budget: 6 initial + 9 global acquisitions + 12 validations.
- Ten PyTorch ReLU networks for each of L and R; 180 epochs.
- Development seeds: 20261015--20261114.
- Independent confirmation seeds: 20361015--20361114.
- Development sweep: lambda = 0, 0.25, 0.5, 1, 2, 4.
- Confirmation comparison: development-selected lambda versus lambda = 0.
- Paired bootstrap: 4,000 resamples; negative loss difference favors the new
  term.

The PyTorch ensemble is not identical to the manuscript's scikit-learn
ensemble, so absolute values must not be substituted into the paper's existing
method table. The paired lambda comparison remains internally controlled,
because lambda=0 and lambda>0 use identical training code and paired initial
samples.

## Development results (100 seeds)

| lambda | Mean loss | Feasible rate | Exact optimum rate | Paired loss difference vs 0 (95% CI) |
|---:|---:|---:|---:|---:|
| 0 | 0.06956 | 94.30% | 76.53% | baseline |
| 0.25 | **0.06272** | **95.08%** | 73.14% | -0.00683 [-0.02593, 0.01235] |
| 0.5 | 0.07022 | 94.22% | 72.15% | +0.00067 [-0.01867, 0.02083] |
| 1 | 0.08339 | 93.09% | 68.08% | +0.01383 [-0.01277, 0.04130] |
| 2 | 0.08473 | 93.15% | 59.21% | +0.01518 [-0.00540, 0.03693] |
| 4 | 0.10922 | 90.92% | 53.99% | +0.03966 [0.01515, 0.06540] |

Lambda 0.25 was selected by development mean loss. Its interval nevertheless
crossed zero, and its exact-optimum rate was already 3.39 percentage points
below the baseline. Lambda 4 was significantly harmful.

## Independent confirmation (100 new seeds)

| Method | Mean loss | L MAE (mH) | R MAE (ohm) | Feasible rate | Exact optimum rate |
|---|---:|---:|---:|---:|---:|
| lambda = 0 | **0.07775** | 0.8265 | 0.1749 | **93.55%** | **73.70%** |
| lambda = 0.25 | 0.08599 | **0.8169** | **0.1594** | 92.91% | 69.67% |

Paired confirmation loss difference (lambda 0.25 minus baseline):

- mean: +0.00824;
- bootstrap 95% CI: [-0.01360, +0.02924];
- relative change in mean loss: +10.60% (worse);
- improved/tied/worse seeds: 44/0/56;
- median paired difference: +0.00528.

Although response MAE improved slightly, optimization performance worsened:
qualification fell by 0.64 percentage points and exact-optimum recovery fell by
4.03 percentage points. This demonstrates that global prediction error and
constrained-design decision quality are not interchangeable.

## Interpretation

The switch score detects proximity to learned piecewise-linear boundaries, but
such boundaries need not coincide with inductance-feasibility boundaries or
regions capable of changing the minimum-resistance feasible design. Multiplying
global disagreement by this score can therefore divert evaluations toward
network partition complexity rather than useful epistemic uncertainty. The
increasing degradation at larger lambda is consistent with that mechanism.

Before reconsidering the idea, the gate score should first be validated as an
error-ranking signal on held-out configurations. A safer later experiment could
use it only as a small tie-breaker or combine it with target-local feasibility
and improvement, but that would be a new hypothesis requiring another isolated
development/confirmation study.

## Artifacts

- Implementation: `coil_relu_100seed_benchmark.py`
- Imported instructions: `COIL_RELU_100SEED_README.md`
- Full results: `../results/coil_2d_relative_tolerance/relu_gate_acquisition/full_relu_100_plus_100/`
- The optional GP run was stopped because the supplied comparator produced
  repeated numerical warnings and excessive kernel-optimization cost. It is
  not needed for the paired lambda ablation and is not used in this conclusion.
