# Surrogate review and checks — 2026-10-10

## What the scripts do

The baseline fits two MLP regressors to 15 random designs (one each for L and R).
It has no uncertainty estimate and validates candidates ranked by predicted R
inside the predicted feasible band, falling back to proximity to the target.
It considers both training and validation designs in its final decision.

GP uses posterior predictive standard deviation from a fitted RBF + white-noise
kernel. The ensemble uses disagreement between independently initialised MLPs
trained on the same data, without bootstrap resampling. MC-dropout uses the
spread of stochastic forward passes from one dropout network; this is an
approximate Bayesian technique, not exact BNN posterior inference.

All three acquire the most uncertain unobserved design across the full grid.
Acquisition does not depend on target L. Final validation is ranked by closeness
of predicted L to the target, followed by true feasibility and resistance. These
methods only choose their final answer from final-validation designs, whereas
the baseline also considers its training points. The ensemble additionally
validates each member's nomination. These differing decision rules confound an
attempt to attribute all performance differences to active learning alone.

## Actual 20 ± 0.5 mH checks

| Method | Training designs | Result |
|---|---:|---|
| GP, current 6+4+10 config | 10 | Succeeded; 10/10 final designs feasible; selected T=10, n=45 |
| Ensemble, current config | 14 | Succeeded; mean rule 1/10 feasible; individual rule 3/10 nominations feasible; both selected T=10, n=45 |
| Baseline, saved result only | 15 | Same optimum; not rerun in this check |
| MC-dropout, saved result only | 16 | Failed feasibility; not rerun in this check |

The optimum has N=450, L=19.777287339 mH, R=6.278828979 ohm.
GP total-budget-10 output is saved as `results/gp_total_budget_10_result.json`.
The workspace GP configuration is now 6 initial + 4 acquired = 10 total training
designs, followed by 10 final validations. Its normal result file was refreshed.

The GP emitted convergence and floating-point warnings; its saved candidate
numbers were finite. A successful decision does not establish numerical health
over the whole domain. The future runner explicitly rejects nonfinite full-grid
predictions and uncertainty values and retains stderr logs.

GP used fewer training designs here; that suggests sample efficiency for this
case, not general superiority. The ensemble reached its training cap with its
last acquisition sigma around 3.95 mH, far above the 0.01 mH stopping threshold.
Finding an optimum does not mean its uncertainty converged. The saved BNN
failure shows failure of this implementation/configuration on this target, not
failure of Bayesian neural networks as a class. More dropout passes reduce
Monte Carlo estimation noise but do not repair a biased predictive mean.

## Other uncertainty estimates worth testing

1. **Bayesian regression on polynomial or fixed radial-basis features.** Fit a
   compact feature expansion to the two inputs and infer a distribution over
   its coefficients. This is my first inexpensive alternative for this small,
   smooth problem. Predictive uncertainty is available in Bayesian ridge
   regression; model misspecification can still make it overconfident.
   [Official BayesianRidge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html).
2. **Bootstrap/jackknife uncertainty.** Resample observations, refit regressors,
   and measure prediction variability. This captures sensitivity to the training
   sample, complementing the current ensemble's initialisation variability.
   Jackknife/infinitesimal-jackknife methods provide a more principled alternative
   to treating raw random-forest tree spread as a standard error. With only six
   initial observations, bootstrap diversity may be poor.
   [Wager, Hastie & Efron, 2014](https://jmlr.org/papers/v15/wager14a.html).
3. **Laplace approximation, especially the final layer.** Approximate a trained
   network's parameter posterior locally and propagate it to predictions.
   It is an alternative to dropout with relatively low extra fitting cost;
   uncertainty in a fixed feature extractor can be missed by final-layer-only
   inference. [Laplace Redux](https://arxiv.org/abs/2106.14806).
4. **Conformal residual intervals.** Calibrate intervals using held-out residuals
   around any surrogate. This is an interval-calibration layer, not automatically
   a useful spatial acquisition score. Calibration data cost evaluations, and
   ordinary exchangeability assumptions do not automatically hold for adaptively
   selected samples. [Conformal validity under adaptive distributions](https://proceedings.mlr.press/v235/prinster24a.html).

## Proposed heuristic (not implemented or validated)

To reduce false confidence when all neural networks share the same bias, use

    u(x) = sqrt(sigma_ensemble(x)^2 + (c * d_nearest(x))^2)

where distances use normalised design coordinates, and c has units of mH per
normalised distance. Estimate c from leave-one-out prediction errors divided by
nearest retained-training-point distances, using an upper quantile. This adds
a distance-based uncertainty floor to disagreement. It is a heuristic score,
not a posterior standard deviation or a guaranteed confidence bound; the
small-sample calibration can be unstable.

Separately, to make acquisition respond to target L, alternate global maximum-u
queries with queries whose interval [mean-k*u, mean+k*u] overlaps the target
band. Among those, prioritise potentially low-resistance designs. Keep global
exploration so an overconfident incorrect mean cannot permanently exclude the
true feasible region. This adapts established exploration/exploitation ideas;
no claim of research novelty is made.

For a convincing next comparison, measure feasible-target success over seeds,
resistance regret, distinct true evaluations, held-out prediction error and
uncertainty coverage. An additional matched-model random-acquisition ablation
would be needed to isolate the value of uncertainty-guided acquisition itself.
The three-seed sweep across targets 5, 10, 20, 30, 40, 60, 80 and 100 mH was run
on 2026-10-10. Results and score plots are in
`results/target_sweep_20261010_8targets_3seeds/`. The 100 mH control has no
feasible grid design because the grid maximum is 87.8809 mH; it is not evidence
for or against a method.
