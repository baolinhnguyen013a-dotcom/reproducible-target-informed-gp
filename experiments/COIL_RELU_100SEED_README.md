# Coil ReLU acquisition benchmark (100 development + 100 confirmation seeds)

## What it tests

A balanced two-output acquisition uses ensemble disagreement for inductance and resistance:

`U(x) = (1-alpha)*sigma_L(x)/s_L + alpha*sigma_R(x)/s_R`

The proposed ReLU acquisition uses downstream-weighted activation changes near each candidate:

`A(x) = U(x) * (1 + lambda * normalized_ReLU_switch(x))`

`lambda=0` is **exactly the same ensemble training and sampling code** without the ReLU term. The baseline and proposed methods start with identical six coil configurations per paired seed. All use 15 global model-building evaluations (at B=27), and 12 target-specific unseen validations, with a validation-only final decision pool. GP is an optional two-output comparator. Per-method output includes held-out MAE for inductance/resistance, mean bounded resistance loss over 87 targets, feasible rate, and exact optimum recovery.

## Setup

Python 3.10+ recommended.

```bash
python -m pip install numpy scipy scikit-learn torch
```

## Execute

```bash
python coil_relu_100seed_benchmark.py --seeds 100 --split both \
  --lambdas 0 0.25 0.5 1 2 4 --budget 27 \
  --output-dir results_100_seeds
```

This performs 100 seeds for **each** lambda on a development set, selects one lambda based only on those results, then uses 100 entirely different seeds to compare only that lambda, the baseline (`lambda=0`), and active GP. Add `--include-random` to include both random-sampling GP and random-sampling ensemble baselines. This saves substantial computation and avoids selecting a winner on the confirmation set. To repeat **all** lambda settings on the second 100-seed set too, add `--test-all-lambdas` (but **do not** pick a new winner from those confirmation results).

For a short hardware check:

```bash
python coil_relu_100seed_benchmark.py --seeds 2 --epochs 5 --no-include-gp \
  --lambdas 0 1 --output-dir quick_test
```

Each completed trial is appended to `dev_rows.csv` or `test_rows.csv` immediately. Re-run with exactly the same arguments to **resume** interrupted experiments. Avoid using `--reset` unless you intend to delete the previous results.

## Very important limitations

The provided default coil evaluator reproduces equations from an **earlier modeling summary**, NOT a confirmed exact physical evaluator for your latest IEEE PDF. It uses the current PDF's full 1,110-point grid (no turn-count cap), relative ±2.5% tolerance, and 87 inductance targets. The actual original scripts have not been provided. The implemented ensemble training is **an explicit approximation**, not a confirmed replica of your prior scikit-learn code. The validation ranking is explicitly coded as the predicted inductance feasibility probability times a decreasing predicted-resistance utility, because the PDF does not specify its exact numerical scoring formula. Consequently, these results are valid **within this self-contained benchmark**, but must not be presented as exact paper reproductions.

To use your own full oracle, prepare a CSV (one row per allowable configuration) with this header:

```csv
T,n,L_mH,R_ohm
1,20,0.5,0.1
1,21,0.6,0.11
```

The numbers above are **illustrative CSV syntax only**, not actual coil data. Your real CSV must have complete configurations and all requested target bands must be feasible. Pass it using `--data my_real_grid.csv`. If each FEM evaluation is expensive, this precomputed-grid script is inappropriate for doing real-time FEM active learning; a callable FEM evaluator and a limited candidate set would instead be needed.

## Metrics and statistics

- `*_rows.csv`: one row per seed-method, saved as it completes.
- `*_summary.csv`: mean metrics across seeds.
- `*_summary.json`: per-lambda paired **bootstrap 95% CI** of the mean difference in bounded loss vs lambda=0 (negative means better), and fraction of seeds improved.
- `chosen_lambda.json`: dev-only selected lambda.
- `settings.json`: all run settings.

Matched method comparisons are paired by seed. 100 seeds alone do **not** guarantee a meaningful improvement: inspect the CI, sign of the held-out difference, and actual FEM validation.

## Comparability notes

- The latest draft models **both** inductance and resistance, so this code does too.
- Exactly 6 initial + 9 acquisitions + 12 validations when `--budget 27`.
- All algorithms are target independent during the 15 reusable model-building evaluations.
- In the latest draft, GP and 10-NN ensembles use dual response uncertainty and the same validation-only decision pool; this program follows that design at the conceptual level.
- The original manuscript's full source code is needed to match weight initialization, stopping conditions, kernel hyperparameters, data normalization, and exact validation score precisely.

## Two distinctions in the latest PDF

Each GP or ensemble method trains an L-output surrogate and an R-output surrogate. **This is not what “random sampling” vs “uncertainty sampling” means.** Those refer solely to how new configuration inputs are selected. Use `--include-random` to include both random-sampling surrogates (`random` = ensemble random, `gp_random` = GP random) alongside GP active and the ReLU-hybrid ensemble.

