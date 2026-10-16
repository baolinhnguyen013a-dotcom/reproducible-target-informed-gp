# ICECE 2026 reproducibility archive

This archive supports the numerical and direct field-solver claims in **Reproducible Target-Informed Gaussian-Process Optimization for Data-Efficient Electromagnetic Design**.

## Contents

- `essay/data/`: retained row-level results, summaries, and run manifests.
- `essay/scripts/`: experiment, analysis, rendering, and audit programs used for the manuscript.
- `essay/figures/`: rendered figures, including the conference figures.
- `experiments/`: original 2-D benchmark programs.
- `results/coil_2d_relative_tolerance/`: retained original 2-D experiment outputs.
- `methods/` and `config/`: shared optimizer implementations and problem configuration.
- `essay/evidence_manifest.md`: claim-to-code-to-data map for the complete study.
- `essay/coil_surrogate_icece_5page.tex` and PDF: conference manuscript source and rendered paper.
- `essay/data/aedt_wpt_1000_matched/`: complete automatic and matched
  Sobol-6+cEI-14 Maxwell run packages, including ledgers, source, frozen
  confirmation plans, logs, summaries, and the selected automatic AEDT case.

The third-party source article PDF is not redistributed. Its public citation and DOI are given in the manuscript; `essay/wpt_source_scope.md` records how its 14-input model is restricted to the nominal-pose eight-variable benchmark.

## Integrity audit

From the archive's `algorithm_discussion` directory, run:

```sh
python3 essay/scripts/audit_revised_manuscript.py --numerical-only
python3 essay/scripts/audit_aedt_wpt_matched.py
```

The first audit independently recomputes the bounded losses and budget identities; verifies the 870-case acquisition and random-start records; checks the 73 held-out stopping trajectories, deterministic replay, calibration summaries, confirmation ablation, and 1,000-seed initialization screen; and validates the automated and 21-target numerical WPT records. The second checks every reported value in the matched 1000-nH Maxwell comparison, including accepted budgets, unique designs, feasibility counts, selected responses, natural stopping, and the disclosed mesh difference. The `--numerical-only` option skips only the private working-directory inventory, which includes the third-party source PDF and superseded manuscript assets that are intentionally not redistributed.

## Figure regeneration

The conference figures can be regenerated with:

```sh
python3 essay/scripts/render_figures.py
python3 essay/scripts/render_automated_stopping.py
python3 essay/scripts/render_wpt_selected_geometry.py
```

The principal Python dependencies are NumPy, SciPy, pandas, scikit-learn, and Matplotlib. Run manifests retain the experiment-specific settings and seeds. Rendering does not rerun the expensive experiment sweeps.

## Manuscript build

From `essay/`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
```

The supplied PDF is five-page, US Letter, IEEE conference format. All figures used in that PDF are raster exports to avoid Type-3 plot-font fragments.
