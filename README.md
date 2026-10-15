# Reproducible Target-Informed GP Optimization

Code, configurations, row-level evidence, and executable audits for
**“Reproducible Target-Informed Gaussian-Process Optimization for
Data-Efficient Electromagnetic Design.”**

Public repository: https://github.com/baolinhnguyen013a-dotcom/reproducible-target-informed-gp

The study develops an auditable low-budget constrained Bayesian-optimization
workflow. Its core contribution is reproducible allocation of observations:
deterministic center initialization, uncertainty-aware feasibility discovery,
fixed Gaussian-process hyperparameter starts, a dimension-aware evidence
guard, relative constrained-expected-improvement stopping, and a separately
charged confirmation batch. Wireless-power-transfer (WPT) experiments are
application checks of the frozen workflow, not a claim of WPT-specific theory
or universal optimizer superiority.

## Reproduced claims

- Matched GP/neural-surrogate and acquisition comparisons on a 1,110-design
  two-dimensional coil lattice.
- Frozen development/held-out protocol: 73/73 held-out exhaustive optima with
  21.81 mean oracle calls.
- Random-six versus deterministic-center adaptive-stopping evidence.
- A 21-target, eight-dimensional numerical WPT transfer with matched controls.
- A matched 1000-nH Ansys Maxwell application check: 30 accepted evaluations
  per policy, with complete ledgers and the selected automatic AEDT project.

## Quick verification

Python 3.11--3.13 is recommended. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python essay/scripts/audit_revised_manuscript.py --numerical-only
python essay/scripts/audit_aedt_wpt_matched.py
```

The first audit checks the numerical experiments, budgets, held-out results,
stopping histories, deterministic replays, and WPT controls. The second checks
every claim in the direct Maxwell comparison from the archived accepted-call
ledgers. It does not require Ansys.

## Repository map

- `methods/`: maintained surrogate and direct-search implementations.
- `experiments/`: two-dimensional benchmark runners.
- `config/`: shared physical and optimizer configurations.
- `results/01_current_relative_tolerance/`: retained benchmark outputs.
- `essay/scripts/`: WPT, stopping, analysis, rendering, and audit programs.
- `essay/data/`: row-level evidence and machine-readable summaries.
- `essay/data/aedt_wpt_1000_matched/`: complete Maxwell run archives.
- `essay/evidence_manifest.md`: claim-to-code-to-data map.
- `essay/coil_surrogate_icece_5page.pdf`: five-page conference manuscript.

## Rebuild the conference paper

A TeX distribution with `pdflatex` is required:

```sh
cd essay
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
pdflatex -interaction=nonstopmode -halt-on-error coil_surrogate_icece_5page.tex
```

## Regenerate figures

```sh
python essay/scripts/render_figures.py
python essay/scripts/render_automated_stopping.py
python essay/scripts/render_wpt_selected_geometry.py
```

Figure rendering uses retained data and does not rerun the optimization
sweeps.

## Maxwell evidence

The accepted-call ledgers, exact optimizers, solver-construction script, logs,
frozen confirmation plans, summaries, and selected AEDT case are archived in
`essay/data/aedt_wpt_1000_matched/`. Reproducing new physical calls requires
Ansys Electronics Desktop 2020.1 on Windows; auditing the reported records does
not. Failed solver launches repeat the identical frozen geometry and are not
counted as observations. The automatic and baseline winning designs used
different declared mesh paths, so the paper treats their 1.50% objective gap
as provisional pending a common mesh-convergence check.

## Reproducibility scope

The third-party WPT source article is cited but not redistributed. The public
package contains no credentials, private paths required for audit execution,
or superseded disconnected AEDT handoff artifacts. The retained experiments
are those needed to audit the published claims and their declared controls.

## Citation

Machine-readable citation metadata are provided in `CITATION.cff`. A fixed
Zenodo DOI can be added after the first public GitHub release is archived.

The manuscript author **Le Dong** and the GitHub display name
**Addy Tong Lok** refer to the same author. The paper name is retained in the
citation metadata so citations remain consistent with the publication.

## License

The code and accompanying reproducibility materials are released under the
MIT License; see `LICENSE`.
