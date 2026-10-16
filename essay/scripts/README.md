# Reproduction scripts

Script prefixes indicate their role:

- `run_*` executes an experiment.
- `analyze_*` derives a stored summary from completed runs.
- `render_*` regenerates manuscript figures.
- `audit_*` checks published evidence and budget identities.
- `verify_*` performs a focused replay or integrity check.

For the shortest complete verification, run from the repository root:

```sh
python3 essay/scripts/audit_revised_manuscript.py --numerical-only
python3 essay/scripts/audit_aedt_wpt_matched.py
```
