# Evidence data

This directory contains evidence used by the manuscript and executable audits.
Names follow three recognizable families:

- `center_*`, `random6_*`, `initialization_*`, `stopping_*`, and the unprefixed
  GP/surrogate tables support the two-dimensional coil study.
- `wpt_8d_*` and `wpt_14d_*` support the numerical WPT studies.
- `aedt_wpt_1000_matched/` contains the matched Maxwell application check.

The files remain together because the manuscript audit resolves them from one
evidence root. The authoritative mapping is `../evidence_manifest.md`; avoid
moving an individual dataset without updating and rerunning both audits.
