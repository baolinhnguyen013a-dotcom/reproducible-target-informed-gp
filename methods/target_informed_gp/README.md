# Target-informed GP methods

## Conventional acquisition baselines

`gp_conventional_acquisition_baselines.py` is the maintained matched
implementation for conventional cEI, cLCB, and IVR, together with the retained
TC-FEI research prototype. It fits separate inductance and resistance GPs and
keeps every physically evaluated point in the final decision pool. This file
produces the acquisition comparison; it is **not** the proposed automatic
method.

## Proposed automatic workflow

The automatic method is an experimental protocol rather than another
acquisition formula. Its two frozen entry points are:

- `../../essay/scripts/run_automatic_gp_2d.py` for the 2-D coil benchmark;
- `../../essay/scripts/run_automatic_gp_wpt8d.py` for the 8-D WPT benchmark.

Both use deterministic center initialization, the 75/25
feasibility/constraint-uncertainty discovery rule until feasibility, ordinary
cEI afterward, deterministic five-start GP fitting, the `2d+1` evidence guard,
the 1% relative maximum-cEI stopping rule with patience three, and ten final
confirmation evaluations. `run_center_start_control.py` contains the named
center-start control and shared deterministic optimizer used by these runners.

The obsolete 80-call development implementation has been removed from the
maintained tree; its history remains available through Git tags.
