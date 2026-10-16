# Methods

The method tree follows the paper rather than the chronology of code
development.

## 1. Surrogate selection

`surrogate_selection/` contains the target-independent GP and tested neural
surrogates used to decide which uncertainty model should support subsequent
optimization. `gp_dual_response.py` is the reusable two-response GP control.

## 2. Target-informed acquisition

`target_informed_gp/gp_conventional_acquisition_baselines.py` contains the
matched conventional cEI, cLCB, and IVR comparison, plus the retained TC-FEI
research prototype. It is a comparison implementation, not the paper's final
automatic workflow.

The deterministic center initialization, feasibility/uncertainty start,
evidence guard, relative-cEI stopping rule, and final confirmation accounting
are experiment protocols. Their frozen, explicitly named entry points remain
under `essay/scripts/`, next to the evidence they generate:

- `run_automatic_gp_2d.py`: the proposed automatic workflow on the 2-D coil
  benchmark;
- `run_automatic_gp_wpt8d.py`: the same workflow transferred to the 8-D WPT
  benchmark;
- `run_center_start_control.py`: the center-start development/control code
  shared by the automatic-workflow experiments.

## 3. Direct-search baselines

`direct_search_baselines/` contains exhaustive search, simulated annealing,
the genetic algorithm, and particle-swarm optimization. Every file owns its
problem evaluation, repair, ranking, and algorithm loop. The former
all-in-one `optimization_algorithms/engine.py` was removed; running a visible
baseline no longer delegates to hidden engine code.

This limited duplication is deliberate: these small baseline scripts are
independently inspectable and correspond directly to the baseline named in
the manuscript.
