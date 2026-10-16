# Methods

The method tree follows the paper rather than the chronology of code
development.

## 1. Surrogate selection

`surrogate_selection/` contains the target-independent GP and tested neural
surrogates used to decide which uncertainty model should support subsequent
optimization. `gp_dual_response.py` is the reusable two-response GP control.

## 2. Target-informed acquisition

`target_informed_gp/gp_acquisition_policies.py` contains the matched cEI, cLCB,
IVR, and retained TC-FEI prototype used by the acquisition comparison.
`gp_cei_legacy_budget80.py` is a self-contained earlier reference and is not
used for the paper's matched B=27 result.

The deterministic center initialization, feasibility/uncertainty start,
evidence guard, relative-cEI stopping rule, and final confirmation accounting
are experiment protocols. Their frozen runners therefore remain under
`essay/scripts/`, next to the evidence they generate.

## 3. Direct-search baselines

`direct_search_baselines/` contains exhaustive search, simulated annealing,
the genetic algorithm, and particle-swarm optimization. Every file owns its
problem evaluation, repair, ranking, and algorithm loop. The former
all-in-one `optimization_algorithms/engine.py` was removed; running a visible
baseline no longer delegates to hidden engine code.

This limited duplication is deliberate: these small baseline scripts are
independently inspectable and correspond directly to the baseline named in
the manuscript.
