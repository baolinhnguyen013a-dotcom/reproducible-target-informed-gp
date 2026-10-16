# Direct-search baselines

Each script is self-contained and can be executed directly from the repository
root:

```sh
python3 methods/direct_search_baselines/search_exhaustive.py
python3 methods/direct_search_baselines/search_simulated_annealing.py
python3 methods/direct_search_baselines/search_genetic_algorithm.py
python3 methods/direct_search_baselines/search_particle_swarm.py
```

The scripts intentionally repeat the small coil evaluation and feasible-first
ranking functions. This makes each baseline auditable without a shared engine.
The manuscript's matched-budget direct-search sweep is implemented separately
in `experiments/adaptive_relative_tolerance_direct.py`.
