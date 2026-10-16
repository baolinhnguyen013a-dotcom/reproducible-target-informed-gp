# Surrogate-selection methods

These scripts implement the first stage of the paper: compare uncertainty-aware
surrogate families before choosing GP models for target-informed optimization.

- `gp_uncertainty_sampling.py`
- `ensemble_uncertainty_sampling.py`
- `mc_dropout_uncertainty_sampling.py`
- `neural_surrogate_scan.py`
- `global_dual_gp.py`

Configurations are in `config/surrogates/`. The matched runners in
`experiments/` execute these scripts in isolated temporary workspaces.
