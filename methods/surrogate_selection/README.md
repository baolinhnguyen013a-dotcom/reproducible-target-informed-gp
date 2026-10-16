# Surrogate-selection methods

These scripts implement the first stage of the paper: compare uncertainty-aware
surrogate families before choosing GP models for target-informed optimization.

- `gp_single_response.py`
- `nn_deep_ensemble.py`
- `nn_mc_dropout.py`
- `nn_fixed_sample.py`
- `gp_dual_response.py`

Configurations are in `config/surrogates/`. The matched runners in
`experiments/` execute these scripts in isolated temporary workspaces.
