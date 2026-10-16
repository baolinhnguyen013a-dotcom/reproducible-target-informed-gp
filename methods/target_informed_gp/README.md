# Target-informed GP methods

`gp_acquisition_policies.py` is the maintained matched implementation for cEI,
cLCB, IVR, and the retained TC-FEI prototype. It fits separate inductance and
resistance GPs and keeps every physically evaluated point in the final decision
pool.

`gp_cei_legacy_budget80.py` is an earlier self-contained 80-call reference.
It is retained for provenance but is not used for the paper's B=27 comparison
or automatic stopping results.
