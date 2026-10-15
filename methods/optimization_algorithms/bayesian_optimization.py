"""Target-conditioned surrogate-based optimizer (constrained GP-EI).

This is not a direct optimizer: the physical model supplies observations, two
Gaussian-process surrogates model inductance and resistance, and constrained
expected improvement selects each subsequent physical evaluation.  The engine
function retains its historical 16-initial/80-total protocol; the current
matched B=27 paper experiment is experiments/run_cei_budget27.py.
"""
from pathlib import Path
from engine import CoilProblem, bayesian_optimization

if __name__ == "__main__":
    config = Path(__file__).parents[2] / "config" / "coil_problem.json"
    print(bayesian_optimization(CoilProblem(config)))
