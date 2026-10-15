"""Simulated-annealing entry point."""
from engine import CoilProblem, simulated_annealing

if __name__ == "__main__":
    print(simulated_annealing(CoilProblem()))
