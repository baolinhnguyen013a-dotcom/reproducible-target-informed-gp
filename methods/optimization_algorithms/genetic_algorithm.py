"""Genetic-algorithm entry point."""
from engine import CoilProblem, genetic_algorithm

if __name__ == "__main__":
    print(genetic_algorithm(CoilProblem()))
