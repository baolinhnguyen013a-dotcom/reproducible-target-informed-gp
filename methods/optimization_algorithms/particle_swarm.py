"""Integer-repaired particle-swarm entry point."""
from engine import CoilProblem, particle_swarm

if __name__ == "__main__":
    print(particle_swarm(CoilProblem()))
