"""Standalone exhaustive search: loads no Python project module."""
import json
from pathlib import Path

c = json.loads((Path(__file__).parents[2] / "config" / "coil_problem.json").read_text())
w, g, q = c["wire"], c["geometry"], c["inductance_target"]
p = w["diameter_mm"] * w["winding_factor"]

def evaluate(T, n):
    N, H, t = T * n, n * p, T * p
    D = g["inner_diameter_mm"] + t
    L = .08 * N**2 * D**2 * 1e-4 / (3 * D + 9 * H + 10 * t)
    R = 4 * w["resistivity_ohm_mm"] * N * D / w["diameter_mm"]**2
    return {"T": T, "n": n, "N": N, "L_mH": L, "R_ohm": R,
            "feasible": abs(L - q["mH"]) <= q["tolerance_mH"]}

points = [evaluate(T, n) for T in range(g["layers"]["min"], g["layers"]["max"] + 1)
          for n in range(g["turns_per_layer"]["min"], g["turns_per_layer"]["max"] + 1)]
print(json.dumps(min([x for x in points if x["feasible"]], key=lambda x: x["R_ohm"]), indent=2))
