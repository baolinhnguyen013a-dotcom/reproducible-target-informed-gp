"""Independent 15-design NN training experiment with top-1% validation."""
import json
import math
import random
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

methods_dir = Path(__file__).parent
workspace_dir = methods_dir.parents[1]
config_path = workspace_dir / "config" / "coil_problem.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
wire, geometry, target = (config[key] for key in ("wire", "geometry", "inductance_target"))
settings = json.loads((workspace_dir / "config" / "surrogates" / "neural_surrogate_scan.json").read_text(encoding="utf-8"))
d, K, rho, D1 = wire["diameter_mm"], wire["winding_factor"], wire["resistivity_ohm_mm"], geometry["inner_diameter_mm"]
pitch = d * K
candidates = [(T, n) for T in range(geometry["layers"]["min"], geometry["layers"]["max"] + 1) for n in range(geometry["turns_per_layer"]["min"], geometry["turns_per_layer"]["max"] + 1)]

def evaluate(T, n):
    N, H, t = T * n, n * pitch, T * pitch
    D = D1 + t
    L = 0.08 * N**2 * D**2 * 1e-4 / (3 * D + 9 * H + 10 * t)
    R = 4 * rho * N * D / d**2
    return {"T": T, "n": n, "N": N, "H_mm": H, "t_mm": t, "D_avg_mm": D, "L_mH": L, "R_ohm": R,
            "feasible": abs(L - target["mH"]) <= target["tolerance_mH"]}

normal = np.array([[(T - 1) / 9, (n - 20) / 110] for T, n in candidates])
rng = random.Random(settings["seed"])
train_ids = rng.sample(range(len(candidates)), settings["training_samples"])
train = [evaluate(*candidates[i]) for i in train_ids]
scaler_x = StandardScaler().fit(normal[train_ids])

def fit_predict(field, seed):
    y = np.array([item[field] for item in train])
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))
    model = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu", solver="adam", learning_rate_init=0.01, alpha=1e-3, max_iter=1500, random_state=seed)
    model.fit(scaler_x.transform(normal[train_ids]), scaler_y.transform(y.reshape(-1, 1)).ravel())
    return scaler_y.inverse_transform(model.predict(scaler_x.transform(normal)).reshape(-1, 1)).ravel()

L_hat, R_hat = fit_predict("L_mH", settings["seed"] + 1), fit_predict("R_ohm", settings["seed"] + 2)
count = math.ceil(settings["validation_fraction"] * len(candidates))
unseen = np.array([index for index in range(len(candidates)) if index not in train_ids])
predicted = unseen[(L_hat[unseen] >= target["mH"] - target["tolerance_mH"]) & (L_hat[unseen] <= target["mH"] + target["tolerance_mH"])]
preferred = predicted[np.argsort(R_hat[predicted])]
fallback = unseen[np.argsort(abs(L_hat[unseen] - target["mH"]))]
chosen = np.array(list(preferred) + [index for index in fallback if index not in set(preferred)], dtype=int)[:count]
tested = train + [evaluate(*candidates[i]) for i in chosen]
feasible = [item for item in tested if item["feasible"]]
result = min(feasible, key=lambda item: item["R_ohm"]) if feasible else min(tested, key=lambda item: abs(item["L_mH"] - target["mH"]))

output = {
    "training_samples": settings["training_samples"],
    "validation_fraction": settings["validation_fraction"],
    "validated_candidate_count": count,
    "best": result,
}
(workspace_dir / "results" / "neural_surrogate_15_train_result.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2))
