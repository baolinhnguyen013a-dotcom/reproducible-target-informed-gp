"""Standalone ensemble active learning for inductance uncertainty reduction."""
import json, math, random, warnings
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

root = Path(__file__).parents[2]
c = json.loads((root / "config" / "coil_problem.json").read_text())
s = json.loads((root / "config" / "surrogates" / "deep_ensemble.json").read_text())
w,g,q = c["wire"],c["geometry"],c["inductance_target"]
p=w["diameter_mm"]*w["winding_factor"]
points=[(T,n) for T in range(g["layers"]["min"],g["layers"]["max"]+1) for n in range(g["turns_per_layer"]["min"],g["turns_per_layer"]["max"]+1)]
X=np.array([[(T-g["layers"]["min"])/(g["layers"]["max"]-g["layers"]["min"]),(n-g["turns_per_layer"]["min"])/(g["turns_per_layer"]["max"]-g["turns_per_layer"]["min"])] for T,n in points])
def truth(i):
 T,n=points[i]; N,H,t=T*n,n*p,T*p; D=g["inner_diameter_mm"]+t
 return .08*N*N*D*D*1e-4/(3*D+9*H+10*t)
rng=random.Random(s["seed"]); observed=set(rng.sample(range(len(points)),s["initial_samples"])); history=[]
while len(observed)<min(s["maximum_true_evaluations"],len(points)):
 ids=sorted(observed); xs=X[ids]; ys=np.array([truth(i) for i in ids]); sx=StandardScaler().fit(xs); sy=StandardScaler().fit(ys.reshape(-1,1)); preds=[]
 for k in range(s["ensemble_size"]):
  nn=MLPRegressor(hidden_layer_sizes=(16,8),solver="adam",learning_rate_init=.01,alpha=.001,max_iter=1500,random_state=s["seed"]+k)
  nn.fit(sx.transform(xs),sy.transform(ys.reshape(-1,1)).ravel()); preds.append(sy.inverse_transform(nn.predict(sx.transform(X)).reshape(-1,1)).ravel())
 sigma=np.std(np.array(preds),axis=0); unobserved=np.array([i for i in range(len(points)) if i not in observed]); pick=unobserved[np.argmax(sigma[unobserved])]; max_sigma=float(sigma[pick]); history.append({"point":points[int(pick)],"sigma_mH":max_sigma})
 if max_sigma < s["sigma_tolerance_mH"]: break
 observed.add(int(pick))
# Refit after the final selected evaluation and use the ensemble mean as the
# prediction reported to the user.
ids=sorted(observed); xs=X[ids]; ys=np.array([truth(i) for i in ids]); sx=StandardScaler().fit(xs); sy=StandardScaler().fit(ys.reshape(-1,1)); final_preds=[]
for k in range(s["ensemble_size"]):
 nn=MLPRegressor(hidden_layer_sizes=(16,8),solver="adam",learning_rate_init=.01,alpha=.001,max_iter=1500,random_state=s["seed"]+100+k)
 nn.fit(sx.transform(xs),sy.transform(ys.reshape(-1,1)).ravel()); final_preds.append(sy.inverse_transform(nn.predict(sx.transform(X)).reshape(-1,1)).ravel())
mean_L=np.mean(np.array(final_preds),axis=0)
unobserved=np.array([i for i in range(len(points)) if i not in observed])
final_ids=unobserved[np.argsort(abs(mean_L[unobserved]-q["mH"]))[:s["final_validation_count"]]]
def final_truth(i):
 T,n=points[int(i)]; N,H,t=T*n,n*p,T*p; D=g["inner_diameter_mm"]+t; L=truth(int(i)); R=4*w["resistivity_ohm_mm"]*N*D/w["diameter_mm"]**2
 return {"point":[T,n],"predicted_mean_L_mH":float(mean_L[int(i)]),"true_L_mH":L,"true_R_ohm":R,"feasible":abs(L-q["mH"])<=q["tolerance_mH"]}
final_validation=[final_truth(i) for i in final_ids]
valid=[x for x in final_validation if x["feasible"]]
decision=min(valid,key=lambda x:x["true_R_ohm"]) if valid else min(final_validation,key=lambda x:abs(x["true_L_mH"]-q["mH"]))
out={"initial_samples":s["initial_samples"],"ensemble_size":s["ensemble_size"],"sigma_tolerance_mH":s["sigma_tolerance_mH"],"active_learning_true_evaluations":len(observed),"final_validation_count":s["final_validation_count"],"stopped_at_max_sigma_mH":max_sigma,"last_selected":history[-1] if history else None,"mean_rule_decision":decision,"mean_rule_validation":final_validation,"history":history}
(root/"results"/"active_learning_ensemble_result.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
