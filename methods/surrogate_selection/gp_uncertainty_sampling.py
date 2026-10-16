"""Gaussian-process active learning configured by config/surrogates/gaussian_process.json."""
import json, random
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

root=Path(__file__).parents[2]; c=json.loads((root/'config'/'coil_problem.json').read_text()); s=json.loads((root/'config'/'surrogates'/'gaussian_process.json').read_text())
w,g,q=c['wire'],c['geometry'],c['inductance_target']; p=w['diameter_mm']*w['winding_factor']
pts=[(T,n) for T in range(g['layers']['min'],g['layers']['max']+1) for n in range(g['turns_per_layer']['min'],g['turns_per_layer']['max']+1)]
X=np.array([[(T-1)/9,(n-20)/110] for T,n in pts])
def ev(i):
 T,n=pts[i]; N,H,t=T*n,n*p,T*p; D=g['inner_diameter_mm']+t; L=.08*N*N*D*D*1e-4/(3*D+9*H+10*t); R=4*w['resistivity_ohm_mm']*N*D/w['diameter_mm']**2
 return {'point':[T,n],'L_mH':L,'R_ohm':R,'feasible':abs(L-q['mH'])<=q['tolerance_mH']}
rng=random.Random(s['seed']); seen=set(rng.sample(range(len(pts)),s['initial_samples']))
for _ in range(s['maximum_true_evaluations']-s['initial_samples']):
 ids=sorted(seen); y=np.array([ev(i)['L_mH'] for i in ids]); gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*RBF([.25,.25])+WhiteKernel(1e-6),normalize_y=True,random_state=s['seed'],n_restarts_optimizer=3).fit(X[ids],y)
 mean,std=gp.predict(X,return_std=True); unseen=[i for i in range(len(pts)) if i not in seen]; seen.add(max(unseen,key=lambda i:std[i]))
ids=sorted(seen); gp=GaussianProcessRegressor(kernel=ConstantKernel(1.)*RBF([.25,.25])+WhiteKernel(1e-6),normalize_y=True,random_state=s['seed'],n_restarts_optimizer=5).fit(X[ids],np.array([ev(i)['L_mH'] for i in ids])); mean,std=gp.predict(X,return_std=True)
unseen=np.array([i for i in range(len(pts)) if i not in seen]); final_ids=unseen[np.argsort(abs(mean[unseen]-q['mH']))[:s['final_validation_count']]]; final=[dict(ev(i),predicted_mean_L_mH=float(mean[i]),sigma_L_mH=float(std[i])) for i in final_ids]; valid=[x for x in final if x['feasible']]; decision=min(valid,key=lambda x:x['R_ohm']) if valid else min(final,key=lambda x:abs(x['L_mH']-q['mH']))
out={'method':'Gaussian process','true_evaluations':len(seen),'final_validation_count':len(final),'decision':decision,'final_validation':final}; (root/'results'/'active_learning_gp_result.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
