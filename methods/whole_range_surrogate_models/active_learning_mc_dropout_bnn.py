"""Approximate BNN active learning configured by config/surrogates/mc_dropout_bnn.json."""
import json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

root=Path(__file__).parents[2]; c=json.loads((root/'config'/'coil_problem.json').read_text()); s=json.loads((root/'config'/'surrogates'/'mc_dropout_bnn.json').read_text())
w,g,q=c['wire'],c['geometry'],c['inductance_target']; p=w['diameter_mm']*w['winding_factor']; torch.manual_seed(s['seed'])
pts=[(T,n) for T in range(g['layers']['min'],g['layers']['max']+1) for n in range(g['turns_per_layer']['min'],g['turns_per_layer']['max']+1)]
X=np.array([[(T-1)/9,(n-20)/110] for T,n in pts],dtype=np.float32)
def ev(i):
 T,n=pts[i]; N,H,t=T*n,n*p,T*p; D=g['inner_diameter_mm']+t; L=.08*N*N*D*D*1e-4/(3*D+9*H+10*t); R=4*w['resistivity_ohm']*N*D/w['diameter_mm']**2 if 'resistivity_ohm' in w else 4*w['resistivity_ohm_mm']*N*D/w['diameter_mm']**2
 return {'point':[T,n],'L_mH':L,'R_ohm':R,'feasible':abs(L-q['mH'])<=q['tolerance_mH']}
class Net(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(2,24),nn.ReLU(),nn.Dropout(.15),nn.Linear(24,12),nn.ReLU(),nn.Dropout(.15),nn.Linear(12,1))
 def forward(self,x): return self.net(x)
def posterior(ids,seed):
 torch.manual_seed(seed); y=np.array([ev(i)['L_mH'] for i in ids],dtype=np.float32); ym,ys=y.mean(),max(y.std(),1e-6); model=Net(); opt=torch.optim.Adam(model.parameters(),lr=.01); xx=torch.tensor(X[ids]); yy=torch.tensor((y-ym)/ys).reshape(-1,1)
 for _ in range(1200): opt.zero_grad(); loss=((model(xx)-yy)**2).mean()+1e-4*sum((z*z).sum() for z in model.parameters()); loss.backward(); opt.step()
 model.train(); allx=torch.tensor(X); samples=np.array([(model(allx).detach().numpy().ravel()*ys+ym) for _ in range(s['mc_dropout_samples'])])
 return samples.mean(0),samples.std(0)
rng=random.Random(s['seed']); seen=set(rng.sample(range(len(pts)),s['initial_samples']))
for step in range(s['maximum_true_evaluations']-s['initial_samples']):
 mean,std=posterior(sorted(seen),s['seed']+step); unseen=[i for i in range(len(pts)) if i not in seen]; seen.add(max(unseen,key=lambda i:std[i]))
mean,std=posterior(sorted(seen),s['seed']+99); final_ids=np.argsort(abs(mean-q['mH']))[:s['final_validation_count']]; final=[dict(ev(i),predicted_mean_L_mH=float(mean[i]),sigma_L_mH=float(std[i])) for i in final_ids]; valid=[x for x in final if x['feasible']]; decision=min(valid,key=lambda x:x['R_ohm']) if valid else min(final,key=lambda x:abs(x['L_mH']-q['mH']))
out={'method':'MC-dropout approximate BNN','true_evaluations':len(seen),'final_validation_count':len(final),'decision':decision,'final_validation':final}; (root/'results'/'active_learning_mc_dropout_bnn_result.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
