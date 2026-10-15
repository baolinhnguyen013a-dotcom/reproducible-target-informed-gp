#!/usr/bin/env python3
"""Analyze actual full GP-cEI outcome CSV. DOES NOT execute or emulate GP optimization."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('csv',type=Path)
p.add_argument('--output-dir',type=Path,default=Path('gp_analyses'))
args=p.parse_args()
a=pd.read_csv(args.csv,dtype={'method':str,'seed':str})
need={'method','seed','target_mH','feasible','best_R_ohm','evaluations'}
if not need.issubset(a.columns): p.error('Missing columns: '+', '.join(sorted(need-set(a.columns))))
if a.empty: p.error('CSV contains no full GP outcomes. Provide real optimizer output; this is not the seed screen.')
if not set(a.feasible.dropna().astype(int)).issubset({0,1}):p.error('feasible must be 0 or 1')
if (a.target_mH<1).any() or (a.target_mH>87).any():p.error('Targets must be 1..87')
if (a.feasible.eq(1)&a.best_R_ohm.isna()).any():p.error('Feasible rows require observed best_R_ohm')
if a.duplicated(['method','seed','target_mH']).any():p.error('Duplicate method+seed+target rows; deterministic replays must not be mislabeled independent')
TT,NN=np.meshgrid(np.arange(1,11),np.arange(20,131),indexing='ij')
T,n=TT.ravel(),NN.ravel();d=.68;pt=1.05*d;D=80+T*pt
L=.08*(T*n)**2*D**2*1e-4/(3*D+9*n*pt+10*T*pt)
R=4*1.851e-5*T*n*D/(d**2)
opt={lt:float(np.min(R[np.abs(L-lt)/lt<=.025])) for lt in range(1,88)}
a['reference_R_ohm']=a.target_mH.map(opt)
a['bounded_loss']=np.where(a.feasible.eq(1),1-a.reference_R_ohm/a.best_R_ohm,1.)
if (a.bounded_loss < -1e-5).any():p.error('At least one reported feasible best_R_ohm beats exhaustive lattice optimum: input mismatch/units error')
a['bounded_loss']=a.bounded_loss.clip(0,1)
a['exact']=a.feasible.eq(1)&np.isclose(a.best_R_ohm,a.reference_R_ohm,atol=1e-8,rtol=1e-7)
by_method=a.groupby('method').agg(runs=('bounded_loss','size'),targets=('target_mH','nunique'),seed_labels=('seed','nunique'),
   feasible_frac=('feasible','mean'),exact_frac=('exact','mean'),mean_loss=('bounded_loss','mean'),
   p95_loss=('bounded_loss',lambda x: x.quantile(.95)),p99_loss=('bounded_loss',lambda x:x.quantile(.99)),
   worst_loss=('bounded_loss','max'),mean_calls=('evaluations','mean')).reset_index()
by_target=a.groupby(['method','target_mH']).agg(runs=('bounded_loss','size'),feasible_frac=('feasible','mean'),
   exact_frac=('exact','mean'),mean_loss=('bounded_loss','mean'),p95_loss=('bounded_loss',lambda x:x.quantile(.95)),
   p99_loss=('bounded_loss',lambda x:x.quantile(.99)),worst_loss=('bounded_loss','max'),
   mean_calls=('evaluations','mean')).reset_index()
args.output_dir.mkdir(parents=True,exist_ok=True)
a.to_csv(args.output_dir/'scored_complete_GP_trajectories.csv',index=False)
by_method.to_csv(args.output_dir/'GP_method_summary.csv',index=False)
by_target.to_csv(args.output_dir/'GP_targetwise_summary.csv',index=False)
print(by_method.to_string(index=False))
print('NO full GP-cEI was conducted by this analyzer; these scores are derived from the supplied actual optimizer outputs.')
