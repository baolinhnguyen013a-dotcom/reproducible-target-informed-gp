#!/usr/bin/env python3
"""Independently verify archived initialization-only counts and 2-mH stress seed."""
from pathlib import Path
import pandas as pd, numpy as np, math
BASE=Path(__file__).resolve().parents[1]/'data'/'initialization_screen_1000'
s=pd.read_csv(BASE/'seed_initialization_1000x87.csv')
p=pd.read_csv(BASE/'seed_initialization_points.csv')
p=p.rename(columns={'six_designs_T:n':'six_designs_Tn'})
assert s.shape==(87000,5)
assert len(p)==1000 and p.seed.nunique()==1000
assert sorted(s.seed.unique())==list(range(20261025,20262025))
assert list(sorted(s.target_mH.unique()))==list(range(1,88))
TT,NN=np.meshgrid(np.arange(1,11),np.arange(20,131),indexing='ij')
T,n=TT.ravel(),NN.ravel();d=.68;pitch=1.05*d;D=80+T*pitch
L=.08*(T*n)**2*D**2*1e-4/(3*D+9*n*pitch+10*T*pitch)
for t,expected in [(2,34),(3,65),(4,32),(5,72),(87,22)]:
    k=int(s[s.target_mH==t].initial_feasible.sum())
    assert k==expected,(t,k)
    K=int(sum(abs(L-t)/t<=.025))
    phit=1-math.prod((1110-K-j)/(1110-j) for j in range(6))
    print(f'{t:>2} mH: {k}/1000 empirically initial-feasible; {K} feasible lattice designs; exact hit chance {phit:.6%}')
# Independently reconstruct every recorded seed-target row from the archived six points.
lookup={(int(t),int(nn)):float(ll) for t,nn,ll in zip(T,n,L)}
seed_L=np.array([[lookup[tuple(map(int,pair.split(':')))] for pair in line.split(',')]
                 for line in p.six_designs_Tn], dtype=float)
# dataset is sorted by seed then ascending target
expected_rel=np.abs(seed_L[:,:,None]-np.arange(1,88)[None,None,:])/np.arange(1,88)[None,None,:]
expected_min=np.min(expected_rel,axis=1).reshape(-1)
expected_feas=np.any(expected_rel<=.025,axis=1).astype(int).reshape(-1)
assert np.allclose(expected_min,s.min_relative_L_error.to_numpy(),rtol=0,atol=1e-12)
assert np.array_equal(expected_feas,s.initial_feasible.to_numpy())
assert abs(s[(s.seed==20261105)&(s.target_mH==2)].min_relative_L_error.iloc[0]-10.608542163517397)<1e-8
print('VERIFIED: initialization-only input arrays and derived statistics.')
print('NOT VERIFIED: no end-to-end Gaussian-process optimizer was run.')
