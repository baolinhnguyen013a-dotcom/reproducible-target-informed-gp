#!/usr/bin/env python3
"""1000-seed initial-design sensitivity screen for the 2-D coil oracle.

IMPORTANT: This is an INITIALIZATION-ONLY screen, not an end-to-end cEI or GP test.
No posterior fitting, active querying, stopping, or confirmations are modeled.

Inputs: analytical coil oracle and search domain from manuscript edition (8).
Seeds: 20261025..20262024 inclusive (1000 consecutive integer seeds).
One 6-design uniform without-replacement initial set per seed, reused for all targets.
"""
import csv
import pathlib
import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "initialization_screen_1000"
NSEEDS = 1000
BASE = 20261025
TT, NN = np.meshgrid(np.arange(1, 11), np.arange(20,131), indexing='ij')
T, n = TT.ravel(), NN.ravel()
d = 0.68
p = 1.05*d
D, H, C, N = 80+T*p, n*p, T*p, T*n
L = 0.08 * N**2 * D**2 * 1e-4 / (3*D + 9*H + 10*C)
R = 4 * 1.851e-5 * N * D / d**2
TARGETS = np.arange(1, 88)
rows=[]
seed_points=[]
for seed in range(BASE, BASE+NSEEDS):
    rng=np.random.default_rng(seed)
    inds=rng.choice(len(T), 6, replace=False)
    Li=L[inds]; Ri=R[inds]
    seed_points.append((seed,','.join(f'{int(T[j])}:{int(n[j])}' for j in inds)))
    for Lt in TARGETS:
        mask=np.abs(Li-Lt)/Lt<=.025
        min_rel_err=float(np.min(np.abs(Li-Lt)/Lt))
        # relative inductance error to the nominal constraint and optimum, if initial feasible
        all_feas=np.abs(L-Lt)/Lt<=.025
        Ropt=np.min(R[all_feas])
        init_Rbest=np.min(Ri[mask]) if np.any(mask) else float('nan')
        bounded_loss=1.0 if not np.any(mask) else float(1-Ropt/init_Rbest)
        rows.append((seed,int(Lt),int(np.any(mask)),min_rel_err,bounded_loss))

path=OUT/'seed_initialization_1000x87.csv'
with path.open('w', newline='') as f:
    w=csv.writer(f);w.writerow(['seed','target_mH','initial_feasible','min_relative_L_error','initial_bounded_loss']); w.writerows(rows)
with (OUT/'seed_initialization_points.csv').open('w', newline='') as f:
    w=csv.writer(f);w.writerow(['seed','six_designs_T:n']);w.writerows(seed_points)

print('1000 seeds, 87 targets:',len(rows),'cases')
a=np.array(rows,dtype=float)
for Lt in [1,2,3,4,5,20,47,60,87]:
    s=a[a[:,1]==Lt]
    e=s[:,3]
    print('target',Lt, 'initial feasibility',int(s[:,2].sum()),'/1000', 'min relative L error p50,p95,max',np.quantile(e,[.5,.95,1]).round(3))
# Highest average loss from initialization over all 87 targets (not end-to-end)
worst=[]
for i in range(NSEEDS):
    s=a[i*len(TARGETS):(i+1)*len(TARGETS)]
    worst.append((s[:,4].mean(), int(s[0,0]),int(s[:,2].sum()),float(s[:,3].mean())))
worst.sort(reverse=True)
print('Worst five by mean initial bounded loss:',worst[:5])
print('Best five by mean initial bounded loss:',worst[-5:])
print('Overall initial feasible cases:',int(a[:,2].sum()),'of',len(a))
print('Output:',path)
