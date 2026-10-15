#!/usr/bin/env python3
"""First-order neural active selection, paired with original coil experiment.

Experimental reproduction, not a reproduction of the original IEEE training code.
Imported `coil_relu_100seed_benchmark.py` supplies exactly the existing training,
normalization, scoring, reference oracle and final-decision code. New acquisitions:
 - grad: variance of input Jacobians across the L ensemble, combined with U
 - neighbor: ensemble variance of slopes to ONE valid neighboring design, with U
 - ntk: full-parameter-gradient linearized network kernel posterior variance

No unobserved labels enter acquisition; truth is read only for currently selected
configurations and for retrospective final scores. Split development and confirmation.
Separate empirical performance from claimed methodological novelty. Dependencies:
numpy, scipy, sklearn, torch.
"""
import argparse,csv,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
import coil_relu_100seed_benchmark as b

FIELDS=b.FIELDS+['candidate']

@torch.no_grad()
def input_jac_variance(model,ys,normalized):
    xt=torch.as_tensor(normalized,dtype=torch.float32)
    _,z1,z2=model.parts(xt)
    m1=(z1>0).numpy().astype(np.float64)
    m2=(z2>0).numpy().astype(np.float64)
    w1=model.w1.detach().numpy().astype(np.float64)
    w2=model.w2.detach().numpy().astype(np.float64)
    w3=model.w3.detach().numpy()[:,:,0].astype(np.float64)
    J=ys*np.einsum('mda,mna,mab,mnb,mb->mnd',w1,m1,w2,m2,w3,optimize=True)
    return np.sum(J.var(axis=0),axis=-1)

@torch.no_grad()
def neighbor_variance(model,ys,normalized,steps):
    xt=torch.as_tensor(normalized,dtype=torch.float32)
    p0=model(xt).numpy()*ys
    items=[]
    for d,step in enumerate(steps):
        new=normalized.copy()
        shifts=np.where(normalized[:,d] <= 1.0-step+1e-8,step,-step)
        new[:,d]=np.clip(new[:,d]+shifts,0.0,1.0)
        pn=model(torch.as_tensor(new,dtype=torch.float32)).numpy()*ys
        slope=(pn-p0)/shifts[None,:]
        items.append(slope.var(axis=0))
    return np.mean(items,axis=0)

@torch.no_grad()
def ntk_features(model,normalized):
    """Exact finite-width ∇_theta f of each member's standardized output.

    Result shape (members, N, P=193). Includes all weights AND biases.
    Independent NN members' kernels averaged by concatenating / sqrt(M).
    """
    xt=torch.as_tensor(normalized,dtype=torch.float32)
    _,z1,z2=model.parts(xt)
    x=normalized.astype(np.float64)
    a1=z1.clamp_min(0).numpy().astype(np.float64)
    a2=z2.clamp_min(0).numpy().astype(np.float64)
    z1=z1.numpy(); z2=z2.numpy()
    w2=model.w2.detach().numpy().astype(np.float64)
    w3=model.w3.detach().numpy()[:,:,0].astype(np.float64)
    delta2=(z2>0)*w3[:,None,:]
    delta1=(z1>0)*np.einsum('mij,mnj->mni',w2,delta2,optimize=True)
    M,N=a1.shape[:2]
    w1grad=(x[None,:,:,None]*delta1[:,:,None,:]).reshape(M,N,-1)
    w2grad=(a1[:,:,:,None]*delta2[:,:,None,:]).reshape(M,N,-1)
    # list in model.w1/w2/w3/b1/b2/b3 registration order (order immaterial for dot products)
    features=np.concatenate([
        w1grad,w2grad,a2,delta1,delta2,np.ones((M,N,1))
    ],axis=-1)
    assert features.shape[-1]==193
    return features

def ntk_variance(feat,selected,regularization=1e-3):
    # (M,N,P) -> (N,M*P); each ensemble member contributes normalized feature kernel
    m,n,p=feat.shape
    f=np.transpose(feat,(1,0,2)).reshape(n,m*p)/np.sqrt(m)
    scale=np.mean(np.einsum('ij,ij->i',f,f))
    f=f/max(np.sqrt(scale),1e-12)
    fs=f[np.asarray(selected,dtype=int)]
    kernel_tt=fs@fs.T
    kernel_st=f@fs.T
    prior=np.einsum('ij,ij->i',f,f)
    jitter=regularization*np.eye(len(selected))
    try:
        v=np.linalg.solve(kernel_tt+jitter,kernel_st.T)
    except np.linalg.LinAlgError:
        v=np.linalg.lstsq(kernel_tt+jitter,kernel_st.T,rcond=None)[0]
    return np.maximum(prior-np.einsum('ij,ji->i',kernel_st,v),0.0)

def acquire(xnorm,selected,y,fit_seed,method,weight,epochs,ensemble,alpha,scales,steps):
    mask=np.ones(len(xnorm),dtype=bool)
    mask[selected]=False
    mus=[]; sigs=[]; aux=[]
    for output_index in range(2):
        seed=(int(fit_seed)*1009+100003*output_index)%(2**32)
        model,ym,ys=b.fit_nn(xnorm[selected],y[selected,output_index],ensemble,epochs,seed)
        mean,std,_=b.predict_nn(model,ym,ys,xnorm,False)
        mus.append(mean);sigs.append(std)
        if method=='grad' and output_index==0:
            aux.append(input_jac_variance(model,ys,xnorm))
        elif method=='neighbor' and output_index==0:
            aux.append(neighbor_variance(model,ys,xnorm,steps))
        elif method=='ntk':
            features=ntk_features(model,xnorm)
            aux.append(ntk_variance(features,selected))
    sig=np.column_stack(sigs)
    if scales is None: scales=np.maximum(np.percentile(sig[mask],90,axis=0),1e-9)
    U=(1-alpha)*sig[:,0]/scales[0]+alpha*sig[:,1]/scales[1]
    if method=='uncertainty':score=U
    elif method in ('grad','neighbor'):
        signal=b.normalize_relative(np.sqrt(np.maximum(aux[0],0.0)),mask)
        score=U*(1+weight*signal)
    elif method=='ntk':
        vs=np.column_stack(aux)
        nscales=np.maximum(np.percentile(np.sqrt(vs[mask]),90,axis=0),1e-9)
        score=(1-alpha)*np.sqrt(vs[:,0])/nscales[0]+alpha*np.sqrt(vs[:,1])/nscales[1]
    else:raise ValueError(method)
    score[~mask]=-np.inf
    return int(np.argmax(score)),scales

def run_one(x,y,seed,method,weight,params):
    tic=time.monotonic()
    norm=(x-x.min(0))/np.maximum(x.max(0)-x.min(0),1e-12)
    rng=np.random.default_rng(seed)
    chosen=list(rng.choice(len(y),size=6,replace=False))
    scales=None
    for k in range(params.train_count-6):
        current=np.asarray(chosen)
        # Predefined late-stage ablation: use ordinary ensemble uncertainty
        # for the first five active evaluations. From 11 points onward, use
        # first-order sensitivity disagreement. Same total label budget.
        effective_method=method
        if method in ('gradlate','neighborlate'):
            effective_method='uncertainty' if len(chosen)<11 else method.removesuffix('late')
        idx,scales=acquire(norm,current,y,seed*100+k,effective_method,weight,
                          params.epochs,params.ensemble,params.alpha,scales,
                          steps=[1/(len(np.unique(x[:,0]))-1),1/(len(np.unique(x[:,1]))-1)])
        assert idx not in chosen
        chosen.append(idx)
    selected=np.array(chosen,dtype=int)
    mu,sig,_=b.posterior_models(norm,norm[selected],y[selected],
                                seed=seed*100+params.train_count,
                                kind='ensemble',ensemble=params.ensemble,
                                epochs=params.epochs,gates=False)
    metrics=b.metric_report(y,mu,sig,selected,params.targets,params.validations,params.epsilon)
    return dict(seed=seed,method=method,weight=weight,**metrics,
                runtime_sec=time.monotonic()-tic,indices=','.join(map(str,selected)),
                train_evals=params.train_count,budget=params.budget)

def paired_report(rows,challenger,seed_offset=73737,bootstraps=5000):
    base={int(r['seed']):r for r in rows if r['method']=='uncertainty'}
    trial={int(r['seed']):r for r in rows if r['method']==challenger}
    common=sorted(base.keys()&trial.keys())
    if not common:return {}
    def vals(metric):
        a=np.asarray([float(base[s][metric]) for s in common]);c=np.asarray([float(trial[s][metric]) for s in common]);return a,c
    rng=np.random.default_rng(seed_offset)
    bb=rng.integers(0,len(common),(bootstraps,len(common)))
    report={'method':challenger,'n':len(common),'metrics':{}}
    for m in ['mean_loss','mae_L_mH','mae_R_ohm','feasible_targets','exact_optima']:
        a,c=vals(m);delta=c-a
        bmeans=delta[bb].mean(axis=1)
        report['metrics'][m]={'baseline_mean':float(a.mean()),'trial_mean':float(c.mean()),
                    'delta_mean':float(delta.mean()),
                    'CI95_percentile':[float(z) for z in np.quantile(bmeans,[.025,.975])],
                    'improved_fraction':float(np.mean(delta<0)) if m in ('mean_loss','mae_L_mH','mae_R_ohm') else float(np.mean(delta>0)),
                    'ties_fraction':float(np.mean(delta==0))}
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev','test'],required=True)
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--dev-start',type=int,default=21261011)
    parser.add_argument('--test-start',type=int,default=22261011)
    parser.add_argument('--methods',nargs='+',default=['uncertainty','grad:0.5','grad:1.0','neighbor:0.5','neighbor:1.0','ntk'])
    parser.add_argument('--epochs',type=int,default=180)
    parser.add_argument('--ensemble',type=int,default=10)
    parser.add_argument('--budget',type=int,default=27)
    parser.add_argument('--validations',type=int,default=12)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument('--epsilon',type=float,default=0.025)
    parser.add_argument('--output-dir',default='/mnt/data/coil_review/first_order_study')
    parser.add_argument('--data',help='Physical truth CSV T,n,L_mH,R_ohm; otherwise experimental reconstructed oracle')
    args=parser.parse_args()
    torch.set_num_threads(1)
    x,y=b.read_oracle_csv(args.data) if args.data else b.reconstructed_oracle()
    params=SimpleNamespace(epochs=args.epochs,ensemble=args.ensemble,budget=args.budget,
             validations=args.validations,train_count=args.budget-args.validations,
             alpha=args.alpha,epsilon=args.epsilon,targets=list(range(1,88)))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    name=out/f'{args.split}_rows.csv'
    methods=args.methods
    def parse_method(name):
        parts=name.split(':');return parts[0],(float(parts[1]) if len(parts)>1 else 0.0)
    existing={}
    if name.exists():
        with name.open(newline='') as f:
            for r in csv.DictReader(f):existing[(int(r['seed']),r['method'])]=r
    header=list(run_one.__annotations__) if False else ['seed','method','weight','mae_L_mH','rmse_L_mH','mae_R_ohm','rmse_R_ohm','mean_loss','feasible_targets','total_targets','exact_optima','runtime_sec','indices','train_evals','budget']
    rows=[]
    for i in range(args.seeds):
        seed=(args.dev_start if args.split=='dev' else args.test_start)+i
        for method_name in methods:
            if (seed,method_name) in existing:
                rows.append(existing[(seed,method_name)]);continue
            method,weight=parse_method(method_name)
            result=run_one(x,y,seed,method,weight,params)
            result['method']=method_name
            with name.open('a',newline='') as f:
                w=csv.DictWriter(f,fieldnames=header)
                if f.tell()==0:w.writeheader()
                w.writerow(result)
            rows.append(result)
        print(f'{args.split} {i+1}/{args.seeds} seeds completed',flush=True)
    baseline=[r for r in rows if r['method']=='uncertainty']
    report={'split':args.split,'oracle':'user-data' if args.data else 'reconstructed demonstration oracle',
            'seed_start':args.dev_start if args.split=='dev' else args.test_start,
            'seeds':args.seeds,'methods':methods,'epochs':args.epochs,
            'budget':args.budget,'param_ensemble':args.ensemble,
            'mean':{},'paired':[]}
    for name_m in methods:
        rs=[r for r in rows if r['method']==name_m]
        if rs:report['mean'][name_m]={k:float(np.mean([float(r[k]) for r in rs])) for k in ['mean_loss','feasible_targets','exact_optima','mae_L_mH','mae_R_ohm']}
        if name_m!='uncertainty':report['paired'].append(paired_report(rows,name_m))
    (out/f'{args.split}_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['mean'],indent=2),flush=True)
    for p in report['paired']:
        l=p['metrics']['mean_loss']; print(f"{p['method']} paired loss Δ={l['delta_mean']:+.5f} CI95={l['CI95_percentile']}",flush=True)

if __name__=='__main__':main()
