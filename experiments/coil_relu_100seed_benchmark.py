#!/usr/bin/env python3
"""100+100-seed, paired, dual-output acquisition experiment for air-core coils.

IMPORTANT: The latest uploaded IEEE PDF describes the benchmark but does not
contain the original training scripts or full oracle formula. Hence:
  - The optional built-in oracle is reconstructed from an EARLIER modeling note;
    it is for exploratory replication, NOT a verified copy of the paper's oracle.
  - Neural training and final validation ranking are explicitly defined here,
    and may differ from the original paper's implementation.
  - For a study of your exact evaluator, pass --data actual_simulation_grid.csv,
    with columns T,n,L_mH,R_ohm. DO NOT claim exact paper reproduction until
    the original training/validation code has been matched as well.

All methods use the same initial six observations for each seed, the same
15 model-building evaluations (at B=27), and 12 unseen validations per target.
Surrogate model-building is globally reusable for 87 integer targets.
The candidate chosen for each target comes ONLY from its 12 validations,
following the latest draft's validation-only decision pool.

Example (100 seeds per lambda, separate 100-seed confirmation):
  python coil_relu_100seed_benchmark.py --split both --seeds 100 \
      --lambdas 0 0.25 0.5 1 2 4 --budget 27 \
      --output-dir coil_100seed_results

To test YOUR physical simulator output (recommended):
  python coil_relu_100seed_benchmark.py --data truth.csv --split both --seeds 100

This script NEVER uses unseen L,R in training or acquisitions; the oracle is
queried at evaluated configurations only. Full truth is used for reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

FIELDS = ["split", "seed", "method", "lambda", "budget", "train_evals",
          "mae_L_mH", "rmse_L_mH", "mae_R_ohm", "rmse_R_ohm",
          "mean_loss", "feasible_targets", "total_targets", "exact_optima", "runtime_sec"]


def reconstructed_oracle():
    """Explicitly NON-VERIFIED earlier analytical description; demonstration only."""
    t, n = np.meshgrid(np.arange(1, 11), np.arange(20, 131), indexing="ij")
    t, n = t.ravel(), n.ravel()
    wire_d_mm = 0.68
    winding_pitch_mm = wire_d_mm * 1.05
    diameter_mm = 80.0 + t * winding_pitch_mm
    h_mm = n * winding_pitch_mm
    thick_mm = t * winding_pitch_mm
    turns = t * n
    rho = 1.851e-5
    inductance_mH = (0.08 * turns**2 * diameter_mm**2 * 1e-4 /
                     (3*diameter_mm + 9*h_mm + 10*thick_mm))
    resistance_ohm = 4*rho*turns*diameter_mm/(wire_d_mm**2)
    return np.column_stack([t, n]).astype(float), np.stack([inductance_mH, resistance_ohm],axis=1)


def read_oracle_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fp:
        data = list(csv.DictReader(fp))
    cols = ("T", "n", "L_mH", "R_ohm")
    if not data or not set(cols).issubset(data[0]):
        raise ValueError("Expected columns T,n,L_mH,R_ohm")
    x = np.array([[float(r["T"]),float(r["n"])] for r in data], float)
    y = np.array([[float(r["L_mH"]),float(r["R_ohm"])] for r in data], float)
    if len(np.unique(x,axis=0)) != len(x):
        raise ValueError("Input grid has duplicate (T,n)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite input/output")
    if (y[:,1] <= 0).any():
        raise ValueError("All resistances must be >0")
    order = np.lexsort((x[:,1],x[:,0]))
    return x[order], y[order]


class VectorEnsemble(torch.nn.Module):
    """Vectorized 10-independent-member 2->16->8->1 ReLU MLP ensemble."""
    def __init__(self, count:int, seed:int):
        super().__init__()
        rg = np.random.default_rng(seed % (2**32))
        def w(i,o):
            return torch.nn.Parameter(torch.tensor(
                rg.normal(0, np.sqrt(2/(i+o)),(count,i,o)),dtype=torch.float32))
        self.w1,self.w2,self.w3=w(2,16),w(16,8),w(8,1)
        self.b1=torch.nn.Parameter(torch.zeros(count,1,16))
        self.b2=torch.nn.Parameter(torch.zeros(count,1,8))
        self.b3=torch.nn.Parameter(torch.zeros(count,1,1))
        self.count=count

    def parts(self, x):
        h = x.unsqueeze(0).expand(self.count,-1,-1)
        z1=torch.bmm(h,self.w1)+self.b1
        z2=torch.bmm(torch.relu(z1),self.w2)+self.b2
        p=torch.bmm(torch.relu(z2),self.w3)+self.b3
        return p.squeeze(-1),z1,z2

    def forward(self,x): return self.parts(x)[0]


def fit_nn(x,y,ensemble,epochs,seed):
    ym, ys = float(np.mean(y)), max(float(np.std(y)),1e-6)
    model=VectorEnsemble(ensemble,seed)
    xt=torch.tensor(np.asarray(x),dtype=torch.float32)
    target=torch.tensor((np.asarray(y)-ym)/ys,dtype=torch.float32)[None,:]
    opt=torch.optim.Adam(model.parameters(),lr=0.025)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out=model(xt)
        loss=((out-target)**2).mean()
        loss+=0.001 * sum((p*p).sum() for p in (model.w1,model.w2,model.w3))/(ensemble*100)
        loss.backward()
        opt.step()
    return model,ym,ys


@torch.no_grad()
def predict_nn(model,ym,ys,x,need_gates):
    xt=torch.tensor(x,dtype=torch.float32)
    pred,z1,z2=model.parts(xt)
    means=pred.mean(0).numpy()*ys+ym
    sigma=pred.std(0,unbiased=False).numpy()*ys
    if not need_gates:
        return means,sigma,None
    h=0.035
    # Downstream-output-weighted gate switches; this is NOT a proven error proxy.
    w2=model.w3[:,:,0].abs()
    w1=(model.w2.abs().unsqueeze(1)*w2[:,None,None,:]*(z2>0).float().unsqueeze(2)).sum(-1)
    denominator=(w1.sum(-1)+w2.sum(-1)[:,None]).clamp_min(1e-9)
    gate=torch.zeros(len(x))
    for j in range(2):
        xp=xt.clone(); xm=xt.clone()
        xp[:,j]=torch.clamp(xp[:,j]+h,0,1)
        xm[:,j]=torch.clamp(xm[:,j]-h,0,1)
        _,z1p,z2p=model.parts(xp)
        _,z1m,z2m=model.parts(xm)
        flip1=((z1p>0)!=(z1m>0)).float()
        flip2=((z2p>0)!=(z2m>0)).float()
        g=((flip1*w1).sum(-1)+(flip2*w2[:,None,:]).sum(-1))/denominator
        gate += g.mean(0)/2
    return means,sigma,gate.numpy()


def gp_model(x,y,seed):
    ker=(ConstantKernel(1.0,(1e-2,1e2))*RBF([0.3,0.3],(0.02,5.0))
         +WhiteKernel(1e-6,(1e-10,1e-2)))
    model=GaussianProcessRegressor(kernel=ker,normalize_y=True,
                                   n_restarts_optimizer=0,random_state=seed%(2**32),alpha=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x,y)
    return model


def normalize_relative(values,mask):
    """Fixed interpretation: 90th-percentile gate normalization, clip to [0,2]."""
    scale=max(float(np.percentile(values[mask],90)),1e-12)
    return np.clip(np.asarray(values)/scale,0,2)


def posterior_models(xgrid,xtrain,ytrain,seed,kind,ensemble,epochs,gates):
    mus=[];sigs=[];gs=[]
    for j in range(2):
        fit_seed=(int(seed)*1009 + 100003*j) % (2**32)
        if kind=="gp":
            m=gp_model(xtrain,ytrain[:,j],fit_seed)
            mu,sig=m.predict(xgrid,return_std=True)
            gate=None
        else:
            m,ym,ys=fit_nn(xtrain,ytrain[:,j],ensemble,epochs,fit_seed)
            mu,sig,gate=predict_nn(m,ym,ys,xgrid,gates)
        mus.append(mu);sigs.append(np.maximum(sig,1e-10))
        if gates: gs.append(gate)
    return np.stack(mus,axis=1),np.stack(sigs,axis=1),(np.stack(gs,axis=1) if gates else None)


def acquisition_scores(sig,gate,mask,alpha,sigma_scales,lam):
    global_score=((1-alpha)*sig[:,0]/sigma_scales[0]
                  + alpha*sig[:,1]/sigma_scales[1])
    if lam == 0:
        return global_score
    raw_gate=(1-alpha)*gate[:,0]+alpha*gate[:,1]
    activation=normalize_relative(raw_gate,mask)
    return global_score * (1+lam*activation)


def metric_report(y,pred,sig,selected,targets,validation_count,epsilon):
    n=len(y)
    mask=np.ones(n,dtype=bool);mask[selected]=False
    mae=np.mean(np.abs(pred[mask]-y[mask]),axis=0)
    rmse=np.sqrt(np.mean((pred[mask]-y[mask])**2,axis=0))
    unseen=np.where(mask)[0]
    bests=[];feasible_count=0;exact_count=0
    # Explicitly stated validation-score implementation: probability of
    # inductance feasibility times decreasing predicted-resistance utility.
    # Not guaranteed identical to the latest paper's unstated score.
    res_scale=max(float(np.std(y[selected,1])),1e-6)
    res_anchor=float(np.min(y[selected,1]))
    effR=np.clip((pred[unseen,1]-res_anchor)/res_scale,-10,10)
    resistance_utility=np.exp(-np.maximum(effR,0))
    for t in targets:
        actual_feasible=np.abs(y[:,0]-t)<=epsilon*t
        if not actual_feasible.any():
            raise ValueError(f"Oracle contains no feasible point for target {t:g}; consider a different target range")
        optimum=np.min(y[actual_feasible,1])
        s=np.maximum(sig[unseen,0],1e-9)
        p_feas=np.maximum(ndtr((t*(1+epsilon)-pred[unseen,0])/s)
                          -ndtr((t*(1-epsilon)-pred[unseen,0])/s),0)
        score=p_feas * resistance_utility
        # Stable tie-break: best mean target proximity.
        ordered=np.lexsort((np.abs(pred[unseen,0]-t),-score))
        validated=unseen[ordered[:validation_count]]
        eligible=validated[actual_feasible[validated]]
        if len(eligible)==0:
            bests.append(1.0)
        else:
            feasible_count+=1
            got=float(np.min(y[eligible,1]))
            # r/(1+r) = 1 - R*/R for feasible solutions.
            bests.append(max(0,1-optimum/got))
            exact_count+=int(np.isclose(got,optimum,rtol=1e-9,atol=1e-10))
    return dict(mae_L_mH=mae[0],rmse_L_mH=rmse[0],mae_R_ohm=mae[1],rmse_R_ohm=rmse[1],
                mean_loss=float(np.mean(bests)),feasible_targets=feasible_count,
                total_targets=len(bests),exact_optima=exact_count)


def run_one(xgrid,y,seed,split,method,lam,params):
    t0=time.monotonic()
    norm=(xgrid-xgrid.min(0))/np.maximum(xgrid.max(0)-xgrid.min(0),1e-12)
    rg=np.random.default_rng(seed)
    init=np.asarray(rg.choice(len(y),size=6,replace=False),dtype=int)
    chosen=list(init)
    train_count=params.budget-params.validations
    if not 6 < train_count < len(y)-params.validations:
        raise ValueError("Budget requires >6 model-building points and enough unseen validations")
    scales=None
    model_kind="gp" if method in ("gp", "gp_random") else "ensemble"
    for k in range(train_count-6):
        selected=np.array(chosen,dtype=int)
        mask=np.ones(len(y),dtype=bool);mask[selected]=False
        if method in ("random", "gp_random"):
            new=int(rg.choice(np.where(mask)[0]))
        else:
            mu,sig,gate=posterior_models(norm,norm[selected],y[selected],
                     seed=seed*100 + k,kind=model_kind,ensemble=params.ensemble,
                     epochs=params.epochs,gates=(method=="relu" and lam>0))
            if scales is None:
                scales=np.maximum(np.percentile(sig[mask],90,axis=0),1e-9)
            scores=acquisition_scores(sig,gate,mask,params.alpha,scales,lam)
            scores[~mask]=-np.inf
            new=int(np.argmax(scores))
        chosen.append(new)
    selected=np.array(chosen,dtype=int)
    mu,sig,_=posterior_models(norm,norm[selected],y[selected],
             seed=seed*100+train_count,kind=model_kind,ensemble=params.ensemble,
             epochs=params.epochs,gates=False)
    metrics=metric_report(y,mu,sig,selected,params.targets,params.validations,params.epsilon)
    return {"split":split,"seed":seed,"method":method,"lambda":lam,"budget":params.budget,
            "train_evals":train_count,**metrics,"runtime_sec":time.monotonic()-t0}


def write_row(path,row):
    must_header=not path.exists() or path.stat().st_size==0
    with open(path,"a",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        if must_header: writer.writeheader()
        writer.writerow(row)
        file.flush()


def load_rows(path):
    if not path.exists():return []
    with open(path,newline="",encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def matching_key(row):
    return str(row["split"]),int(row["seed"]),str(row["method"]),round(float(row["lambda"]),9)


def means(rows,method,lam):
    matched=[r for r in rows if r["method"]==method and abs(float(r["lambda"])-lam)<1e-9]
    if not matched:return None
    agg={"method":method,"lambda":lam,"n":len(matched)}
    for v in ["mean_loss","mae_L_mH","mae_R_ohm"]:
        arr=np.array([float(r[v]) for r in matched])
        agg[v+"_mean"]=float(np.mean(arr))
        agg[v+"_sd"]=float(np.std(arr,ddof=1)) if len(arr)>1 else float("nan")
    for out,col in [("feasibility_rate","feasible_targets"),("exact_optimum_rate","exact_optima")]:
        agg[out]=float(np.mean([int(r[col])/int(r["total_targets"]) for r in matched]))
    return agg


def paired_ci(rows,lam,rng_seed=331,boots=4000):
    baseline={int(r["seed"]):float(r["mean_loss"]) for r in rows if
              r["method"]=="relu" and abs(float(r["lambda"]))<1e-9}
    trial={int(r["seed"]):float(r["mean_loss"]) for r in rows if
           r["method"]=="relu" and abs(float(r["lambda"])-lam)<1e-9}
    overlap=sorted(set(baseline)&set(trial))
    if not overlap:return None
    delta=np.asarray([trial[s]-baseline[s] for s in overlap],float)
    rng=np.random.default_rng(rng_seed)
    boot_means=np.mean(delta[rng.integers(0,len(delta),(boots,len(delta)))],axis=1)
    return {"lambda":lam,"paired_n":len(delta),"difference_loss_mean":float(np.mean(delta)),
            "ci95_low":float(np.quantile(boot_means,0.025)),
            "ci95_high":float(np.quantile(boot_means,0.975)),
            "improved_seed_fraction":float(np.mean(delta<0)),
            "tied_seed_fraction":float(np.mean(delta==0))}


def summaries(rows,lambdas):
    data={"methods":[],"paired_vs_lambda_0":[]}
    for lam in lambdas:
        x=means(rows,"relu",lam)
        if x:data["methods"].append(x)
        if lam!=0:
            p=paired_ci(rows,lam)
            if p:data["paired_vs_lambda_0"].append(p)
    for name in ["gp", "random", "gp_random"]:
        x=means(rows,name,0)
        if x:data["methods"].append(x)
    return data


def save_summary(outpath,split,rows,lambdas):
    s=summaries(rows,lambdas)
    with open(outpath/f"{split}_summary.json","w") as file:json.dump(s,file,indent=2)
    header=list(s["methods"][0]) if s["methods"] else []
    if header:
        with open(outpath/f"{split}_summary.csv","w",newline="") as fp:
            writer=csv.DictWriter(fp,fieldnames=header)
            writer.writeheader();writer.writerows(s["methods"])
    print("\n",split.upper(),"SUMMARY",flush=True)
    for i in s["methods"]:
        print(f'  {i["method"]:7} lambda={i["lambda"]:<4} N={i["n"]:<3} '
              f'loss={i["mean_loss_mean"]:.5f}  L_MAE={i["mae_L_mH_mean"]:.3f} '
              f'feas={i["feasibility_rate"]:.2%} exact={i["exact_optimum_rate"]:.2%}',flush=True)
    for p in s["paired_vs_lambda_0"]:
        print(f'  paired Δ lambda={p["lambda"]:g}: {p["difference_loss_mean"]:+.5f} '
              f'95%CI [{p["ci95_low"]:+.5f}, {p["ci95_high"]:+.5f}], '
              f'wins={p["improved_seed_fraction"]:.1%}',flush=True)
    return s


def parse():
    a=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                              description="Matched dual-output GP/ensemble ReLU acquisition study")
    a.add_argument("--data",help="CSV with T,n,L_mH,R_ohm. Omit for demonstration oracle only")
    a.add_argument("--output-dir",default="coil_relu_100seed_results")
    a.add_argument("--split",choices=["dev","test","both"],default="both")
    a.add_argument("--seeds",type=int,default=100,help="Seeds PER split PER acquisition")
    a.add_argument("--dev-start",type=int,default=20261015)
    a.add_argument("--test-start",type=int,default=20361015)
    a.add_argument("--lambdas",nargs="+",type=float,default=[0,0.25,0.5,1,2,4])
    a.add_argument("--budget",type=int,default=27)
    a.add_argument("--validations",type=int,default=12)
    a.add_argument("--epochs",type=int,default=180)
    a.add_argument("--ensemble",type=int,default=10)
    a.add_argument("--alpha",type=float,default=.5,help="Weight on normalized R disagreement")
    a.add_argument("--epsilon",type=float,default=.025)
    a.add_argument("--targets",nargs="+",type=float,default=list(range(1,88)))
    a.add_argument("--include-gp",action=argparse.BooleanOptionalAction,default=True)
    a.add_argument("--include-random",action=argparse.BooleanOptionalAction,default=False,help="Also compare GP-random and ensemble-random, in addition to GP-active and ensemble-active")
    a.add_argument("--test-all-lambdas",action="store_true",
                   help="Run all lambda values on confirmation seeds. Otherwise test only dev-selected lambda and 0")
    a.add_argument("--threads",type=int,default=1)
    a.add_argument("--reset",action="store_true",help="Overwrite existing results; default resumes")
    return a.parse_args()


def main():
    args=parse()
    torch.set_num_threads(args.threads)
    if args.seeds<1:raise ValueError("seeds must be positive")
    if any(x<0 for x in args.lambdas):raise ValueError("lambda must be nonnegative")
    if not 0 <=args.alpha<=1:raise ValueError("alpha must be in [0,1]")
    if args.split=="both" and set(range(args.dev_start,args.dev_start+args.seeds)) & set(range(args.test_start,args.test_start+args.seeds)):
        raise ValueError("Development and test seeds overlap")
    lambdas=sorted(set([0.0]+args.lambdas))
    x,y=read_oracle_csv(args.data) if args.data else reconstructed_oracle()
    for t in args.targets:
        if not np.any(abs(y[:,0]-t)<=args.epsilon*t):
            raise ValueError(f"No feasible design for requested target {t}. Adjust --targets")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ("reset","split","test_all_lambdas")}
    config["source_oracle"]="csv" if args.data else "DEMO ONLY: reconstructed earlier formulation"
    cfg_file=out/"settings.json"
    if cfg_file.exists() and not args.reset:
        old=json.loads(cfg_file.read_text())
        if old!=config:
            raise ValueError("Existing output directory has different settings. Change --output-dir or use --reset")
    cfg_file.write_text(json.dumps(config,indent=2))
    if args.reset:
        for xfile in out.glob("*_rows.csv"):xfile.unlink()
    print("WARNING: demonstration oracle, not latest manuscript's actual simulator" if not args.data
          else f"Using supplied oracle: {args.data}",flush=True)
    print(f"Grid: {len(y)} configurations, seeds per split={args.seeds}, "
          f"lambda values={lambdas}; B={args.budget}",flush=True)

    def run_split(which,lam_run,first_seed):
        path=out/f"{which}_rows.csv"
        existing={matching_key(r) for r in load_rows(path)}
        methods=[("relu",v) for v in lam_run]
        if args.include_gp:methods.append(("gp",0.0))
        if args.include_random:
            methods.extend([("random",0.0), ("gp_random",0.0)])
        total=args.seeds*len(methods);n=0;time0=time.monotonic()
        for seed in range(first_seed,first_seed+args.seeds):
            for method,lam in methods:
                key=(which,seed,method,round(lam,9))
                if key not in existing:
                    row=run_one(x,y,seed,which,method,lam,args)
                    write_row(path,row)
                    existing.add(key)
                n+=1
                if n%max(1,len(methods))==0:
                    print(f"{which}: seeds {n//len(methods)}/{args.seeds} "
                          f"({n}/{total} cases), elapsed {time.monotonic()-time0:.1f}s",flush=True)
        rows=load_rows(path)
        return save_summary(out,which,rows,lam_run)

    if args.split in ("dev","both"):
        dev=run_split("dev",lambdas,args.dev_start)
    else:
        f=out/"dev_rows.csv"
        if not f.exists():raise ValueError("Run --split dev first, then --split test")
        dev=summaries(load_rows(f),lambdas)
    dev_models=[m for m in dev["methods"] if m["method"]=="relu" and m["n"]==args.seeds]
    if not dev_models:
        raise RuntimeError("No complete dev lambda models to select")
    best=min(dev_models,key=lambda d:(d["mean_loss_mean"],d["lambda"]))["lambda"]
    selected={"chosen_lambda_on_dev":best,"criterion":"minimum dev mean bounded loss",
              "confirmatory_test_lambdas":[0,best]}
    (out/"chosen_lambda.json").write_text(json.dumps(selected,indent=2))
    print("PRESELECTED lambda (development only):",best,flush=True)
    if args.split in ("test","both"):
        run_split("test",lambdas if args.test_all_lambdas else sorted(set([0.,best])),args.test_start)
    print("DONE. Results in:",out.resolve(),flush=True)

if __name__=="__main__":
    main()

