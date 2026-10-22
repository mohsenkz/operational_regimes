"""Import-safe Powell-Hestenes-Rockafellar inequality ALM pilot.
Run from repository root. Signature-based penalty damping is a hypothesis.
"""
import importlib.util
import json
import hashlib
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'experiments/sqp/benchmark_sqp.py'
spec=importlib.util.spec_from_file_location('sqp_problem_source',PARENT)
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
problem=source.problem
features_from_reaction=source.features_from_reaction


def augmented(evaluate,x,lam,rho):
    f,grad,_,g,J=evaluate(x)
    shifted=np.maximum(0.,lam+rho*g)
    value=f+(shifted@shifted-lam@lam)/(2*rho)
    derivative=grad+J.T@shifted
    return float(value),derivative


def certificate(evaluate,x,lam):
    f,grad,_,g,J=evaluate(x)
    feas=float(max(0.,g.max()))
    stat=float(np.linalg.norm(grad+J.T@lam,np.inf))
    comp=float(np.max(np.abs(lam*g)))
    return dict(objective=float(f),feasibility=feas,stationarity=stat,complementarity=comp,
                success=bool(feas<=1e-6 and stat<=1e-4 and comp<=1e-5))


def solve(evaluate,x0,method,factor,seed,maxouter=60,budget=20000):
    x=x0.copy();lam=np.zeros(len(evaluate(x)[3]));rho=1.;previous=np.inf
    rng=np.random.default_rng(seed);start=perf_counter();trace=[];nfev=0;nit=0
    actions=0;inner_failures=0;feature_time=0.;reason='outer_budget'
    cert=certificate(evaluate,x,lam)
    for outer in range(1,maxouter+1):
        remaining=budget-nfev
        if remaining<=0:reason='evaluation_budget';break
        result=minimize(lambda y:augmented(evaluate,y,lam,rho),x,jac=True,method='L-BFGS-B',
                        options=dict(gtol=1e-7,ftol=1e-15,maxiter=400,maxfun=remaining,maxls=40))
        x=result.x;nfev+=int(result.nfev);nit+=int(result.nit)
        inner_failures+=int(not result.success)
        f,grad,_,g,J=evaluate(x)
        # ALM updates the actual nonnegative dual, never replacing it with NNLS.
        violation=float(np.linalg.norm(np.maximum(g,-lam/rho),np.inf))
        lam=np.maximum(0.,lam+rho*g)
        cert=certificate(evaluate,x,lam)
        stalled=violation>.5*previous
        gate=False;I=0.;L=0.
        if method in ('signature','random'):
            t=perf_counter();reaction=lam*np.linalg.norm(J,axis=1)
            L,theta,I=features_from_reaction(reaction[reaction>0],np.linalg.norm(J.T@lam))
            gate=bool(L>0 and I>1.3)
            if method=='random':gate=bool(rng.random()<.5)
            feature_time+=perf_counter()-t
        elif method=='history':
            gate=bool(np.isfinite(previous) and violation>.9*previous)
        next_rho=rho
        if stalled:
            multiplier=factor if gate else 5.
            next_rho=min(1e8,rho*multiplier)
            actions+=int(gate)
        trace.append(dict(outer=outer,rho=rho,next_rho=next_rho,violation=violation,gate=gate,
                          I=float(I),L=float(L),inner_success=bool(result.success),inner_message=str(result.message),
                          inner_nfev=int(result.nfev),**cert))
        if cert['success']:reason='kkt';break
        if not np.all(np.isfinite(x)) or not np.isfinite(f):reason='nonfinite';break
        previous=violation;rho=next_rho
    return dict(**cert,outer_iterations=len(trace),inner_iterations=nit,nfev=nfev,
                time=perf_counter()-start,feature_time=feature_time,actions=actions,
                inner_failures=inner_failures,reason=reason,x=x.tolist(),dual=lam.tolist(),trace=trace)


def main():
    out=ROOT/'results/alm_pilot';out.mkdir(parents=True,exist_ok=False)
    cfg=dict(families=['generic','correlated','opposed'],methods=['baseline','history','signature','random'],
             calibration_seeds=[41,52,63],test_seeds=list(range(201,213)),factors=[2.,3.],
             initial_rho=1.,baseline_multiplier=5.,progress_threshold=.5,maxouter=60,
             inner_method='L-BFGS-B',inner_maxiter=400,inner_gtol=1e-7,inner_ftol=1e-15,
             evaluation_budget=20000,success=dict(feasibility=1e-6,stationarity=1e-4,complementarity=1e-5))
    meta=dict(config=cfg,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              problem_source_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
              random_control='Bernoulli(0.5), not a matched permutation; no regime labels used')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    rows=[];selected={};rng=np.random.default_rng(20261023);start=perf_counter()
    with (out/'runs.jsonl').open('w',buffering=1) as handle:
        for fi,family in enumerate(cfg['families']):
            for split,seeds in [('calibration',cfg['calibration_seeds']),('test',cfg['test_seeds'])]:
                tasks=[]
                for seed in seeds:
                    for method in cfg['methods']:
                        factors=([5.] if method=='baseline' else cfg['factors']) if split=='calibration' else [selected[family+':'+method]]
                        tasks.extend((seed,method,v) for v in factors)
                rng.shuffle(tasks)
                for seed,method,factor in tasks:
                    evaluate,x,c=problem(seed+fi*100000,family)
                    r=solve(evaluate,x,method,factor,seed+999)
                    row=dict(split=split,family=family,seed=seed,method=method,factor=factor,**r)
                    handle.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if split=='calibration':
                    for method in cfg['methods']:
                        def score(v):
                            rr=[r for r in rows if r['family']==family and r['split']==split and r['method']==method and r['factor']==v]
                            return (sum(not r['success'] for r in rr),sum(r['nfev'] if r['success'] else cfg['evaluation_budget'] for r in rr),v)
                        selected[family+':'+method]=min([5.] if method=='baseline' else cfg['factors'],key=score)
                print(f'{family} {split} completed in {perf_counter()-start:.1f}s',flush=True)
    summary={}
    for method in cfg['methods']:
        rr=[r for r in rows if r['split']=='test' and r['method']==method]
        summary[method]=dict(n=len(rr),success=sum(r['success'] for r in rr),median_nfev=float(np.median([r['nfev'] for r in rr])),median_outer=float(np.median([r['outer_iterations'] for r in rr])),median_time=float(np.median([r['time'] for r in rr])),actions=sum(r['actions'] for r in rr),inner_failures=sum(r['inner_failures'] for r in rr))
    final=dict(methods=summary,selected_factors=selected,wall_time=perf_counter()-start,total_runs=len(rows))
    (out/'summary.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))

if __name__=='__main__':main()
