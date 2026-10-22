"""Import-safe feasible primal log-barrier Newton pilot (not a production IPM)."""
import importlib.util
import json
import hashlib
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'experiments/sqp/benchmark_sqp.py'
spec=importlib.util.spec_from_file_location('ip_problem_source',PARENT)
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
problem=source.problem


def barrier(evaluate,x,curvature,mu):
    f,grad,H,g,J=evaluate(x)
    if np.any(g>=0):raise ValueError('Barrier requires strictly feasible x')
    lam=mu/(-g)
    value=f-mu*np.log(-g).sum()
    gradient=grad+J.T@lam
    hessian=H+2*np.dot(lam,curvature)*np.eye(len(x))+J.T@((mu/g**2)[:,None]*J)
    return float(value),gradient,hessian,lam


def certificate(evaluate,x,mu):
    f,grad,_,g,J=evaluate(x)
    if np.any(g>=0):raise ValueError('Non-interior iterate')
    lam=mu/(-g)
    stat=float(np.linalg.norm(grad+J.T@lam,np.inf));comp=float(np.max(abs(lam*g)))
    return dict(objective=float(f),feasibility=float(max(0.,g.max())),stationarity=stat,
                complementarity=comp,success=bool(stat<=1e-4 and comp<=1e-5)),lam


def solve(evaluate,x0,curvature,method,factor,seed,maxouter=30,budget=10000):
    start=perf_counter();nfev=0
    def ev(x):
        nonlocal nfev
        nfev+=1
        return evaluate(x)
    x=x0.copy();contractions=0
    # Common feasible initialization uses known strictly feasible origin.
    while np.max(ev(x)[3])>=-1e-4:
        x*=.5;contractions+=1
        if contractions>60:raise ValueError('Cannot find strict interior on ray to origin')
    mu=.1;rng=np.random.default_rng(seed);trace=[];steps=0;actions=0;feature_time=0.;reason='outer_budget';line_failures=0
    for outer in range(1,maxouter+1):
        converged=False;condition=0.;inner_count=0
        for inner in range(100):
            if nfev>=budget:reason='evaluation_budget';break
            val,grad,H,lam=barrier(ev,x,curvature,mu)
            if np.linalg.norm(grad,np.inf)<=max(1e-7,.1*mu):
                converged=True;break
            eig,V=np.linalg.eigh(H)
            B=(V*np.maximum(eig,1e-5))@V.T
            condition=float(np.linalg.cond(B))
            p=np.linalg.solve(B,-grad);slope=grad@p;accepted=False
            for bt in range(50):
                if nfev>=budget:break
                alpha=.5**bt;trial=x+alpha*p
                ft,_,_,gt,_=ev(trial)
                if np.all(gt<0):
                    trialval=ft-mu*np.log(-gt).sum()
                    if trialval<=val+1e-4*alpha*slope:
                        x=trial;accepted=True;break
            if not accepted:line_failures+=1;reason='line_search';break
            steps+=1;inner_count+=1
        cert,lam=certificate(ev,x,mu)
        gate=False;I=0.;feature_start=perf_counter()
        if method in ('signature','random'):
            _,_,_,g,J=ev(x)
            reaction=lam*np.linalg.norm(J,axis=1)
            L,theta,I=source.features_from_reaction(reaction[reaction>0],np.linalg.norm(J.T@lam))
            gate=bool(L>0 and I>1.3)
            if method=='random':gate=bool(rng.random()<.5)
        elif method=='conditioning':gate=bool(condition>1e4)
        feature_time+=perf_counter()-feature_start
        reduction=factor if gate else .2
        trace.append(dict(outer=outer,mu=mu,inner_converged=converged,inner_steps=inner_count,
                          condition=condition,I=float(I),gate=gate,next_mu_factor=reduction,**cert))
        if cert['success']:reason='kkt';break
        if not converged:
            if reason=='outer_budget':reason='inner_budget'
            break
        actions+=int(gate)
        if outer<maxouter:mu*=reduction
    cert,lam=certificate(ev,x,mu)
    return dict(**cert,time=perf_counter()-start,nfev=nfev,newton_steps=steps,outer_iterations=len(trace),
                actions=actions,feature_time=feature_time,reason=reason,line_failures=line_failures,
                initial_contractions=contractions,x=x.tolist(),dual=lam.tolist(),trace=trace)


def main():
    out=ROOT/'results/interior_point_pilot';out.mkdir(parents=True,exist_ok=False)
    cfg=dict(families=['generic','correlated','opposed'],methods=['baseline','conditioning','signature','random'],
             calibration_seeds=[71,82,93],test_seeds=list(range(301,313)),factors=[.4,.6],
             initial_mu=.1,baseline_reduction=.2,maxouter=30,inner_maxiter=100,evaluation_budget=10000,
             caveat='Feasible primal log-barrier, known feasible origin, not predictor-corrector primal-dual IPM. Random is unmatched Bernoulli placebo.')
    (out/'metadata.json').write_text(json.dumps(dict(config=cfg,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),problem_source_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest()),indent=2))
    rows=[];selected={};rng=np.random.default_rng(20261024);start=perf_counter()
    with (out/'runs.jsonl').open('w',buffering=1) as handle:
        for fi,family in enumerate(cfg['families']):
            for split,seeds in [('calibration',cfg['calibration_seeds']),('test',cfg['test_seeds'])]:
                tasks=[]
                for seed in seeds:
                    for method in cfg['methods']:
                        factors=([.2] if method=='baseline' else cfg['factors']) if split=='calibration' else [selected[family+':'+method]]
                        tasks.extend((seed,method,v) for v in factors)
                rng.shuffle(tasks)
                for seed,method,factor in tasks:
                    ev,x,c=problem(seed+fi*100000,family)
                    row=dict(split=split,family=family,seed=seed,method=method,factor=factor,
                             **solve(ev,x,c,method,factor,seed+999))
                    handle.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if split=='calibration':
                    for method in cfg['methods']:
                        def score(v):
                            rr=[r for r in rows if r['family']==family and r['split']==split and r['method']==method and r['factor']==v]
                            return (sum(not r['success'] for r in rr),sum(r['nfev'] if r['success'] else 10000 for r in rr),v)
                        selected[family+':'+method]=min([.2] if method=='baseline' else cfg['factors'],key=score)
                print(f'{family} {split}: {perf_counter()-start:.1f}s',flush=True)
    summary={}
    for method in cfg['methods']:
        rr=[r for r in rows if r['split']=='test' and r['method']==method]
        summary[method]=dict(n=len(rr),success=sum(r['success'] for r in rr),median_nfev=float(np.median([r['nfev'] for r in rr])),median_newton_steps=float(np.median([r['newton_steps'] for r in rr])),median_time=float(np.median([r['time'] for r in rr])),actions=sum(r['actions'] for r in rr))
    final=dict(methods=summary,selected_factors=selected,total_runs=len(rows),wall_time=perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))

if __name__=='__main__':main()
