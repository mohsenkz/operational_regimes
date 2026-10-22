"""Projected BCD block-selection pilot. Import safe; g_b(x_b)<=0.
Convex block-separable constraints, nonconvex coupled objective.
"""
import json
import hashlib
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import minimize, nnls

ROOT=Path(__file__).resolve().parents[2]
CONDITIONS=[dict(name='small_mild',blocks=4,per_block=4,amplitude=.1,frequency=2.,coupling=.2),
            dict(name='medium_nonlinear',blocks=8,per_block=16,amplitude=.4,frequency=4.,coupling=.5),
            dict(name='many_constraints',blocks=12,per_block=64,amplitude=.4,frequency=4.,coupling=.5)]


def problem(seed,cfg):
    rng=np.random.default_rng(seed);B=cfg['blocks'];n=2*B;m=cfg['per_block']
    R=rng.normal(size=(n,n))/np.sqrt(n)
    Q=np.diag(rng.uniform(.5,2.,n))+cfg['coupling']*(R.T@R)
    target=rng.normal(size=n)*1.5
    A=rng.normal(size=(B,m,2));A/=np.linalg.norm(A,axis=2)[:,:,None]
    # Slightly curved inequalities plus unit disk, all convex and block local.
    c=rng.uniform(.05,.3,size=(B,m));offset=rng.uniform(.25,.9,size=(B,m))
    amp=cfg['amplitude'];freq=cfg['frequency']
    lips=np.array([np.linalg.eigvalsh(Q[2*b:2*b+2,2*b:2*b+2]).max()+amp*freq**2+.24 for b in range(B)])
    def objective(x):
        d=x-target
        return float(.5*d@Q@d+amp*np.sin(freq*x).sum()+.02*np.sum(x**4)),Q@d+amp*freq*np.cos(freq*x)+.08*x**3
    def constraints(b,z):
        g=np.r_[A[b]@z+c[b]*(z@z)-offset[b],z@z-1.]
        J=np.vstack((A[b]+2*c[b,:,None]*z,2*z))
        return g,J
    return objective,constraints,lips,np.zeros(n)


class Projector:
    def __init__(self,constraints):self.constraints=constraints;self.calls=0;self.nit=0;self.failures=0
    def __call__(self,b,y,start):
        self.calls+=1
        g,_=self.constraints(b,y)
        if g.max()<=0:return y.copy()
        def con(z):return -self.constraints(b,z)[0]
        def jac(z):return -self.constraints(b,z)[1]
        res=minimize(lambda z:(.5*np.sum((z-y)**2),z-y),start,jac=True,method='SLSQP',
                     constraints=[dict(type='ineq',fun=con,jac=jac)],
                     options=dict(ftol=1e-12,maxiter=80))
        self.nit+=int(res.nit)
        g,J=self.constraints(b,res.x)
        active=g>=-1e-6
        dual=nnls(J[active].T,y-res.x,maxiter=1000)[0] if np.any(active) else np.zeros(0)
        residual=np.linalg.norm(res.x-y+(J[active].T@dual if np.any(active) else 0),np.inf)
        # SLSQP status alone is not a projection certificate: check primal/KKT residuals.
        if g.max()>1e-7 or residual>1e-6:
            self.failures+=1
            raise RuntimeError('Projection failed verification: '+str(res.message))
        return res.x


def diagnostic(objective,constraints,lips,x,proj):
    f,grad=objective(x);norms=[];feas=0.
    for b,L in enumerate(lips):
        sl=slice(2*b,2*b+2);z=x[sl];p=proj(b,z-grad[sl]/L,z)
        norms.append(float(np.linalg.norm(L*(z-p),np.inf)))
        feas=max(feas,float(max(0.,constraints(b,z)[0].max())))
    return dict(objective=f,stationarity=max(norms),feasibility=feas,
                success=bool(max(norms)<=1e-4 and feas<=1e-6))


def solve(objective,constraints,lips,x0,method,alpha,seed,maxupdates=600):
    x=x0.copy();B=len(lips);proj=Projector(constraints);rng=np.random.default_rng(seed)
    start=perf_counter();trace=[];feature_time=0.;nnls_calls=0;nonzero_features=0;updates=0;reason='update_budget'
    cert=diagnostic(objective,constraints,lips,x,proj)
    try:
        for k in range(maxupdates):
            if cert['success']:reason='stationarity';break
            f,grad=objective(x)
            if method in ('cyclic','random'):
                b=k%B if method=='cyclic' else int(rng.integers(B));sl=slice(2*b,2*b+2)
                chosen=proj(b,x[sl]-grad[sl]/lips[b],x[sl])
            else:
                candidates=[];gains=[];weights=[]
                for b,L in enumerate(lips):
                    sl=slice(2*b,2*b+2);z=x[sl];p=proj(b,z-grad[sl]/L,z)
                    d=p-z;candidates.append(p);gains.append(max(0.,float(-grad[sl]@d-.5*L*(d@d))))
                    if method in ('signature','shuffled'):
                        t=perf_counter();g,J=constraints(b,z);active=g>=-1e-4
                        w=0.
                        if np.any(active):
                            lam=nnls(J[active].T,-grad[sl],maxiter=500)[0];nnls_calls+=1
                            reaction=lam*np.linalg.norm(J[active],axis=1)
                            if reaction.sum()>0:
                                I=reaction.sum()/max(np.linalg.norm(J[active].T@lam),1e-12)
                                participation=reaction.sum()/max(np.linalg.norm(grad[sl]),1e-12)
                                w=float(np.log1p(min(I,10.))*min(participation,2.));nonzero_features+=1
                        weights.append(w);feature_time+=perf_counter()-t
                gains=np.array(gains)
                if method in ('signature','shuffled'):
                    weights=np.array(weights)
                    if method=='shuffled':weights=rng.permutation(weights)
                    gains*=1+alpha*weights
                b=int(np.argmax(gains));chosen=candidates[b]
            sl=slice(2*b,2*b+2);x[sl]=chosen;updates=k+1
            # Common full certificate every B updates; its work is included.
            if updates%B==0 or updates==maxupdates:
                cert=diagnostic(objective,constraints,lips,x,proj)
                trace.append(dict(updates=updates,projection_calls=proj.calls,**cert))
    except RuntimeError as exc:
        reason=str(exc)
        cert=dict(objective=objective(x)[0],stationarity=1e100,feasibility=max(max(0.,constraints(b,x[2*b:2*b+2])[0].max()) for b in range(B)),success=False)
    return dict(**cert,time=perf_counter()-start,updates=updates,projection_calls=proj.calls,
                projection_iterations=proj.nit,projection_failures=proj.failures,feature_time=feature_time,
                nnls_calls=nnls_calls,nonzero_features=nonzero_features,reason=reason,x=x.tolist(),trace=trace)


def main():
    out=ROOT/'results/bcd_pilot';out.mkdir(parents=True,exist_ok=False)
    cfg=dict(conditions=CONDITIONS,calibration_seeds=[1111,1122],test_seeds=list(range(1201,1207)),
             methods=['cyclic','random','gs','signature','shuffled'],alphas=[.25,1.],maxupdates=600,
             certificate_interval='number of blocks',stationarity_tol=1e-4,
             caveat='Block-separable convex constraints, coupled nonconvex objective; projected BCD with safe block Lipschitz steps. Shuffled permutes weights among blocks at each selection.')
    (out/'metadata.json').write_text(json.dumps(dict(config=cfg,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    rows=[];selected={};rng=np.random.default_rng(5612);start=perf_counter()
    with (out/'runs.jsonl').open('w',buffering=1) as handle:
        for ci,cond in enumerate(CONDITIONS):
            name=cond['name']
            for split,seeds in [('calibration',cfg['calibration_seeds']),('test',cfg['test_seeds'])]:
                tasks=[]
                for seed in seeds:
                    for method in cfg['methods']:
                        vals=([0.] if method in ('cyclic','random','gs') else cfg['alphas']) if split=='calibration' else [selected[name+':'+method]]
                        tasks.extend((seed,method,v) for v in vals)
                rng.shuffle(tasks)
                for seed,method,alpha in tasks:
                    obj,con,L,x=problem(seed+ci*100000,cond)
                    row=dict(condition=name,split=split,seed=seed,method=method,alpha=alpha,**solve(obj,con,L,x,method,alpha,seed+99))
                    handle.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                if split=='calibration':
                    for method in cfg['methods']:
                        def score(a):
                            rr=[r for r in rows if r['condition']==name and r['split']==split and r['method']==method and r['alpha']==a]
                            return (sum(not r['success'] for r in rr),sum(r['projection_calls'] for r in rr),a)
                        selected[name+':'+method]=min([0.] if method in ('cyclic','random','gs') else cfg['alphas'],key=score)
                print(f'{name} {split}: {perf_counter()-start:.1f}s',flush=True)
    summary={}
    for method in cfg['methods']:
        rr=[r for r in rows if r['split']=='test' and r['method']==method]
        summary[method]=dict(n=len(rr),success=sum(r['success'] for r in rr),median_updates=float(np.median([r['updates'] for r in rr])),median_projection_calls=float(np.median([r['projection_calls'] for r in rr])),median_time=float(np.median([r['time'] for r in rr])),projection_failures=sum(r['projection_failures'] for r in rr))
    final=dict(methods=summary,selected_alphas=selected,total_runs=len(rows),wall_time=perf_counter()-start)
    (out/'summary.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))

if __name__=='__main__':main()
