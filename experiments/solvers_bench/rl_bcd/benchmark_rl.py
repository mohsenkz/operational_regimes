"""Ordinary on-policy REINFORCE + linear state-value baseline for BCD selection.
No meta-learning, no test-time training, no external ML framework required.
"""
from pathlib import Path
import importlib.util
import hashlib
import json
import platform
from time import perf_counter
import numpy as np
import scipy
from scipy.optimize import nnls, minimize

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'experiments/bcd/benchmark_bcd.py'
spec=importlib.util.spec_from_file_location('bcd',P);bcd=importlib.util.module_from_spec(spec);spec.loader.exec_module(bcd)
D=10;H=16

class Projector(bcd.Projector):
    """Retry numerical projection failures without relaxing the certificate."""
    def __call__(self,b,y,start):
        try:return super().__call__(b,y,start)
        except RuntimeError:
            res=minimize(lambda z:(.5*np.sum((z-y)**2),z-y),np.zeros_like(start),jac=True,method='SLSQP',
                         constraints=[dict(type='ineq',fun=lambda z:-self.constraints(b,z)[0],
                                           jac=lambda z:-self.constraints(b,z)[1])],
                         options=dict(ftol=1e-14,maxiter=500))
            self.nit+=int(res.nit)
            g,J=self.constraints(b,res.x);active=g>=-1e-6
            dual=nnls(J[active].T,y-res.x,maxiter=1000)[0] if np.any(active) else np.zeros(0)
            residual=np.linalg.norm(res.x-y+(J[active].T@dual if np.any(active) else 0),np.inf)
            if g.max()>1e-7 or residual>1e-6:raise RuntimeError(f'Retry failed: primal={g.max()}, KKT={residual}')
            return res.x

class Policy:
    def __init__(self,seed):
        rng=np.random.default_rng(seed)
        self.p={'W':rng.normal(size=(D,H))*.15,'b':np.zeros(H),'v':rng.normal(size=H)*.05,'critic':np.zeros(D+1)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()};self.s={k:np.zeros_like(v) for k,v in self.p.items()};self.steps=0
    def forward(self,F):
        hidden=np.tanh(F@self.p['W']+self.p['b']);logits=hidden@self.p['v'];logits-=logits.max()
        prob=np.exp(logits);prob/=prob.sum()
        state=np.r_[F.mean(axis=0),1.]
        return prob,(F,hidden,state),float(state@self.p['critic'])
    def update(self,trajectory):
        # Episodic reward-to-go with a learned baseline; parameters fixed inside episode.
        returns=[];G=0.
        for _,_,_,r,_ in trajectory[::-1]:G=r+.99*G;returns.append(G)
        returns=returns[::-1]
        grads={k:np.zeros_like(v) for k,v in self.p.items()}
        for (cache,prob,action,_,value),G in zip(trajectory,returns):
            F,h,state=cache;adv=float(np.clip(G-value,-10,10))
            one=np.zeros(len(prob));one[action]=1.
            entropy=-np.sum(prob*np.log(prob+1e-20))
            entropy_grad=-prob*(np.log(prob+1e-20)+entropy)
            dz=-(adv*(one-prob)+.01*entropy_grad)
            grads['v']+=h.T@dz
            dh=dz[:,None]*self.p['v'][None,:]*(1-h*h)
            grads['W']+=F.T@dh;grads['b']+=dh.sum(axis=0)
            grads['critic']+=(value-G)*state
        for k in grads:grads[k]/=max(1,len(trajectory))
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()))
        self.steps+=1
        for k,g in grads.items():
            g=g/max(1.,norm)
            self.m[k]=.9*self.m[k]+.1*g;self.s[k]=.999*self.s[k]+.001*g*g
            mh=self.m[k]/(1-.9**self.steps);sh=self.s[k]/(1-.999**self.steps)
            self.p[k]-=.003*mh/(np.sqrt(sh)+1e-8)
        return float(np.mean(returns))
    def save(self,path):np.savez(path,**self.p)


class Env:
    def __init__(self,seed,cfg):
        self.obj,self.con,self.L,self.x=bcd.problem(seed,cfg);self.B=len(self.L)
        self.proj=Projector(self.con);self.age=np.zeros(self.B);self.drop=np.zeros(self.B)
        self.f,self.grad=self.obj(self.x);self.initial_f=self.f;self.scale=max(1.,np.linalg.norm(self.grad))
        self.steps=0;self.nnls_calls=0;self.feature_time=0.;self.cert_time=0.;self.cert_calls=0;self.trace=[]
        self.backtracks=0;self.rejected_steps=0
        self.feature_positive=0;self.feature_rows=0;self.check()
    def check(self):
        t=perf_counter();self.maps=np.zeros(self.B);feas=0.
        for b,L in enumerate(self.L):
            sl=slice(2*b,2*b+2);z=self.x[sl];p=self.proj(b,z-self.grad[sl]/L,z)
            self.maps[b]=np.linalg.norm(L*(z-p),np.inf)
            feas=max(feas,float(max(0.,self.con(b,z)[0].max())))
        self.res=float(self.maps.max());self.feas=feas;self.success=self.res<=1e-4 and feas<=1e-6
        self.cert_calls+=1;self.cert_time+=perf_counter()-t
        self.trace.append(dict(updates=self.steps,objective=self.f,stationarity=self.res,projection_calls=self.proj.calls))
    def features(self,variant,rng):
        t=perf_counter();g=self.grad.reshape(self.B,2)
        # Seven ordinary inputs, three geometry inputs. Same 10x16 architecture in all variants.
        F=np.zeros((self.B,D));F[:,0]=np.log1p(np.linalg.norm(g,axis=1));F[:,1]=np.log1p(self.maps)
        F[:,2]=np.minimum(self.age/self.B,5.)/5.;F[:,3]=np.clip(self.drop/self.scale,0,2)
        F[:,4]=np.log1p(self.L);F[:,5]=min(self.steps/600.,1.);F[:,6]=np.log1p(self.res)
        if variant!='base':
            for b in range(self.B):
                vals,J=self.con(b,self.x[2*b:2*b+2]);active=vals>=-1e-4
                F[b,9]=np.mean(active)
                if np.any(active):
                    lam=nnls(J[active].T,-g[b],maxiter=1000)[0];self.nnls_calls+=1
                    reaction=lam*np.linalg.norm(J[active],axis=1)
                    if reaction.sum()>0:
                        I=reaction.sum()/max(np.linalg.norm(J[active].T@lam),1e-12)
                        part=reaction.sum()/max(np.linalg.norm(g[b]),1e-12)
                        F[b,7]=np.log1p(min(I,10.));F[b,8]=min(part,2.);self.feature_positive+=1
                self.feature_rows+=1
            if variant=='shuffled':F[:,7:]=F[rng.permutation(self.B),7:]
        self.feature_time+=perf_counter()-t
        return F
    def step(self,b,candidate=None,limit=600):
        old_f=self.f;sl=slice(2*b,2*b+2)
        if candidate is None:candidate=self.proj(b,self.x[sl]-self.grad[sl]/self.L[b],self.x[sl])
        origin=self.x[sl].copy();direction=candidate-origin
        for attempt in range(25):
            self.x[sl]=origin+(0.5**attempt)*direction;new_f,new_grad=self.obj(self.x)
            if new_f<=old_f+1e-12:break
            self.backtracks+=1
        else:
            self.x[sl]=origin;new_f,new_grad=self.obj(self.x);self.rejected_steps+=1
        self.f,self.grad=new_f,new_grad
        self.age+=1;self.age[b]=0;self.drop[b]=max(0.,old_f-self.f);self.steps+=1
        reward=-.01;oldres=self.res
        if self.steps%self.B==0 or self.steps==limit:
            self.check()
            reward+=float(np.clip(np.log((oldres+1e-6)/(self.res+1e-6)),-5.,5.))
            if self.success:reward+=2.
        return reward
    def gs(self):
        scores=[];candidates=[]
        for b,L in enumerate(self.L):
            sl=slice(2*b,2*b+2);p=self.proj(b,self.x[sl]-self.grad[sl]/L,self.x[sl]);d=p-self.x[sl]
            scores.append(-self.grad[sl]@d-.5*L*(d@d));candidates.append(p)
        b=int(np.argmax(scores));return b,candidates[b]


def episode(seed,cfg,variant,policy,actionseed,limit=600,train=False):
    start=perf_counter();env=Env(seed,cfg);rng=np.random.default_rng(actionseed);trajectory=[];rewards=[];reason='update_budget'
    for k in range(limit):
        if env.success:reason='stationarity';break
        candidate=None
        if policy is not None:
            F=env.features(variant,rng);prob,cache,value=policy.forward(F)
            action=int(rng.choice(env.B,p=prob))
        elif variant=='cyclic':action=k%env.B
        elif variant=='random':action=int(rng.integers(env.B))
        else:action,candidate=env.gs()
        reward=env.step(action,candidate,limit=limit);rewards.append(reward)
        if train:trajectory.append((cache,prob,action,reward,value))
    if env.success:reason='stationarity'
    elapsed=perf_counter()-start
    result=dict(success=bool(env.success),reason=reason,objective=float(env.f),initial_objective=float(env.initial_f),
                stationarity=env.res,feasibility=env.feas,updates=env.steps,projection_calls=env.proj.calls,
                projection_iterations=env.proj.nit,projection_failures=env.proj.failures,backtracks=env.backtracks,rejected_steps=env.rejected_steps,time=elapsed,
                feature_time=env.feature_time,certificate_time=env.cert_time,nnls_calls=env.nnls_calls,
                positive_feature_fraction=env.feature_positive/max(1,env.feature_rows),reward=float(sum(rewards)),x=env.x.tolist(),trace=env.trace)
    if train:result['return_mean']=policy.update(trajectory)
    return result


TRAIN_CFGS=[dict(name='train_small',blocks=4,per_block=4,amplitude=.1,frequency=2.,coupling=.2),
            dict(name='train_medium',blocks=8,per_block=16,amplitude=.4,frequency=4.,coupling=.5)]
TEST_CFGS=[dict(TRAIN_CFGS[1],name='heldout_ID'),
           dict(name='heldout_OOD',blocks=12,per_block=64,amplitude=.6,frequency=6.,coupling=.8)]


def main():
    out=ROOT/'results/rl_bcd_pilot';out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir()
    config=dict(algorithm='episodic REINFORCE, learned linear state baseline, Adam',train_seeds=[71,83,97],
                episodes_per_model=80,train_horizon=128,test_horizon=600,variants=['base','regime','shuffled'],
                train_conditions=TRAIN_CFGS,test_conditions=TEST_CFGS,test_instances_per_condition=6,
                architecture=dict(inputs=D,hidden=H,shared_block_scorer='tanh'),learning_rate=.003,gamma=.99,entropy=.01,
                reward='-0.01 per update + clipped log decrease of projected stationarity each B updates +2 at success',
                evaluation='stochastic frozen policy, single action draw sequence per model-instance; no adaptation',
                checkpoints='final fixed training budget, no validation/test checkpoint selection')
    metadata=dict(config=config,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),bcd_sha256=hashlib.sha256(P.read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));models={};start=perf_counter()
    with (out/'training.jsonl').open('w',buffering=1) as handle:
        for ts in config['train_seeds']:
            for variant in config['variants']:
                policy=Policy(ts);t0=perf_counter()
                for ep in range(config['episodes_per_model']):
                    cfg=TRAIN_CFGS[ep%2]
                    row=episode(2000000+ts*1000+ep,cfg,variant,policy,900000+ts*1000+ep,limit=128,train=True)
                    for k in ['x','trace']:row.pop(k)
                    handle.write(json.dumps(dict(train_seed=ts,variant=variant,episode=ep,**row))+'\n')
                policy.save(out/'models'/f'{variant}_{ts}.npz');models[variant,ts]=policy
                print(f'trained {variant} seed={ts}: {perf_counter()-t0:.1f}s',flush=True)
    tasks=[]
    for ci,cfg in enumerate(TEST_CFGS):
        for j in range(6):
            ps=7000000+ci*10000+j
            for variant,ts in models:tasks.append((ci,j,ps,variant,ts))
            for variant in ['cyclic','random','gs']:tasks.append((ci,j,ps,variant,None))
    rng=np.random.default_rng(831);rng.shuffle(tasks)
    with (out/'test.jsonl').open('w',buffering=1) as handle:
        for ci,j,ps,variant,ts in tasks:
            cfg=TEST_CFGS[ci];policy=models.get((variant,ts))
            row=episode(ps,cfg,variant,policy,8000000+ci*10000+j*100+(ts or 0))
            handle.write(json.dumps(dict(condition=cfg['name'],instance=j,problem_seed=ps,variant=variant,train_seed=ts,**row))+'\n')
    (out/'elapsed.json').write_text(json.dumps(dict(total_seconds=perf_counter()-start)))
    print(f'Completed in {perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__':main()
