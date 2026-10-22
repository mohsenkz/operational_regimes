"""Small first-order Reptile meta-RL pilot; no test-set tuning."""
from pathlib import Path
import importlib.util, json, hashlib, platform
from time import perf_counter
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'experiments/rl_bcd/benchmark_rl.py'
spec=importlib.util.spec_from_file_location('rl',P);rl=importlib.util.module_from_spec(spec);spec.loader.exec_module(rl)
OUT=ROOT/'results/meta_rl_bcd_pilot'
SEEDS=[113,127];VARIANTS=['base','regime','shuffled'];OUTER=12;INNER=2;HORIZON=96;QUERY=384
TRAIN=[dict(name='train_A',blocks=4,per_block=4,amplitude=.1,frequency=2.,coupling=.2),
       dict(name='train_B',blocks=6,per_block=8,amplitude=.25,frequency=3.,coupling=.35),
       dict(name='train_C',blocks=8,per_block=16,amplitude=.4,frequency=4.,coupling=.5)]
TEST=[dict(name='unseen_D',blocks=8,per_block=24,amplitude=.5,frequency=5.,coupling=.65),
      dict(name='unseen_E',blocks=12,per_block=48,amplitude=.65,frequency=7.,coupling=.85)]
def clone(policy):
 q=rl.Policy(0)
 q.p={k:v.copy() for k,v in policy.p.items()}
 return q

def interpolate(policy,adapted,rate):
 for k in policy.p:policy.p[k]+=rate*(adapted.p[k]-policy.p[k])

def task(base,seed):
 rng=np.random.default_rng(seed);cfg=dict(base)
 cfg['coupling']*=float(rng.uniform(.9,1.1));cfg['amplitude']*=float(rng.uniform(.9,1.1))
 return cfg

def compact(row):return {k:v for k,v in row.items() if k not in ['x','trace']}
def training(algorithm,variant,seed):
 p=rl.Policy(seed);start=perf_counter();rows=[]
 for outer in range(OUTER):
  cfg=task(TRAIN[outer%len(TRAIN)],31000000+outer)
  inner=clone(p) if algorithm=='reptile' else p
  for ep in range(INNER):
   r=rl.episode(32000000+outer*10+ep,cfg,variant,inner,33000000+seed*1000+outer*10+ep,limit=HORIZON,train=True)
   rows.append(dict(algorithm=algorithm,variant=variant,train_seed=seed,outer=outer,inner=ep,cfg=cfg,**compact(r)))
  if algorithm=='reptile':interpolate(p,inner,.25-(.20*outer/max(1,OUTER-1)))
 p.save(OUT/'models'/f'{algorithm}_{variant}_{seed}.npz')
 return p,rows,perf_counter()-start

def main():
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'models').mkdir()
 config=dict(seeds=SEEDS,variants=VARIANTS,outer_tasks=OUTER,inner_episodes=INNER,train_horizon=HORIZON,query_horizon=QUERY,
             train_families=TRAIN,test_families=TEST,tasks_per_test_family=3,support_episodes=2,
             algorithm='first-order Reptile over REINFORCE+critic inner updates',meta_rate='linear .25 to .05',
             optimizer='fresh Adam per meta-task and per test adaptation; persistent Adam for pooled training',
             checkpoint='final fixed budget; no validation or test tuning',adaptation='policy+critic, two independent support instances; optimizer reset; no query gradients')
 meta=dict(config=config,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
           source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),P,rl.P]})
 (OUT/'metadata.json').write_text(json.dumps(meta,indent=2));start=perf_counter();models={};times=[]
 with (OUT/'training.jsonl').open('w',buffering=1) as f:
  for seed in SEEDS:
   for variant in VARIANTS:
    for algorithm in ['reptile','pooled']:
     p,rows,elapsed=training(algorithm,variant,seed);models[algorithm,variant,seed]=p
     for r in rows:f.write(json.dumps(r)+'\n')
     times.append(dict(algorithm=algorithm,variant=variant,seed=seed,seconds=elapsed))
     print('trained',algorithm,variant,seed,round(elapsed,2),flush=True)
 (OUT/'training_times.json').write_text(json.dumps(times,indent=2))
 # Save the predeclared evaluation schedule; resume can only run missing entries.
 jobs=[]
 for fi,base in enumerate(TEST):
  for j in range(3):
   cfg=task(base,41000000+fi*100+j)
   for seed in SEEDS:
    for variant in VARIANTS:
     for method in ['reptile_frozen','reptile_adapted','pooled_frozen','pooled_adapted','random_adapted']:
      jobs.append(dict(family=base['name'],task=j,cfg=cfg,train_seed=seed,variant=variant,method=method))
   for method in ['cyclic','random','gs']:
    jobs.append(dict(family=base['name'],task=j,cfg=cfg,train_seed=None,variant=method,method=method))
 np.random.default_rng(419).shuffle(jobs)
 (OUT/'schedule.json').write_text(json.dumps(jobs,indent=2))
 evaluate(jobs,models)
 (OUT/'elapsed.json').write_text(json.dumps(dict(total_seconds=perf_counter()-start)))

def evaluate(jobs,models=None):
 done=set()
 if (OUT/'test.jsonl').exists():
  done={r['job_id'] for r in map(json.loads,(OUT/'test.jsonl').read_text().splitlines())}
 models={} if models is None else models
 with (OUT/'test.jsonl').open('a',buffering=1) as f:
  for i,job in enumerate(jobs):
   if i in done:continue
   fi=next(k for k,c in enumerate(TEST) if c['name']==job['family']);j=job['task'];seed=job['train_seed'];variant=job['variant'];method=job['method'];cfg=job['cfg']
   start=perf_counter();support=[];policy=None;initial=None
   if seed is not None:
    algorithm=method.split('_')[0]
    if algorithm=='random':initial=rl.Policy(seed)
    else:
     key=(algorithm,variant,seed)
     if key not in models:
      p=rl.Policy(seed)
      with np.load(OUT/'models'/f'{algorithm}_{variant}_{seed}.npz') as weights:p.p={k:weights[k].copy() for k in p.p}
      models[key]=p
     initial=models[key]
    policy=clone(initial)
    if method.endswith('_adapted'):
     for ep in range(2):
      r=rl.episode(42000000+fi*10000+j*10+ep,cfg,variant,policy,43000000+fi*100000+j*1000+seed*3+ep,limit=HORIZON,train=True)
      support.append(compact(r))
   adapt_seconds=perf_counter()-start
   delta=0. if policy is None else float(np.sqrt(sum(np.sum((policy.p[k]-initial.p[k])**2) for k in policy.p)))
   digest=lambda p:None if p is None else hashlib.sha256(b''.join(p.p[k].tobytes() for k in sorted(p.p))).hexdigest()
   before=digest(policy)
   query_seed=44000000+fi*10000+j
   row=rl.episode(query_seed,cfg,variant,policy,45000000+fi*100000+j*1000+(seed or 0),limit=QUERY)
   assert before==digest(policy),'Query evaluation changed weights'
   f.write(json.dumps(dict(job_id=i,**job,query_seed=query_seed,support=support,adaptation_seconds=adapt_seconds,parameter_change=delta,query_weights_hash=before,total_online_seconds=adapt_seconds+row['time'],**row))+'\n')
   print(f'evaluated {i+1}/{len(jobs)}',flush=True)
if __name__=='__main__':
 import sys
 if '--resume' in sys.argv:
  meta=json.loads((OUT/'metadata.json').read_text())
  for name,h in meta['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
  evaluate(json.loads((OUT/'schedule.json').read_text()))
 else:main()
