"""Resume missing fixed-policy evaluations, without modifying benchmark or retraining."""
from pathlib import Path
import importlib.util,json,hashlib
from time import perf_counter
import numpy as np
P=Path(__file__).with_name('benchmark_rl.py')
spec=importlib.util.spec_from_file_location('rl',P);rl=importlib.util.module_from_spec(spec);spec.loader.exec_module(rl)
def main():
 out=rl.ROOT/'results/rl_bcd_pilot';meta=json.loads((out/'metadata.json').read_text())
 assert hashlib.sha256(P.read_bytes()).hexdigest()==meta['source_sha256']
 models={}
 for ts in meta['config']['train_seeds']:
  for variant in meta['config']['variants']:
   policy=rl.Policy(ts)
   with np.load(out/'models'/f'{variant}_{ts}.npz') as weights:
    for k in policy.p:policy.p[k]=weights[k].copy()
   models[variant,ts]=policy
 rows=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()]
 done={(r['condition'],r['instance'],r['variant'],r['train_seed']) for r in rows}
 assert len(done)==len(rows)
 tasks=[]
 for ci,cfg in enumerate(rl.TEST_CFGS):
  for j in range(6):
   ps=7000000+ci*10000+j
   for variant,ts in models:tasks.append((ci,j,ps,variant,ts))
   for variant in ['cyclic','random','gs']:tasks.append((ci,j,ps,variant,None))
 np.random.default_rng(831).shuffle(tasks)
 start=perf_counter();completed=0
 with (out/'test.jsonl').open('a',buffering=1) as f:
  for ci,j,ps,variant,ts in tasks:
   cfg=rl.TEST_CFGS[ci]
   if (cfg['name'],j,variant,ts) in done:continue
   row=rl.episode(ps,cfg,variant,models.get((variant,ts)),8000000+ci*10000+j*100+(ts or 0))
   f.write(json.dumps(dict(condition=cfg['name'],instance=j,problem_seed=ps,variant=variant,train_seed=ts,**row))+'\n')
   completed+=1;print(f'Completed {len(done)+completed}/144',flush=True)
 (out/'elapsed.json').write_text(json.dumps(dict(total_seconds=None,resume_seconds=perf_counter()-start,previous_test_rows=len(done),note='Process interrupted between evaluations. Resumed only missing tasks using fixed saved weights and original seeds. Original total elapsed unavailable; per-episode timing preserved.')))
if __name__=='__main__':main()
