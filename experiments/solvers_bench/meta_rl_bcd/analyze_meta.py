"""Independent stored-solution checks and exploratory task-clustered summaries."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
from scipy.optimize import nnls
P=Path(__file__).with_name('benchmark_meta.py');spec=importlib.util.spec_from_file_location('m',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
METHODS=['reptile_frozen','reptile_adapted','pooled_frozen','pooled_adapted','random_adapted']
def main():
 out=m.OUT;meta=json.loads((out/'metadata.json').read_text())
 for name,h in meta['source_hashes'].items():assert hashlib.sha256((m.ROOT/name).read_bytes()).hexdigest()==h
 train=[json.loads(l) for l in (out/'training.jsonl').read_text().splitlines()]
 rows=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()]
 schedule=json.loads((out/'schedule.json').read_text());assert len(train)==288 and len(rows)==len(schedule)==198
 assert len({r['job_id'] for r in rows})==198
 verified=[]
 for r in rows:
  for k,v in schedule[r['job_id']].items():assert r[k]==v
  assert len(r['support'])==(2 if r['method'].endswith('_adapted') else 0)
  if r['method'].endswith('_frozen'):assert r['parameter_change']==0
  cfg=r['cfg'];obj,con,L,_=m.rl.bcd.problem(r['query_seed'],cfg);x=np.array(r['x']);f,g=obj(x)
  assert abs(f-r['objective'])<1e-9
  feas=max(max(0.,con(b,x[2*b:2*b+2])[0].max()) for b in range(len(L)))
  assert feas<=1e-7
  vals=[t['objective'] for t in r['trace']];assert all(b<=a+1e-7 for a,b in zip(vals,vals[1:]))
  cert=m.rl.bcd.diagnostic(obj,con,L,x,m.rl.Projector(con))
  assert abs(cert['stationarity']-r['stationarity'])<=1e-7 and cert['success']==r['success']
  kkt=0.;comp=0.
  for b in range(len(L)):
   sl=slice(2*b,2*b+2);v,J=con(b,x[sl]);active=v>=-1e-5
   lam=nnls(J[active].T,-g[sl],maxiter=1000)[0] if np.any(active) else np.zeros(0)
   kkt=max(kkt,float(np.linalg.norm(g[sl]+(J[active].T@lam if np.any(active) else 0),np.inf)))
   if len(lam):comp=max(comp,float(np.max(np.abs(lam*v[active]))))
  verified.append(dict(job_id=r['job_id'],feasibility=feas,kkt=kkt,complementarity=comp,success=r['success']))
 summary={}
 def stats(rr):
  return dict(n=len(rr),success=sum(r['success'] for r in rr),median_residual=float(np.median([r['stationarity'] for r in rr])),
    median_updates=float(np.median([r['updates'] for r in rr])),median_query_seconds=float(np.median([r['time'] for r in rr])),
    median_total_seconds=float(np.median([r['total_online_seconds'] for r in rr])),median_adaptation_seconds=float(np.median([r['adaptation_seconds'] for r in rr])),
    median_feature_seconds=float(np.median([r['feature_time'] for r in rr])),median_projection_calls=float(np.median([r['projection_calls'] for r in rr])))
 for family in ['all']+[c['name'] for c in m.TEST]:
  summary[family]={}
  for method in METHODS:
   for variant in m.VARIANTS:
    rr=[r for r in rows if r['method']==method and r['variant']==variant and (family=='all' or r['family']==family)]
    summary[family][method+'/'+variant]=stats(rr)
  for method in ['cyclic','random','gs']:
   rr=[r for r in rows if r['method']==method and (family=='all' or r['family']==family)]
   summary[family][method]=stats(rr)
 # Cluster training-seed repetitions within the six held-out tasks. Bootstrap
 # three tasks within each family, maintaining the fixed 50/50 family mixture.
 rng=np.random.default_rng(611);draws=np.c_[rng.integers(0,3,size=(5000,3)),rng.integers(3,6,size=(5000,3))]
 def paired(method_a,variant_a,method_b,variant_b):
  residual=[];timing=[];success=[];objective=[]
  for family in [c['name'] for c in m.TEST]:
   for task in range(3):
    a=[r for r in rows if r['family']==family and r['task']==task and r['method']==method_a and r['variant']==variant_a]
    logs=[];times=[];succ=[];gaps=[]
    for x in a:
     y=next(r for r in rows if r['family']==family and r['task']==task and r['method']==method_b and r['variant']==variant_b and r['train_seed']==x['train_seed'])
     logs.append(np.log((x['stationarity']+1e-6)/(y['stationarity']+1e-6)))
     times.append(np.log(x['total_online_seconds']/y['total_online_seconds']))
     succ.append(int(x['success'])-int(y['success']));gaps.append(x['objective']-y['objective'])
    residual.append(np.mean(logs));timing.append(np.mean(times));success.append(np.mean(succ));objective.append(np.mean(gaps))
  def ratio(arr):
   a=np.array(arr);return dict(geometric_ratio=float(np.exp(a.mean())),bootstrap95=np.quantile(np.exp(a[draws].mean(axis=1)),[.025,.975]).tolist())
  return dict(n_tasks=6,residual=ratio(residual),total_online_time=ratio(timing),mean_success_difference=float(np.mean(success)),median_task_objective_gap=float(np.median(objective)))
 comparisons={}
 for variant in m.VARIANTS:
  for a,b in [('reptile_adapted','pooled_adapted'),('reptile_adapted','reptile_frozen'),('reptile_adapted','random_adapted'),('pooled_adapted','pooled_frozen')]:
   comparisons[f'{a}/{b}:{variant}']=paired(a,variant,b,variant)
 for method in METHODS:
  for other in ['base','shuffled']:comparisons[f'regime/{other}:{method}']=paired(method,'regime',method,other)
 seeds={}
 for method in METHODS:
  for variant in m.VARIANTS:
   seeds[method+'/'+variant]={str(seed):stats([r for r in rows if r['method']==method and r['variant']==variant and r['train_seed']==seed]) for seed in m.SEEDS}
 result=dict(summary=summary,paired=comparisons,per_seed=seeds,checks=dict(verified=len(verified),max_feasibility=max(v['feasibility'] for v in verified),backtracks=sum(r['backtracks'] for r in rows),rejected_steps=sum(r['rejected_steps'] for r in rows),projection_retries=sum(r['projection_failures'] for r in rows)))
 (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'verification.json').write_text(json.dumps(verified,indent=2))
 lines=['# Meta-RL BCD: small exploratory Reptile pilot','','## Predeclared protocol',
 '- First-order Reptile over ordinary REINFORCE+linear critic inner updates. This is meta-learning of an initialization, not MAML, a recurrent memory agent, or a theorem about regime usefulness.',
 '- 2 training seeds x 3 feature variants x 2 training algorithms x 12 tasks x 2 episodes = 288 training episodes, each capped at 96 updates. Reptile inner Adam resets at every task; pooled ordinary RL uses persistent Adam. Identical initial weights, task/instance schedules, architectures and per-variant interaction budgets; early stopping can change realized counts.',
 '- Reptile interpolation linearly decreases from 0.25 to 0.05; both actor and critic are meta-initialized. Each task has two independent problem instances. Pooled training receives exactly the same scheduled instances. Final fixed-budget checkpoints; no test or validation tuning.',
 '- Three training families, two entirely held-out parameter families (unseen_D and unseen_E). Task coupling/amplitude jitter is sampled once and shared by its support and query instances. Family IDs are disjoint; all instances use the same synthetic generator. This is parameter-family holdout, NOT evidence of transfer to a new generator or application.',
 '- 6 independent held-out tasks, 3 per family, one query instance per task. 2 training seeds are repeated measurements, NOT 12 independent tasks. 180 learned-policy query runs + 18 cyclic/random/GS = 198. Very small pilot with only two seeds.',
 '- Five learned-policy conditions: Reptile frozen/adapted, pooled ordinary RL frozen/adapted, and random initialization adapted. Every adapted condition gets exactly two support episodes capped at 96 updates; fresh Adam, same support instances and action seed schedules. Support instances differ from query instances. Query weights are hashed before/after to assert no query learning.',
 '- Query budget 384 updates for every method; unlike the earlier ordinary RL experiment (600), so cross-pilot success numbers are not directly comparable. Success uses projected stationarity <=1e-4 and feasibility <=1e-6. All 198 stored solutions are independently recomputed; ordinary active-set NNLS KKT residuals stored separately. Projection itself reuses the certified implementation.',
 '- Feature variants base/regime/shuffled and 10x16 shared scorer are unchanged. Regime inputs are continuous local NNLS geometry features, not regime class labels. Base already has cached projected-gradient features. Shuffling also changes the random stream, so it is not a perfectly coupled placebo.',
 '- Online time includes support rollouts, optimizer updates, initialization/cloning and query execution. Query time includes feature extraction, certificates, projections and inference. Offline training is reported separately. Costs are NOT amortized over multiple queries; one new task pays for its two support episodes. Frozen methods pay no support cost. Single timing measurement, no hardware-general latency claim.',
 '- Safe projected block steps, identical numerical guards for all methods, convex block-separable constraints and nonconvex coupled objectives. First-order success is not global or second-order optimality. Budget exhaustion does not prove divergence. The reward does not directly penalize feature-computation wall time.',
 '', '## All held-out tasks','', '| Method / features | Success | Median residual | Median updates | Query sec | Total online sec |','|---|---:|---:|---:|---:|---:|']
 for k,s in summary['all'].items():lines.append(f"| {k} | {s['success']}/{s['n']} | {s['median_residual']:.3g} | {s['median_updates']:.0f} | {s['median_query_seconds']:.3f} | {s['median_total_seconds']:.3f} |")
 lines+=['','## By held-out family','','| Family | Method / features | Success | Median residual | Total online sec |','|---|---|---:|---:|---:|']
 for family in [c['name'] for c in m.TEST]:
  for k,s in summary[family].items():lines.append(f"| {family} | {k} | {s['success']}/{s['n']} | {s['median_residual']:.3g} | {s['median_total_seconds']:.3f} |")
 lines+=['','## Exploratory paired contrasts','Residual and total-time geometric ratios below 1 favor the numerator. Average log-ratios across training seeds within task, then bootstrap tasks within each held-out family (5000 draws). Six task clusters only; unadjusted exploratory 95% intervals, not confirmatory inference. Residual floor 1e-6; all successes and budget failures included. Residuals are affected by stopping thresholds and do not directly measure solution quality.']
 for k,s in comparisons.items():
  r=s['residual'];t=s['total_online_time'];lines.append(f"- {k}: residual {r['geometric_ratio']:.3f} [{r['bootstrap95'][0]:.3f}, {r['bootstrap95'][1]:.3f}]; online time {t['geometric_ratio']:.3f} [{t['bootstrap95'][0]:.3f}, {t['bootstrap95'][1]:.3f}]; mean success difference {s['mean_success_difference']:.3f}.")
 lines+=['','## Training and checks']
 times=json.loads((out/'training_times.json').read_text());lines.append(f"- Offline training wall time (including optimizer updates/model save): {sum(t['seconds'] for t in times):.2f} s total across 12 models. Per-model times: training_times.json. Full run: elapsed.json.")
 lines.append('- Training-seed sensitivity, projection calls, feature/adaptation times and paired objective gaps: summary.json; complete support/query records: test.jsonl; all configurations and source hashes: metadata.json.')
 lines.append('- Verification: '+json.dumps(result['checks']))
 lines+=['- Four meta-specific automated tests cover independent clone/reset, exact Reptile interpolation, inner learning/query freeze, task reproducibility/family holdout. Existing BCD/RL tests cover derivatives, projection certification, gradients and numerical safeguards.',
 '- Equal interaction budgets do not imply equal optimizer behavior or converged training. Tiny training budgets, two seeds, one query per task, and no hyperparameter search limit all conclusions. Weak results cannot reject meta-RL in general. Positive differences cannot establish general regime utility.',
 '', '## Interpretation', '- Reptile adapted/regime succeeds on 10/12 repeated-seed evaluations versus 9/12 for pooled adapted/regime, but random-initialization adapted/regime also reaches 10/12 and the same median residual. This is not a compelling meta-initialization advantage.', '- Local exploratory residual improvements are present: Reptile adapted/regime versus pooled adapted/regime has ratio 0.711 [0.599, 0.911], and versus Reptile adapted/base 0.712 [0.449, 0.923]. However, the comparison against shuffled is inconclusive, random-adapted performance is nearly identical, and many residuals are already below the stopping threshold.', '- Adaptation leaves regime success unchanged versus frozen Reptile (10/12 each), while median online time rises from 1.363 to 2.440 seconds. GS succeeds on all six query instances, with median 108 updates and 1.403 seconds; its results are not repeated over training seeds.', '- Overall: limited local residual gains, but no convincing cost-adjusted advantage specifically attributable to meta-learning or regime information in this small fixed-budget pilot. This neither rejects meta-RL generally nor evaluates classification.', '', '## Reproduce','```bash','pip install -r experiments/rl_bcd/requirements.txt', '# Archive results/meta_rl_bcd_pilot first for a fresh run.', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/meta_rl_bcd/benchmark_meta.py','python experiments/meta_rl_bcd/analyze_meta.py', 'python -m unittest discover -s tests -p "test_*bcd.py" -v', '# If interrupted after schedule.json exists, finish only missing evaluations:', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/meta_rl_bcd/benchmark_meta.py --resume','```']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary['all'],checks=result['checks']),indent=2))
if __name__=='__main__':main()
