"""Verification and descriptive held-out analysis; never tunes policies."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).with_name('benchmark_rl.py')
spec=importlib.util.spec_from_file_location('rl',P);rl=importlib.util.module_from_spec(spec);spec.loader.exec_module(rl)


def main():
    out=ROOT/'results/rl_bcd_pilot'
    meta=json.loads((out/'metadata.json').read_text());assert hashlib.sha256(P.read_bytes()).hexdigest()==meta['source_sha256']
    train=[json.loads(l) for l in (out/'training.jsonl').read_text().splitlines()]
    test=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()]
    assert len(train)==720 and len(test)==144
    summary={};verification=[]
    for r in test:
        cfg=next(c for c in rl.TEST_CFGS if c['name']==r['condition'])
        obj,con,L,_=rl.bcd.problem(r['problem_seed'],cfg);x=np.array(r['x']);f,g=obj(x)
        assert abs(f-r['objective'])<1e-9
        feas=max(max(0.,con(b,x[2*b:2*b+2])[0].max()) for b in range(len(L)))
        assert feas<1e-7
        vals=[t['objective'] for t in r['trace']]
        assert all(b<=a+1e-7 for a,b in zip(vals,vals[1:]))
        projection=rl.Projector(con);cert=rl.bcd.diagnostic(obj,con,L,x,projection)
        assert abs(cert['stationarity']-r['stationarity'])<1e-7
        assert cert['success']==r['success']
        kkt=0.;comp=0.
        for b in range(len(L)):
            sl=slice(2*b,2*b+2);v,J=con(b,x[sl]);active=v>=-1e-5
            lam=nnls(J[active].T,-g[sl],maxiter=1000)[0] if np.any(active) else np.zeros(0)
            kkt=max(kkt,float(np.linalg.norm(g[sl]+(J[active].T@lam if np.any(active) else 0),np.inf)))
            if len(lam):comp=max(comp,float(np.max(np.abs(lam*v[active]))))
        verification.append(dict(condition=r['condition'],instance=r['instance'],variant=r['variant'],train_seed=r['train_seed'],success=r['success'],feasibility=feas,kkt_stationarity=kkt,complementarity=comp))
    variants=['base','regime','shuffled','cyclic','random','gs']
    for cond in ['all']+[c['name'] for c in rl.TEST_CFGS]:
        summary[cond]={}
        for variant in variants:
            rr=[r for r in test if r['variant']==variant and (cond=='all' or r['condition']==cond)]
            summary[cond][variant]=dict(n=len(rr),success=sum(r['success'] for r in rr),median_updates=float(np.median([r['updates'] for r in rr])),
                    median_calls=float(np.median([r['projection_calls'] for r in rr])),median_seconds=float(np.median([r['time'] for r in rr])),
                    median_stationarity=float(np.median([r['stationarity'] for r in rr])),median_feature_seconds=float(np.median([r['feature_time'] for r in rr])))
    per_seed={}
    for variant in ['base','regime','shuffled']:
        per_seed[variant]={}
        for seed in meta['config']['train_seeds']:
            tt=[r for r in train if r['variant']==variant and r['train_seed']==seed]
            rr=[r for r in test if r['variant']==variant and r['train_seed']==seed]
            policy=rl.Policy(seed);weights=np.load(out/'models'/f'{variant}_{seed}.npz')
            delta=np.sqrt(sum(float(np.sum((weights[k]-v)**2)) for k,v in policy.p.items()))
            per_seed[variant][seed]=dict(success=sum(r['success'] for r in rr),n=len(rr),train_updates=sum(r['updates'] for r in tt),
                    train_seconds=sum(r['time'] for r in tt),reward_first20=float(np.mean([r['reward'] for r in tt[:20]])),
                    reward_last20=float(np.mean([r['reward'] for r in tt[-20:]])),parameter_change=float(delta))
    # Average log-ratios across training seeds within instance, then bootstrap instances.
    # This avoids treating the 36 trained-policy runs as 36 independent problems.
    paired={};rng=np.random.default_rng(166)
    for other in ['base','shuffled']:
        logratios=[];objectives=[];details=[]
        for cfg in rl.TEST_CFGS:
            for j in range(6):
                aa=[r for r in test if r['variant']=='regime' and r['condition']==cfg['name'] and r['instance']==j]
                ratios=[];gaps=[]
                for a in aa:
                    b=next(r for r in test if r['variant']==other and r['condition']==cfg['name'] and r['instance']==j and r['train_seed']==a['train_seed'])
                    ratios.append(np.log((a['stationarity']+1e-6)/(b['stationarity']+1e-6)));gaps.append(a['objective']-b['objective'])
                logratios.append(float(np.mean(ratios)));objectives.append(float(np.mean(gaps)))
                details.append(dict(condition=cfg['name'],instance=j,log_stationarity_ratio=float(np.mean(ratios)),mean_objective_gap=float(np.mean(gaps))))
        arr=np.array(logratios);boot=np.exp(arr[rng.integers(len(arr),size=(5000,len(arr)))].mean(axis=1))
        paired[other]=dict(geometric_stationarity_ratio=float(np.exp(arr.mean())),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                            median_objective_gap=float(np.median(objectives)),n_independent_instances=12,instance_details=details)
    (out/'summary.json').write_text(json.dumps(dict(summary=summary,per_training_seed=per_seed,paired=paired),indent=2))
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    text=['# Ordinary RL for BCD block selection: exploratory pilot','',
          '## Protocol and scope',
          '- 720 training episodes: 3 feature variants x 3 initialization/training seeds x 80 episodes. Up to 128 block updates per episode. Final fixed-budget checkpoint, no validation/test tuning.',
          '- 144 held-out evaluation runs: 12 independent problem instances x (9 trained policies + cyclic/random/GS). Each learned variant has 36 runs but only 12 independent problem instances; 3 training seeds are repeated measurements, not extra independent problems.',
          '- No meta-learning, adaptation, pretrained policy or test-time optimization of weights. Ordinary on-policy episodic REINFORCE, gamma=0.99, learned linear state-value baseline, entropy coefficient 0.01, Adam lr=0.003, gradient norm clipping. All parameters in metadata.json.',
          '- A shared per-block tanh network (10 inputs, 16 hidden units, scalar logit) yields a categorical distribution over blocks. Frozen stochastic sampling at test, single evaluation draw sequence for each model-instance; not deterministic argmax.',
          '- Base/regime/shuffled have the same architecture and training instance/action seed schedules. Geometry inputs are zeroed for base. Shuffling changes the random-number stream as well as trajectories, and preserves the current geometry-row multiset within each decision.',
          '- Ordinary inputs: block gradient norm, last checkpoint projected-gradient norm, age since selection, last objective improvement, block Lipschitz bound, elapsed-update fraction, global checkpoint residual. Cached mappings are stale between full certificates.',
          '- Added geometry: log(1+clipped cancellation ratio), clipped reaction participation, near-active constraint fraction. NNLS estimates local nonnegative multipliers at current points, not oracle optimal multipliers. These are continuous regime-inspired features, NOT trained regime classification labels.',
          '- Near-active threshold 1e-4. I=sum(lambda_i||J_i||)/||J^T lambda|| clipped at 10; participation clipped at 2. All extraction overhead is included in online timing.',
          '- Reward: -0.01 per block update, plus log decrease of full projected-gradient residual (clipped to [-5,5]) every B updates, plus 2 on reaching success. No reward based solely on objective decrease. Reward does NOT directly penalize geometry-computation wall time; it is an update-efficiency training objective, and online cost is evaluated separately.',
          '- Training alternates (n,m)=(8,20) and (16,136), with frequency 2/4 and amplitude 0.1/0.4. ID tests: (16,136), frequency 4, amplitude 0.4. OOD tests: (24,780), frequency 6, amplitude 0.6, stronger coupling. Size, curvature and constraint count change together, so OOD effects cannot be separated.',
          '- Nonconvex coupled objective; convex block-separable quadratic inequalities plus unit disks. Safe projected-gradient block updates, not exact block minimization. All initial points strictly feasible; no Phase I. NOT a benchmark of nonconvex coupled constraints or industrial-scale hard optimization.',
          '- Test budget 600 block updates. Success: infinity norm of projected-gradient mapping <=1e-4, feasibility <=1e-6, evaluated every B updates. Budget exhaustion is not divergence; first-order success is not global or second-order optimality.',
          '- Every method pays for common full certificates; GS additionally projects every candidate block at each decision. RL projects only the chosen block, but regime/shuffled additionally evaluate all geometry features. Timing includes environment construction, inference, projections, certificates and feature extraction, excludes offline training.',
          '- Same maximum episode/step budget across learned variants; actual training steps differ with early success. Training instance seeds are disjoint from test seeds. A new BCD problem generator is reused but not earlier pilot test instances.',
          '', '## Aggregate held-out results','', '| Method | Success | Median updates | Median projection calls | Median seconds | Median stationarity |','|---|---:|---:|---:|---:|---:|']
    for v,s in summary['all'].items():text.append(f"| {v} | {s['success']}/{s['n']} | {s['median_updates']:.0f} | {s['median_calls']:.0f} | {s['median_seconds']:.4f} | {s['median_stationarity']:.3g} |")
    text+=['','## Per-distribution results','','| Distribution | Method | Success | Median seconds | Median residual |','|---|---|---:|---:|---:|']
    for cond in [c['name'] for c in rl.TEST_CFGS]:
        for v,s in summary[cond].items():text.append(f"| {cond} | {v} | {s['success']}/{s['n']} | {s['median_seconds']:.4f} | {s['median_stationarity']:.3g} |")
    text+=['','## Training-seed sensitivity','','| Variant | Seed | Test success | Actual training updates | Training rollout seconds | Mean reward first/last 20 |','|---|---:|---:|---:|---:|---:|']
    for v,seeds in per_seed.items():
        for seed,s in seeds.items():text.append(f"| {v} | {seed} | {s['success']}/{s['n']} | {s['train_updates']} | {s['train_seconds']:.1f} | {s['reward_first20']:.3f} / {s['reward_last20']:.3f} |")
    text+=['','Training rollout times exclude optimizer-update and serialization overhead. The process was interrupted after 58 test rows; resume_rl.py completed only the 86 missing tasks using identical fixed weights and seeds. Original total elapsed time is unavailable (elapsed.json records null and resume time); all per-episode timings are retained. Timings span two process sessions and should be treated as approximate. Training reward changes across different sampled problems are descriptive, not evidence of generalization.','',
           '## Exploratory paired comparison',
           'Final residual ratios include successes and failures, with floor 1e-6. Ratios <1 favor regime. Average logs across training seeds within problem; bootstrap 12 problem instances, 5000 draws. Unadjusted exploratory intervals; heterogeneous ID/OOD mixture and very small sample. Not an efficiency or objective-quality guarantee.']
    for other,s in paired.items():
        lo,hi=s['bootstrap95'];text.append(f"- Regime/{other}: geometric residual ratio {s['geometric_stationarity_ratio']:.3f}, exploratory 95% [{lo:.3f},{hi:.3f}]. Median instance-averaged objective gap {s['median_objective_gap']:.6g}.")
    text+=['','## Checks and limitations',
           '- Six automated tests cover the descent safeguard, certified projection retry, policy-gradient finite differences, positive-advantage update direction, zero-mask/shuffled features, and deterministic replay including training updates. The BCD test suite separately checks objective/constraint derivatives, projection optimality samples and descent.',
           '- All 144 stored test solutions recomputed for objective, feasibility, projected stationarity and monotonic checkpoint objectives. Independent active-set NNLS ordinary KKT residuals also stored; this is not independent implementation of the projection routine.',
           '- Every method uses the same certified SLSQP projector. A failed 1e-12 solve is retried from zero with ftol=1e-14 and 500 iterations; primal <=1e-7 and projection KKT <=1e-6 remain mandatory. projection_failures records initial failed checks, including recovered retries. An initial incomplete run stopped on such a failure and is archived, not pooled with the completed run.',
           '- Every method applies the same feasibility-preserving segment backtracking if finite-precision projection causes an objective increase; up to 25 halvings, otherwise retain the original point. All objective calls and retry overhead are in online time; backtracks/rejected_steps are recorded. A second incomplete run stopped on a monotonicity check before this safeguard; both incomplete runs are excluded from final statistics.',
           '- Single timing pass, tiny network, 80 episodes and only 3 training seeds. No systematic hyperparameter search. A weak RL result does not rule out richer policies, more training or other tasks; a positive difference does not establish broad regime utility.',
           '- Base already receives cached projected-gradient information, so this is incremental information value beyond a meaningful standard state, not regime versus no constraint information.',
           '- No explicit fallback beyond safe projected steps. No labels derived from solved test optima. No theorem connecting these heuristic weights to optimal scheduling.',
           '', '## Reproduce', '```bash','pip install -r experiments/rl_bcd/requirements.txt','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/rl_bcd/benchmark_rl.py','# results directory must not already exist; archive it first','python experiments/rl_bcd/analyze_rl.py','python -m unittest discover -s tests -p test_rl_bcd.py -v','```']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(summary=summary,paired=paired),indent=2))

if __name__=='__main__':main()
