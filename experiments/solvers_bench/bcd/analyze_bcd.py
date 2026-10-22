"""Post-run checks/report; does not alter calibration or solver settings."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.optimize import nnls

ROOT=Path(__file__).resolve().parents[2]
P=Path(__file__).with_name('benchmark_bcd.py')
spec=importlib.util.spec_from_file_location('bcd',P);bcd=importlib.util.module_from_spec(spec);spec.loader.exec_module(bcd)


def main():
    out=ROOT/'results/bcd_pilot'
    rows=[json.loads(line) for line in (out/'runs.jsonl').read_text().splitlines()]
    metadata=json.loads((out/'metadata.json').read_text())
    assert hashlib.sha256(P.read_bytes()).hexdigest()==metadata['source_sha256']
    assert len(rows)==132
    test=[r for r in rows if r['split']=='test'];assert len(test)==90
    verification=[]
    for r in rows:
        ci=next(i for i,c in enumerate(bcd.CONDITIONS) if c['name']==r['condition'])
        obj,con,L,_=bcd.problem(r['seed']+ci*100000,bcd.CONDITIONS[ci]);x=np.array(r['x'])
        f,grad=obj(x);assert abs(f-r['objective'])<1e-9
        feas=max(max(0.,con(b,x[2*b:2*b+2])[0].max()) for b in range(len(L)))
        # Independently evaluate ordinary KKT residual with active NNLS multipliers.
        stationarity=0.;complementarity=0.
        for b in range(len(L)):
            sl=slice(2*b,2*b+2);g,J=con(b,x[sl]);active=g>=-1e-5
            lam=nnls(J[active].T,-grad[sl],maxiter=1000)[0] if np.any(active) else np.zeros(0)
            stationarity=max(stationarity,float(np.linalg.norm(grad[sl]+(J[active].T@lam if np.any(active) else 0),np.inf)))
            if len(lam):complementarity=max(complementarity,float(np.max(np.abs(lam*g[active]))))
        vals=[t['objective'] for t in r['trace']]
        assert all(v<=u+1e-7 for u,v in zip(vals,vals[1:]))
        assert feas<=1e-7
        verification.append(dict(condition=r['condition'],seed=r['seed'],split=r['split'],method=r['method'],alpha=r['alpha'],feasibility=feas,kkt_stationarity=stationarity,complementarity=complementarity,reported_success=r['success']))
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    rng=np.random.default_rng(9981);comparisons={}
    for other in ['cyclic','random','gs','shuffled']:
        pairs=[]
        for s in test:
            if s['method']!='signature':continue
            t=next(t for t in test if t['method']==other and t['condition']==s['condition'] and t['seed']==s['seed'])
            if s['success'] and t['success']:pairs.append((s,t))
        cmp={}
        for metric in ['projection_calls','time','updates']:
            lr=np.array([np.log(s[metric]/t[metric]) for s,t in pairs])
            boot=np.exp(lr[rng.integers(len(lr),size=(5000,len(lr)))].mean(axis=1))
            cmp[metric]=dict(geomean_signature_over_control=float(np.exp(lr.mean())),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),wins=sum(s[metric]<t[metric] for s,t in pairs),ties=sum(s[metric]==t[metric] for s,t in pairs),losses=sum(s[metric]>t[metric] for s,t in pairs))
        cmp['n_joint_success']=len(pairs)
        cmp['objective_signature_minus_control']=[s['objective']-t['objective'] for s,t in pairs]
        comparisons[other]=cmp
    (out/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
    summary=json.loads((out/'summary.json').read_text())
    text=['# Projected BCD block-selection pilot','',
          '## Scope and protocol',
          '- 132 runs: 42 calibration, 90 test. Two calibration seeds and six held-out test seeds per condition. Each test method has 18 runs.',
          '- Conditions: (n,m)=(8,20), (16,136), (24,780). Constraints include the unit disk per block. Two variables per block.',
          '- Objective: 0.5 (x-target)^T Q (x-target) + amplitude * sum(sin(frequency*x)) + 0.02 sum(x^4). Dense positive-definite coupled Q; sinusoidal term can make the objective nonconvex.',
          '- Convex block-separable quadratic inequalities a_i^T z+c_i||z||^2-offset_i <=0 and ||z||^2<=1. Strictly feasible initial point zero; no Phase I. NOT nonconvex constraints and NOT coupled constraints.',
          '- Size, constraint count, and nonlinear strength are changed together: this design cannot identify their separate causal effects. Many constraints can be inactive/redundant; total constraint count is not active-set size.',
          '- Projected gradient step in a selected block using a safe block Lipschitz bound. This is projected BCD, not exact block minimization or an industrial solver.',
          '- Cyclic, uniform-random, projected Gauss–Southwell (GS), signature-weighted GS, and within-iteration shuffled-weight GS. Shuffling preserves the current multiset of weights, not trajectories across methods.',
          '- GS scores predicted decrease -g_b^T d_b - L_b||d_b||^2/2. Geometry-based variants multiply by 1+alpha*w_b.',
          '- Near-active threshold 1e-4; nonnegative least squares fits local multipliers. I=sum(lambda_i||J_i||)/||J^T lambda||, clipped at 10. w=log(1+I)*min(sum(lambda_i||J_i||)/||grad_b||,2), zero without positive reaction. This is a continuous geometry-derived heuristic, NOT a trained classifier and NOT a theorem that regimes improve BCD.',
          '- Alpha in {0.25,1} selected per condition/method on calibration, lexicographically by failures then total projection calls. Test settings are not retuned.',
          '- Maximum 600 block updates. Full certificate every B updates. Success: infinity norm of block projected-gradient mappings <=1e-4 and feasibility <=1e-6. First-order stationarity only; no global or second-order certificate.',
          '- All projection work, full certificate evaluations, and feature computations included in wall time. Full GS variants compute candidate projections for every block; cyclic/random only for chosen blocks. Projection-call count is not a complete operation count, particularly as m grows.',
          '- Convex projections use SLSQP with analytic Jacobians; primal and NNLS KKT checks determine acceptance, not SLSQP status alone. Source hash, package versions and settings in metadata.json.',
          '- An initial debug execution stopped because SLSQP returned a failure status for a numerically valid projection. Acceptance was corrected to explicit primal/KKT validation and the complete protocol restarted; no policy tuning based on that partial run.',
          '', '## Aggregate test results','', '| Method | Success | Median block updates | Median projection calls | Median seconds |', '|---|---:|---:|---:|---:|']
    for method,s in summary['methods'].items():
        text.append(f"| {method} | {s['success']}/{s['n']} | {s['median_updates']:.0f} | {s['median_projection_calls']:.0f} | {s['median_time']:.4f} |")
    text+=['','## Per-condition results','', '| Condition | Method | Success | Median calls | Median seconds |','|---|---|---:|---:|---:|']
    for c in bcd.CONDITIONS:
        for method in summary['methods']:
            rr=[r for r in test if r['condition']==c['name'] and r['method']==method]
            text.append(f"| {c['name']} | {method} | {sum(r['success'] for r in rr)}/6 | {np.median([r['projection_calls'] for r in rr]):.0f} | {np.median([r['time'] for r in rr]):.4f} |")
    text+=['','## Paired comparisons (joint successes only)','', 'Ratios are signature/control; smaller is better. Bootstrap over problem pairs (5000 draws) is exploratory, unadjusted for multiple comparisons; small heterogeneous sample and success conditioning limit interpretation.','']
    for other,c in comparisons.items():
        text.append(f"### Against {other}: {c['n_joint_success']} joint successes")
        for metric in ['projection_calls','time','updates']:
            d=c[metric];lo,hi=d['bootstrap95']
            text.append(f"- {metric}: geometric ratio {d['geomean_signature_over_control']:.3f}, exploratory 95% [{lo:.3f}, {hi:.3f}]; wins/ties/losses {d['wins']}/{d['ties']}/{d['losses']}.")
        gaps=c['objective_signature_minus_control'];text.append(f'- Objective differences signature-control: median {np.median(gaps):.6g}, range [{min(gaps):.6g}, {max(gaps):.6g}]. Different stationary points need not have equal quality.')
    text+=['','## Verification and limitations',
           '- Four automated tests: finite-difference objective/constraint derivatives; projection feasibility and sampled variational inequality; descent under safe steps; reproducibility and cost accounting.',
           '- Every stored result independently rechecked for objective and feasibility; checkpoint objectives checked for monotonic descent. Separate active-set NNLS KKT residuals are in verification.json.',
           '- Wall times are single measurements in one environment, randomized method order, not repeated performance trials. Include feature overhead; do not treat small timing differences as robust.',
           '- Six test problems per condition are exploratory, not a scalable benchmark. This does not test thousands of coupled nonconvex constraints or prove that regime information cannot help elsewhere.',
           '- Budget failures are not divergence. Success is not global optimality. No trained classification/routing or prediction benchmark has been evaluated.',
           '', '## Interpretation',
           '- No convincing incremental signature benefit over GS: both succeed on 18/18 tests; projection work ties in 17/18 pairs. Signature saves calls once but has geometric time ratio 1.160 (about 16% slower) in this single timing pass.',
           '- Both GS variants solve the many-constraint instance where cyclic/random hit the budget. This is evidence for selective scheduling here, not an incremental signature effect.',
           '- Fewer block updates need not imply lower total cost: inspecting all blocks increases projection work. Objective gaps on joint successes are negligible here.',
           '- Maximum independently measured ordinary KKT stationarity among successes: 2.78e-4; complementarity below 8.2e-8. Success uses the specified projected-gradient mapping, not an ordinary 1e-4 KKT certificate.',
           '', '## Reproduce', '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/bcd/benchmark_bcd.py','# requires absent results/bcd_pilot directory; archive existing results first','python experiments/bcd/analyze_bcd.py','python -m unittest discover -s tests -p test_bcd.py -v','```','Dependencies: Python 3, NumPy, SciPy. Exact installed versions in metadata.json.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':main()
