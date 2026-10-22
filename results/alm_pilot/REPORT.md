# ALM exploratory pilot — completed

## Protocol
207 solves: 63 calibration and 144 held-out test. Three synthetic families (generic/correlated/opposed constraint gradients), 3 calibration seeds and 12 fresh test seeds per family. Test seeds 201–212 differ from the earlier SQP test. No direct numerical comparison of ALM vs SQP is warranted because samples and work units differ.

Same six-variable/eight-constraint generator as SQP: nonconvex sinusoidal/quartic objective, convex quadratic inequalities, known strictly feasible zero. These are small favorable problems, not a broad hard-NLP benchmark.

Powell–Hestenes–Rockafellar inequality ALM:
A(x,lambda,rho)=f(x)+(||max(0,lambda+rho*g(x))||²-||lambda||²)/(2*rho).
Inner minimization: L-BFGS-B, analytic gradient, gtol=1e-7, ftol=1e-15, maxiter=400. Actual ALM dual update lambda+=max(0,lambda+rho*g); no multiplier replacement by NNLS. Initial rho=1. Outer limit=60, nominal aggregate inner evaluation budget=20000 (L-BFGS-B can slightly overrun maxfun); rho cap=1e8. No test reached the evaluation limit.

Violation v=||max(g,-lambda_old/rho)||inf. If v>.5*previous_v, increase penalty by factor 5, except the following damped variants may use factor 2 or 3:
- baseline: always factor 5;
- history: damp if v>.9*previous_v;
- signature: damp if reaction is nonzero and I>1.3;
- random: damp under independent Bernoulli(.5) trigger.

I is computed with the repository reaction-feature routine using current actual ALM multipliers. This tests one feature, not all regime information or a learned classifier. Damping is an empirical hypothesis, not a theoretical consequence of a regime. Random is a placebo, NOT a distribution-matched permutation. Factor selected per family/method on calibration by failures, then capped evaluation count. Baseline factor fixed, not extensively tuned. Task execution order randomized. There is no claim of optimal hyperparameter tuning or general superiority.

Success requires primal violation<=1e-6, ||grad f+J^T lambda||inf<=1e-4, and max|lambda_i*g_i|<=1e-5. This is first-order KKT only; no SOSC/global certificate. All final duals nonnegative.

## Test results
| Method | Success /36 | Median inner evaluations | Median outer iterations | Damped penalty updates |
|---|---:|---:|---:|---:|
| baseline |36|260.5|15|0|
| history |36|270|15|6|
| signature |36|269|15.5|33|
| random |36|276.5|16|81|

Signature vs baseline, paired: 3 evaluation-count wins, 24 ties, 9 losses. Geometric mean evaluation ratio 1.0324; exploratory paired bootstrap 95% interval [1.0043,1.0658] (10000 pooled-instance resamples, unadjusted; not a confirmatory significance test). Versus history: ratio 1.0278, interval [0.9994,1.0620].

Conclusion: no demonstrated improvement from this signature-based ALM penalty damping. It preserved success rate but used approximately 3.2% more evaluations on geometric average than baseline. This finding is limited to the heuristic and sample tested.

Objectives nearly coincide: maximum signature/base difference 1.04e-5; signature/history 8.68e-6. These tiny differences can reflect residual constraint tolerances; they do not establish a better local optimum. Do not interpret objective threshold counts as substantive wins.

Single-run median times: baseline .0413 s; history .0405 s; signature .0435 s; random .0434 s. Too short and unreplicated for robust speed claims. Runtime includes feature computation and logging trace construction inside solve; excludes initial problem generation and disk writing.

Inner solver reported non-success 3 times for baseline/history/signature and once for random across test outer iterations. These statuses were retained, not filtered; all runs subsequently met the external certificate. Trace records messages. Maximum test violations: feasibility 9.982e-7; stationarity 3.016e-6; complementarity 9.579e-6.

## Verification and artifacts
Four unit tests passed: augmented-gradient finite differences across all three families and three penalty scales; exact interior solution; known active-constraint solution/multiplier; complementarity enforcement. Import is safe. Raw traces/results in runs.jsonl, configuration/software/hash in metadata.json, selected factors and aggregates in summary.json, paired checks in verification.json.

Reproduce from repository root (requires NumPy/SciPy):
```
OPENBLAS_NUM_THREADS=1 python experiments/alm/benchmark_alm.py
python -m unittest discover -s tests -p test_alm.py -v
```
Output directory results/alm_pilot must not exist before running; archive it first. Temporary dependency directory used in this environment is not included in the archive.

## Research implication
The ADMM, SQP and ALM pilots do not currently support a general algorithmic-acceleration claim. They do not falsify regime classification or all interventions. Avoid further tuning on these held-out seeds; prioritize fresh diagnostic/classification benchmarks or preregister a more directly motivated intervention with fresh tests.
