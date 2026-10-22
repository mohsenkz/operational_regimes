# Interior-point exploratory pilot

## Scope and method
Completed 207 solves: 63 calibration and 144 test, three synthetic families and four variants. Calibration seeds 71/82/93; fresh test seeds 301–312; 12 test instances per family. Generator shared with prior SQP/ALM: six variables, eight convex quadratic inequalities, nonconvex objective. Different seeds prevent direct solver ranking against earlier pilots.

This is a feasible primal logarithmic-barrier Newton method, NOT a production primal-dual predictor-corrector solver or an IPOPT benchmark. Start is made strictly feasible by contracting the shared sampled x0 toward the known strictly feasible origin. This exploits favorable problem structure and omits general phase-I feasibility search. All variants use identical starts per instance.

Minimize f(x)-mu*sum(log(-g_i(x))). Analytic gradient and Hessian include constraint curvature. Newton Hessian eigenvalues floored at 1e-5; Armijo backtracking preserves strict interior. Initial mu=.1; baseline mu reduction=.2. Inner stationarity target max(1e-7,.1*mu), maximum 100 inner iterations per barrier value, maximum 30 outer stages, nominal 10000 full evaluation budget. Success can be certified before fully centering. Failure to center within 100 inner steps terminates the run; no rescue/restart was added after viewing test results. Budget checks can be exceeded by a few diagnostic/final evaluations.

Actual barrier dual lambda_i=mu/(-g_i), not NNLS. Success: primal violation<=1e-6 (strict feasibility maintained), infinity-norm Lagrangian stationarity<=1e-4, complementarity<=1e-5. No SOSC or global optimality certificate.

## Intervention hypotheses
If triggered, reduce mu less aggressively by factor .4 or .6 rather than .2:
- baseline: no trigger;
- conditioning: last Newton direction's clipped-Hessian condition>1e4 (zero if no direction needed at that stage);
- signature: nonzero total reaction and repository cancellation feature I>1.3;
- random: Bernoulli(.5) placebo, NOT a matched permutation.

Damping central-path progress under cancellation is a heuristic hypothesis, not a theorem. This tests one scalar signature feature, not an entire classifier. Factor selected per family/method on calibration by failures, then capped full evaluation count; baseline not extensively tuned. Execution order randomized. Selected values in summary.json. No tuning on held-out seeds.

## Results
| Method | Success /36 | Median full evaluations, all runs | Median Newton steps, all runs | Damped updates |
|---|---:|---:|---:|---:|
| baseline |32|182|56.5|0|
| conditioning |29|208.5|76.5|165|
| signature |31|218.5|73.5|149|
| random |28|234|82|140|

All failures were inner-iteration-budget terminations, not evidence of divergence. All final iterates finite and strictly feasible. Failure counts: baseline 4, conditioning 7, signature 5, random 8. Runtime on failures is time to budget termination, not time to a shared accuracy.

Signature/base: 30 mutually successful cases; 2 evaluation-count wins, 0 ties, 28 losses. One signature-only success and two baseline-only successes (three cases fail in both). Conditional geometric mean evaluation ratio=1.188; exploratory 95% paired-bootstrap interval [0.915,1.572]. This is NOT statistically conclusive evidence of slowdown and is conditional on common success. No demonstrated improvement. Signature/base maximum objective difference on common-success cases 1.92e-5; no differences over 1e-3. Small objective differences can arise from different complementarity accuracy.

Full evaluation counts include final certificates, initialization, and signature diagnostics (including duplicate evaluations); therefore include information-acquisition overhead, not just Newton linear solves. Per-family metadata and raw traces allow reconstruction. Newton step counts are reported separately. Single-run median time: base .0277s, conditioning .0377s, signature .0341s, random .0371s. Too short and unreplicated for robust speed claims.

## Verification
Four unit tests passed: barrier gradients AND Hessians by finite differences for three families and three mu values; analytic active-constraint optimum and dual; interior optimum; rejection of noninterior points. Unit tests also verify lambda_i*(-g_i)=mu. Import does not run experiments.

Artifacts: metadata.json (configuration, software, source hashes), runs.jsonl (all calibration/test outcomes, iterates, actual duals, full outer traces), summary.json, verification.json (paired results, failure reasons). Bootstrap uses 10000 pooled instance resamples, unadjusted and exploratory.

Reproduce from repository root (NumPy/SciPy required):
```
OPENBLAS_NUM_THREADS=1 python experiments/interior_point/benchmark_ip.py
python -m unittest discover -s tests -p test_interior_point.py -v
```
Archive/remove the old results/interior_point_pilot directory before reproducing. Temporary environment dependencies are not bundled.

## Interpretation
This pilot does not support signature-based slower barrier reduction as an acceleration strategy. It does not test all interior-point methods, other signature interventions, or the original solution-classification objective. The earlier ADMM/SQP/ALM pilots likewise do not establish broad acceleration. Before claiming algorithmic utility, use more directly justified interventions, production-grade baselines, stronger diagnostic controls, and new held-out instances. Classification for real-time decisions remains a separate empirical question requiring feasibility, objective regret, fallback and end-to-end latency evaluation.
