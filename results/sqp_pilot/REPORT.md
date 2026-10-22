# SQP exploratory pilot

Completed 207 solves: 63 calibration, 144 held-out tests. Three families, 3 calibration seeds/family, 12 test seeds/family, four methods. Calibration chooses strength from {0.001, 0.01} by failures then capped QP count; baseline has no additional regularization. Same initialization per paired instance. Runtime about 10.4 seconds.

This is a small custom SQP implementation, not a benchmark of production SQP packages. It uses exact Lagrangian Hessians convexified by eigenvalue clipping, elastic QPs solved using SciPy SLSQP, and merit backtracking. Nonconvex sinusoidal/quartic objectives, six variables and eight convex quadratic constraints. Zero is strictly feasible. Gradient families: generic, correlated, opposed. This is not evidence on large or broadly hard nonlinear programs; constraints themselves are convex. Multipliers are local NNLS estimates from near-active constraints, not certified theoretical regime labels.

Interventions: baseline; extra regularization triggered by Hessian condition >1e4; extra regularization triggered by positive reaction and I>1.3; Bernoulli(0.5) placebo trigger named shuffled. This placebo is NOT a distribution-matched signature permutation. Signature labels use repository formulas and adaptive ksat, but the intervention uses I, not regime-name direction. Scaling reference is the current gradient norm, a practical exploratory choice. No theorem derives these interventions.

Success: primal violation <=1e-6, infinity-norm Lagrangian stationarity <=1e-4, complementarity <=1e-5. No global optimality or second-order certificate.

|Method|Test success /36|Median QPs, all test runs|Triggered steps|
|---|---:|---:|---:|
|baseline|36|7|0|
|conditioning|34|6.5|37|
|signature|34|7|71|
|shuffled/placebo|34|7|123|

Signature vs baseline on 34 mutually successful cases: 2 QP-count wins, 30 ties, 2 losses; no objective differences >1e-5. It fails the common certificate in two additional cases. Thus there is no demonstrated improvement. Unlike the ADMM pilot, the signature intervention actually activates often. This negative result applies to this heuristic only, not all potential uses of regime information.

Conditioning control has 11 QP-count wins / 19 ties / 4 losses among 34 mutually successful cases, but two worse local objectives and two extra failures. It does not establish unqualified improvement either.

Single-run timing ratios should not be treated as robust speedups: baseline solves take about 41 ms median and timing is noisy. Signature paired geometric mean ratio 0.926 is conditional on mutually successful cases and does not offset its reduced success rate. No repeated timing or uncertainty assessment was performed.

Analytic gradients, Hessians and constraint Jacobians passed finite-difference checks at one independent point per family (max error <2e-9). Import performed without triggering experiments. Detailed checks and paired objective comparisons are in verification.json. Raw traces/results: runs.jsonl. Configuration and software/source hash: metadata.json. Aggregates: summary.json.

Next research step: do not tune further on these held-out seeds. Either preregister a stronger intervention and use fresh tests, or prioritize the original diagnostic/classification goal. Regime classification alone is not an equivalent reformulation of the full continuous optimization solution; evaluate solution prediction, feasibility repair/fallback, decision regret and end-to-end latency separately.

Reproduce from repository root:
OPENBLAS_NUM_THREADS=1 python experiments/sqp/benchmark_sqp.py
(The output directory must not already exist.)
