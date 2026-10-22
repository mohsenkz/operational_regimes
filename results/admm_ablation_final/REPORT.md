# ADMM paired ablation — completed

Scope: full execution of configs/benchmarks/admm_ablation.json, NOT all MIMO/drift experiments in the repository.

672 runs: 336 calibration runs (4 conditions × 4 seeds × 3 rho values × 7 methods), and 336 held-out runs (4 conditions × 12 seeds × 7 methods). Wall time: 269.2 seconds. One thread. Shared alpha=1; maxit=4000. Rho selected separately per method and condition using calibration only. Test channel seeds are disjoint from calibration.

| Method | Common-certificate success /48 | Median iterations, successful runs | Median online seconds, all runs |
|---|---:|---:|---:|
| fixed |48|290.5|0.2691|
| balance |36|165.5|0.1654|
| balance_normalized |35|168|0.2016|
| history |39|179|0.1843|
| signature |39|179|0.1934|
| label |39|179|0.1801|
| shuffled |39|179|0.1925|

Success additionally requires relative objective error <=1e-4, cone violation <=1e-5, and phase equality <=1e-7. A run can satisfy internal scale-dependent ADMM stopping tolerances yet fail this external certificate. Such a run is not necessarily divergent. The solver currently stops on the internal criterion; it does NOT continue until the external certificate passes. Consequently reported runtime is runtime to internal stopping/budget, not time-to-common-accuracy for failed certificates. Successful-only iteration medians must not be compared without success rates.

Main finding: no demonstrated added benefit of signature over residual history. On the 39 mutually successful cases, signature/history iterations tie in every case. Paired geometric mean online time ratio is 1.0164; conditional seed-bootstrap 95% interval [0.9699,1.0646]. Signature damping is activated only once across all 48 test runs; label damping never activates. Thus this is a weak intervention test, NOT evidence that informative regime interventions are impossible. Shuffled control also has 39 successes and identical successful-case iteration counts.

Fixed rho has the best common-certificate success here (48/48). History versus balance_normalized improves success from 35 to 39, but changes both wins and losses across paired instances. No broad superiority claim is justified.

Timing: online includes controller, factorization, signatures, residual checks; excludes shared problem construction and oracle solves. Per-run uncached_setup_time adds construction and the gref reference solve for feature methods. Signature mean feature cost is about 7.08 ms/run. Timing is single-run, subject to system noise; bootstrap is conditional on fixed conditions and mutually successful cases, not distributional generalization.

Reference robustness: initial attempts aborted on condition=3, calibration seed=33 because CLARABEL failed. No sample was removed. Final source retries with SCS (eps=1e-9, max_iters=100000) after both CLARABEL attempts throw; all references passed independently checked cone/equality feasibility <=1e-7. CVXPY emitted an inaccurate-solution warning; feasibility checks alone are not an independent dual-gap certificate. Oracle solver choice/status is not logged individually in this version. Final raw records and metadata belong only to the completed run; earlier partial runs are excluded.

Files: runs.jsonl (raw results), cases.jsonl (reference/check/setup data), summary.json (aggregates and paired comparisons), metadata.json (configuration, source hashes, versions), selected_rho.json, calibration_signature_pools.json. Execution log is included separately.

Reproduce from repository root (install numpy scipy cvxpy clarabel scs threadpoolctl first):

```bash
OPENBLAS_NUM_THREADS=1 python experiments/admm/benchmark_ablation.py \
  --config configs/benchmarks/admm_ablation.json --output results/new_run
python -m unittest discover -s tests -v
```

All 13 tests passed after final changes. Dependency installation in this session required a temporary /mnt/data/admm_deps path; that environment is not bundled.
