# Projected BCD block-selection pilot

## Scope and protocol
- 132 runs: 42 calibration, 90 test. Two calibration seeds and six held-out test seeds per condition. Each test method has 18 runs.
- Conditions: (n,m)=(8,20), (16,136), (24,780). Constraints include the unit disk per block. Two variables per block.
- Objective: 0.5 (x-target)^T Q (x-target) + amplitude * sum(sin(frequency*x)) + 0.02 sum(x^4). Dense positive-definite coupled Q; sinusoidal term can make the objective nonconvex.
- Convex block-separable quadratic inequalities a_i^T z+c_i||z||^2-offset_i <=0 and ||z||^2<=1. Strictly feasible initial point zero; no Phase I. NOT nonconvex constraints and NOT coupled constraints.
- Size, constraint count, and nonlinear strength are changed together: this design cannot identify their separate causal effects. Many constraints can be inactive/redundant; total constraint count is not active-set size.
- Projected gradient step in a selected block using a safe block Lipschitz bound. This is projected BCD, not exact block minimization or an industrial solver.
- Cyclic, uniform-random, projected Gauss–Southwell (GS), signature-weighted GS, and within-iteration shuffled-weight GS. Shuffling preserves the current multiset of weights, not trajectories across methods.
- GS scores predicted decrease -g_b^T d_b - L_b||d_b||^2/2. Geometry-based variants multiply by 1+alpha*w_b.
- Near-active threshold 1e-4; nonnegative least squares fits local multipliers. I=sum(lambda_i||J_i||)/||J^T lambda||, clipped at 10. w=log(1+I)*min(sum(lambda_i||J_i||)/||grad_b||,2), zero without positive reaction. This is a continuous geometry-derived heuristic, NOT a trained classifier and NOT a theorem that regimes improve BCD.
- Alpha in {0.25,1} selected per condition/method on calibration, lexicographically by failures then total projection calls. Test settings are not retuned.
- Maximum 600 block updates. Full certificate every B updates. Success: infinity norm of block projected-gradient mappings <=1e-4 and feasibility <=1e-6. First-order stationarity only; no global or second-order certificate.
- All projection work, full certificate evaluations, and feature computations included in wall time. Full GS variants compute candidate projections for every block; cyclic/random only for chosen blocks. Projection-call count is not a complete operation count, particularly as m grows.
- Convex projections use SLSQP with analytic Jacobians; primal and NNLS KKT checks determine acceptance, not SLSQP status alone. Source hash, package versions and settings in metadata.json.
- An initial debug execution stopped because SLSQP returned a failure status for a numerically valid projection. Acceptance was corrected to explicit primal/KKT validation and the complete protocol restarted; no policy tuning based on that partial run.

## Aggregate test results

| Method | Success | Median block updates | Median projection calls | Median seconds |
|---|---:|---:|---:|---:|
| cyclic | 17/18 | 168 | 344 | 0.4718 |
| random | 17/18 | 142 | 294 | 0.4677 |
| gs | 18/18 | 60 | 646 | 0.8880 |
| signature | 18/18 | 60 | 646 | 0.9621 |
| shuffled | 17/18 | 72 | 656 | 1.3538 |

## Per-condition results

| Condition | Method | Success | Median calls | Median seconds |
|---|---|---:|---:|---:|
| small_mild | cyclic | 6/6 | 80 | 0.1671 |
| small_mild | random | 6/6 | 100 | 0.1985 |
| small_mild | gs | 6/6 | 124 | 0.2275 |
| small_mild | signature | 6/6 | 124 | 0.2648 |
| small_mild | shuffled | 5/6 | 144 | 0.2755 |
| medium_nonlinear | cyclic | 6/6 | 384 | 0.5257 |
| medium_nonlinear | random | 6/6 | 336 | 0.5404 |
| medium_nonlinear | gs | 6/6 | 656 | 1.0340 |
| medium_nonlinear | signature | 6/6 | 656 | 1.2321 |
| medium_nonlinear | shuffled | 6/6 | 656 | 1.4506 |
| many_constraints | cyclic | 5/6 | 456 | 0.7370 |
| many_constraints | random | 5/6 | 516 | 1.0555 |
| many_constraints | gs | 6/6 | 1338 | 2.4268 |
| many_constraints | signature | 6/6 | 1338 | 2.7404 |
| many_constraints | shuffled | 6/6 | 1338 | 2.5875 |

## Paired comparisons (joint successes only)

Ratios are signature/control; smaller is better. Bootstrap over problem pairs (5000 draws) is exploratory, unadjusted for multiple comparisons; small heterogeneous sample and success conditioning limit interpretation.

### Against cyclic: 17 joint successes
- projection_calls: geometric ratio 1.832, exploratory 95% [1.588, 2.126]; wins/ties/losses 0/0/17.
- time: geometric ratio 2.170, exploratory 95% [1.760, 2.691]; wins/ties/losses 1/0/16.
- updates: geometric ratio 0.458, exploratory 95% [0.384, 0.538]; wins/ties/losses 17/0/0.
- Objective differences signature-control: median -6.39773e-10, range [-8.26761e-09, 4.90932e-09]. Different stationary points need not have equal quality.
### Against random: 17 joint successes
- projection_calls: geometric ratio 1.649, exploratory 95% [1.390, 1.947]; wins/ties/losses 4/0/13.
- time: geometric ratio 1.842, exploratory 95% [1.530, 2.215]; wins/ties/losses 0/0/17.
- updates: geometric ratio 0.409, exploratory 95% [0.375, 0.448]; wins/ties/losses 17/0/0.
- Objective differences signature-control: median -2.49258e-10, range [-5.78945e-09, 8.18872e-08]. Different stationary points need not have equal quality.
### Against gs: 18 joint successes
- projection_calls: geometric ratio 0.990, exploratory 95% [0.971, 1.000]; wins/ties/losses 1/17/0.
- time: geometric ratio 1.160, exploratory 95% [1.086, 1.235]; wins/ties/losses 2/0/16.
- updates: geometric ratio 0.990, exploratory 95% [0.970, 1.000]; wins/ties/losses 1/17/0.
- Objective differences signature-control: median 0, range [-2.57486e-09, 2.21833e-09]. Different stationary points need not have equal quality.
### Against shuffled: 17 joint successes
- projection_calls: geometric ratio 0.995, exploratory 95% [0.969, 1.015]; wins/ties/losses 1/15/1.
- time: geometric ratio 0.946, exploratory 95% [0.841, 1.053]; wins/ties/losses 8/0/9.
- updates: geometric ratio 0.994, exploratory 95% [0.968, 1.015]; wins/ties/losses 1/15/1.
- Objective differences signature-control: median 2.84217e-14, range [-3.40687e-09, 1.75789e-08]. Different stationary points need not have equal quality.

## Verification and limitations
- Four automated tests: finite-difference objective/constraint derivatives; projection feasibility and sampled variational inequality; descent under safe steps; reproducibility and cost accounting.
- Every stored result independently rechecked for objective and feasibility; checkpoint objectives checked for monotonic descent. Separate active-set NNLS KKT residuals are in verification.json.
- Wall times are single measurements in one environment, randomized method order, not repeated performance trials. Include feature overhead; do not treat small timing differences as robust.
- Six test problems per condition are exploratory, not a scalable benchmark. This does not test thousands of coupled nonconvex constraints or prove that regime information cannot help elsewhere.
- Budget failures are not divergence. Success is not global optimality. No trained classification/routing or prediction benchmark has been evaluated.

## Interpretation
- No convincing incremental benefit of the signature heuristic over ordinary GS: both succeed on 18/18 tests, and projection work is identical in 17/18 paired tests. Signature saves projection calls in one case, but paired geometric wall-time ratio is 1.160 (about 16% slower) in this single timing pass.
- Both GS variants succeed on the many-constraint instance where cyclic and random hit the update budget. This supports selective block scheduling in this pilot, not an incremental signature effect.
- Fewer block updates do not imply lower end-to-end cost: inspecting every block raises projection work and time. Objective differences among jointly successful methods are negligible here.
- Independent ordinary KKT residuals differ from the projected-gradient stopping criterion: maximum infinity-norm KKT stationarity among reported successes is 2.78e-4, with maximum complementarity below 8.2e-8. Reported success uses the stated projected-mapping criterion, not a 1e-4 ordinary KKT certificate.

## Reproduce
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/bcd/benchmark_bcd.py
# requires absent results/bcd_pilot directory; archive existing results first
python experiments/bcd/analyze_bcd.py
python -m unittest discover -s tests -p test_bcd.py -v
```
Dependencies: Python 3, NumPy, SciPy. Exact installed versions in metadata.json.
