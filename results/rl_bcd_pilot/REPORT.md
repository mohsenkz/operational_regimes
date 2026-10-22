# Ordinary RL for BCD block selection: exploratory pilot

## Protocol and scope
- 720 training episodes: 3 feature variants x 3 initialization/training seeds x 80 episodes. Up to 128 block updates per episode. Final fixed-budget checkpoint, no validation/test tuning.
- 144 held-out evaluation runs: 12 independent problem instances x (9 trained policies + cyclic/random/GS). Each learned variant has 36 runs but only 12 independent problem instances; 3 training seeds are repeated measurements, not extra independent problems.
- No meta-learning, adaptation, pretrained policy or test-time optimization of weights. Ordinary on-policy episodic REINFORCE, gamma=0.99, learned linear state-value baseline, entropy coefficient 0.01, Adam lr=0.003, gradient norm clipping. All parameters in metadata.json.
- A shared per-block tanh network (10 inputs, 16 hidden units, scalar logit) yields a categorical distribution over blocks. Frozen stochastic sampling at test, single evaluation draw sequence for each model-instance; not deterministic argmax.
- Base/regime/shuffled have the same architecture and training instance/action seed schedules. Geometry inputs are zeroed for base. Shuffling changes the random-number stream as well as trajectories, and preserves the current geometry-row multiset within each decision.
- Ordinary inputs: block gradient norm, last checkpoint projected-gradient norm, age since selection, last objective improvement, block Lipschitz bound, elapsed-update fraction, global checkpoint residual. Cached mappings are stale between full certificates.
- Added geometry: log(1+clipped cancellation ratio), clipped reaction participation, near-active constraint fraction. NNLS estimates local nonnegative multipliers at current points, not oracle optimal multipliers. These are continuous regime-inspired features, NOT trained regime classification labels.
- Near-active threshold 1e-4. I=sum(lambda_i||J_i||)/||J^T lambda|| clipped at 10; participation clipped at 2. All extraction overhead is included in online timing.
- Reward: -0.01 per block update, plus log decrease of full projected-gradient residual (clipped to [-5,5]) every B updates, plus 2 on reaching success. No reward based solely on objective decrease. Reward does NOT directly penalize geometry-computation wall time; it is an update-efficiency training objective, and online cost is evaluated separately.
- Training alternates (n,m)=(8,20) and (16,136), with frequency 2/4 and amplitude 0.1/0.4. ID tests: (16,136), frequency 4, amplitude 0.4. OOD tests: (24,780), frequency 6, amplitude 0.6, stronger coupling. Size, curvature and constraint count change together, so OOD effects cannot be separated.
- Nonconvex coupled objective; convex block-separable quadratic inequalities plus unit disks. Safe projected-gradient block updates, not exact block minimization. All initial points strictly feasible; no Phase I. NOT a benchmark of nonconvex coupled constraints or industrial-scale hard optimization.
- Test budget 600 block updates. Success: infinity norm of projected-gradient mapping <=1e-4, feasibility <=1e-6, evaluated every B updates. Budget exhaustion is not divergence; first-order success is not global or second-order optimality.
- Every method pays for common full certificates; GS additionally projects every candidate block at each decision. RL projects only the chosen block, but regime/shuffled additionally evaluate all geometry features. Timing includes environment construction, inference, projections, certificates and feature extraction, excludes offline training.
- Same maximum episode/step budget across learned variants; actual training steps differ with early success. Training instance seeds are disjoint from test seeds. A new BCD problem generator is reused but not earlier pilot test instances.

## Aggregate held-out results

| Method | Success | Median updates | Median projection calls | Median seconds | Median stationarity |
|---|---:|---:|---:|---:|---:|
| base | 29/36 | 274 | 558 | 0.9933 | 8.45e-05 |
| regime | 30/36 | 274 | 558 | 1.8287 | 8.05e-05 |
| shuffled | 30/36 | 252 | 512 | 1.8318 | 8.02e-05 |
| cyclic | 10/12 | 268 | 546 | 0.9329 | 7.87e-05 |
| random | 10/12 | 280 | 570 | 0.9966 | 6.81e-05 |
| gs | 12/12 | 90 | 1060 | 1.7101 | 7.15e-05 |

## Per-distribution results

| Distribution | Method | Success | Median seconds | Median residual |
|---|---|---:|---:|---:|
| heldout_ID | base | 16/18 | 0.8246 | 8.14e-05 |
| heldout_ID | regime | 17/18 | 1.3740 | 8.05e-05 |
| heldout_ID | shuffled | 17/18 | 1.4583 | 6.75e-05 |
| heldout_ID | cyclic | 6/6 | 0.7058 | 7.8e-05 |
| heldout_ID | random | 6/6 | 0.6405 | 6.81e-05 |
| heldout_ID | gs | 6/6 | 1.0113 | 7.15e-05 |
| heldout_OOD | base | 13/18 | 1.5159 | 8.62e-05 |
| heldout_OOD | regime | 13/18 | 2.9851 | 8.01e-05 |
| heldout_OOD | shuffled | 13/18 | 2.6346 | 9.22e-05 |
| heldout_OOD | cyclic | 4/6 | 1.1611 | 8.34e-05 |
| heldout_OOD | random | 4/6 | 1.0844 | 6.99e-05 |
| heldout_OOD | gs | 6/6 | 2.5349 | 5.56e-05 |

## Training-seed sensitivity

| Variant | Seed | Test success | Actual training updates | Training rollout seconds | Mean reward first/last 20 |
|---|---:|---:|---:|---:|---:|
| base | 71 | 9/12 | 6888 | 23.3 | 10.257 / 9.562 |
| base | 83 | 10/12 | 7136 | 27.5 | 9.573 / 10.060 |
| base | 97 | 10/12 | 6720 | 23.7 | 9.086 / 9.518 |
| regime | 71 | 10/12 | 6880 | 38.6 | 10.358 / 9.534 |
| regime | 83 | 10/12 | 7148 | 40.2 | 9.615 / 10.062 |
| regime | 97 | 10/12 | 6696 | 34.9 | 9.178 / 9.864 |
| shuffled | 71 | 10/12 | 7144 | 44.5 | 9.554 / 9.252 |
| shuffled | 83 | 10/12 | 7136 | 43.7 | 9.794 / 9.587 |
| shuffled | 97 | 10/12 | 6756 | 36.9 | 9.797 / 8.810 |

Training rollout times exclude optimizer-update and serialization overhead. The process was interrupted after 58 test rows; resume_rl.py completed only the 86 missing tasks using identical fixed weights and seeds. Original total elapsed time is unavailable (elapsed.json records null and resume time); all per-episode timings are retained. Timings span two process sessions and should be treated as approximate. Training reward changes across different sampled problems are descriptive, not evidence of generalization.

## Exploratory paired comparison
Final residual ratios include successes and failures, with floor 1e-6. Ratios <1 favor regime. Average logs across training seeds within problem; bootstrap 12 problem instances, 5000 draws. Unadjusted exploratory intervals; heterogeneous ID/OOD mixture and very small sample. Not an efficiency or objective-quality guarantee.
- Regime/base: geometric residual ratio 1.018, exploratory 95% [0.896,1.192]. Median instance-averaged objective gap -1.17327e-10.
- Regime/shuffled: geometric residual ratio 0.976, exploratory 95% [0.751,1.290]. Median instance-averaged objective gap -2.6765e-10.

## Checks and limitations
- Six automated tests cover the descent safeguard, certified projection retry, policy-gradient finite differences, positive-advantage update direction, zero-mask/shuffled features, and deterministic replay including training updates. The BCD test suite separately checks objective/constraint derivatives, projection optimality samples and descent.
- All 144 stored test solutions recomputed for objective, feasibility, projected stationarity and monotonic checkpoint objectives. Independent active-set NNLS ordinary KKT residuals also stored; this is not independent implementation of the projection routine.
- Every method uses the same certified SLSQP projector. A failed 1e-12 solve is retried from zero with ftol=1e-14 and 500 iterations; primal <=1e-7 and projection KKT <=1e-6 remain mandatory. projection_failures records initial failed checks, including recovered retries. An initial incomplete run stopped on such a failure and is archived, not pooled with the completed run.
- Every method applies the same feasibility-preserving segment backtracking if finite-precision projection causes an objective increase; up to 25 halvings, otherwise retain the original point. All objective calls and retry overhead are in online time; backtracks/rejected_steps are recorded. A second incomplete run stopped on a monotonicity check before this safeguard; both incomplete runs are excluded from final statistics.
- Single timing pass, tiny network, 80 episodes and only 3 training seeds. No systematic hyperparameter search. A weak RL result does not rule out richer policies, more training or other tasks; a positive difference does not establish broad regime utility.
- Base already receives cached projected-gradient information, so this is incremental information value beyond a meaningful standard state, not regime versus no constraint information.
- No explicit fallback beyond safe projected steps. No labels derived from solved test optima. No theorem connecting these heuristic weights to optimal scheduling.

## Reproduce
```bash
pip install -r experiments/rl_bcd/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/rl_bcd/benchmark_rl.py
# results directory must not already exist; archive it first
python experiments/rl_bcd/analyze_rl.py
python -m unittest discover -s tests -p test_rl_bcd.py -v
```
