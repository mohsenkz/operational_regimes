# Meta-RL BCD: small exploratory Reptile pilot

## Predeclared protocol
- First-order Reptile over ordinary REINFORCE+linear critic inner updates. This is meta-learning of an initialization, not MAML, a recurrent memory agent, or a theorem about regime usefulness.
- 2 training seeds x 3 feature variants x 2 training algorithms x 12 tasks x 2 episodes = 288 training episodes, each capped at 96 updates. Reptile inner Adam resets at every task; pooled ordinary RL uses persistent Adam. Identical initial weights, task/instance schedules, architectures and per-variant interaction budgets; early stopping can change realized counts.
- Reptile interpolation linearly decreases from 0.25 to 0.05; both actor and critic are meta-initialized. Each task has two independent problem instances. Pooled training receives exactly the same scheduled instances. Final fixed-budget checkpoints; no test or validation tuning.
- Three training families, two entirely held-out parameter families (unseen_D and unseen_E). Task coupling/amplitude jitter is sampled once and shared by its support and query instances. Family IDs are disjoint; all instances use the same synthetic generator. This is parameter-family holdout, NOT evidence of transfer to a new generator or application.
- 6 independent held-out tasks, 3 per family, one query instance per task. 2 training seeds are repeated measurements, NOT 12 independent tasks. 180 learned-policy query runs + 18 cyclic/random/GS = 198. Very small pilot with only two seeds.
- Five learned-policy conditions: Reptile frozen/adapted, pooled ordinary RL frozen/adapted, and random initialization adapted. Every adapted condition gets exactly two support episodes capped at 96 updates; fresh Adam, same support instances and action seed schedules. Support instances differ from query instances. Query weights are hashed before/after to assert no query learning.
- Query budget 384 updates for every method; unlike the earlier ordinary RL experiment (600), so cross-pilot success numbers are not directly comparable. Success uses projected stationarity <=1e-4 and feasibility <=1e-6. All 198 stored solutions are independently recomputed; ordinary active-set NNLS KKT residuals stored separately. Projection itself reuses the certified implementation.
- Feature variants base/regime/shuffled and 10x16 shared scorer are unchanged. Regime inputs are continuous local NNLS geometry features, not regime class labels. Base already has cached projected-gradient features. Shuffling also changes the random stream, so it is not a perfectly coupled placebo.
- Online time includes support rollouts, optimizer updates, initialization/cloning and query execution. Query time includes feature extraction, certificates, projections and inference. Offline training is reported separately. Costs are NOT amortized over multiple queries; one new task pays for its two support episodes. Frozen methods pay no support cost. Single timing measurement, no hardware-general latency claim.
- Safe projected block steps, identical numerical guards for all methods, convex block-separable constraints and nonconvex coupled objectives. First-order success is not global or second-order optimality. Budget exhaustion does not prove divergence. The reward does not directly penalize feature-computation wall time.

## All held-out tasks

| Method / features | Success | Median residual | Median updates | Query sec | Total online sec |
|---|---:|---:|---:|---:|---:|
| reptile_frozen/base | 10/12 | 6.5e-05 | 258 | 0.671 | 0.671 |
| reptile_frozen/regime | 10/12 | 5.59e-05 | 246 | 1.363 | 1.363 |
| reptile_frozen/shuffled | 9/12 | 6.86e-05 | 240 | 1.267 | 1.267 |
| reptile_adapted/base | 9/12 | 6.17e-05 | 258 | 0.668 | 1.369 |
| reptile_adapted/regime | 10/12 | 4.43e-05 | 258 | 1.424 | 2.440 |
| reptile_adapted/shuffled | 9/12 | 6.7e-05 | 240 | 1.113 | 2.439 |
| pooled_frozen/base | 9/12 | 6.18e-05 | 246 | 0.624 | 0.624 |
| pooled_frozen/regime | 9/12 | 6.65e-05 | 240 | 1.448 | 1.449 |
| pooled_frozen/shuffled | 9/12 | 8.76e-05 | 240 | 1.166 | 1.167 |
| pooled_adapted/base | 9/12 | 6.18e-05 | 246 | 0.652 | 1.302 |
| pooled_adapted/regime | 9/12 | 6.79e-05 | 240 | 1.355 | 2.388 |
| pooled_adapted/shuffled | 9/12 | 8.58e-05 | 240 | 1.195 | 2.441 |
| random_adapted/base | 9/12 | 6.17e-05 | 258 | 0.657 | 1.315 |
| random_adapted/regime | 10/12 | 4.43e-05 | 258 | 1.273 | 2.627 |
| random_adapted/shuffled | 9/12 | 6.7e-05 | 240 | 1.195 | 2.446 |
| cyclic | 5/6 | 6.92e-05 | 210 | 0.518 | 0.518 |
| random | 5/6 | 8.51e-05 | 262 | 0.575 | 0.575 |
| gs | 6/6 | 5.43e-05 | 108 | 1.403 | 1.403 |

## By held-out family

| Family | Method / features | Success | Median residual | Total online sec |
|---|---|---:|---:|---:|
| unseen_D | reptile_frozen/base | 6/6 | 6.57e-05 | 0.696 |
| unseen_D | reptile_frozen/regime | 6/6 | 5.22e-05 | 1.235 |
| unseen_D | reptile_frozen/shuffled | 5/6 | 5.43e-05 | 0.917 |
| unseen_D | reptile_adapted/base | 5/6 | 6.57e-05 | 1.502 |
| unseen_D | reptile_adapted/regime | 6/6 | 4.34e-05 | 2.511 |
| unseen_D | reptile_adapted/shuffled | 5/6 | 5.43e-05 | 2.517 |
| unseen_D | pooled_frozen/base | 5/6 | 6.77e-05 | 0.716 |
| unseen_D | pooled_frozen/regime | 5/6 | 5.31e-05 | 1.337 |
| unseen_D | pooled_frozen/shuffled | 5/6 | 8.14e-05 | 0.973 |
| unseen_D | pooled_adapted/base | 5/6 | 5.69e-05 | 1.541 |
| unseen_D | pooled_adapted/regime | 5/6 | 6.47e-05 | 2.385 |
| unseen_D | pooled_adapted/shuffled | 5/6 | 7.19e-05 | 2.598 |
| unseen_D | random_adapted/base | 5/6 | 6.57e-05 | 1.495 |
| unseen_D | random_adapted/regime | 6/6 | 4.34e-05 | 2.757 |
| unseen_D | random_adapted/shuffled | 5/6 | 5.43e-05 | 2.368 |
| unseen_D | cyclic | 3/3 | 5.61e-05 | 0.609 |
| unseen_D | random | 2/3 | 9.85e-05 | 0.704 |
| unseen_D | gs | 3/3 | 6.16e-05 | 0.968 |
| unseen_E | reptile_frozen/base | 4/6 | 6.39e-05 | 0.653 |
| unseen_E | reptile_frozen/regime | 4/6 | 5.62e-05 | 1.363 |
| unseen_E | reptile_frozen/shuffled | 4/6 | 7.68e-05 | 1.350 |
| unseen_E | reptile_adapted/base | 4/6 | 5.28e-05 | 1.162 |
| unseen_E | reptile_adapted/regime | 4/6 | 5.27e-05 | 2.414 |
| unseen_E | reptile_adapted/shuffled | 4/6 | 7.62e-05 | 2.164 |
| unseen_E | pooled_frozen/base | 4/6 | 5.6e-05 | 0.570 |
| unseen_E | pooled_frozen/regime | 4/6 | 8.11e-05 | 1.472 |
| unseen_E | pooled_frozen/shuffled | 4/6 | 9.16e-05 | 1.259 |
| unseen_E | pooled_adapted/base | 4/6 | 6.39e-05 | 1.296 |
| unseen_E | pooled_adapted/regime | 4/6 | 7.92e-05 | 2.466 |
| unseen_E | pooled_adapted/shuffled | 4/6 | 9.19e-05 | 2.324 |
| unseen_E | random_adapted/base | 4/6 | 5.28e-05 | 1.225 |
| unseen_E | random_adapted/regime | 4/6 | 5.27e-05 | 2.370 |
| unseen_E | random_adapted/shuffled | 4/6 | 7.62e-05 | 2.446 |
| unseen_E | cyclic | 2/3 | 7.83e-05 | 0.450 |
| unseen_E | random | 3/3 | 4.67e-05 | 0.535 |
| unseen_E | gs | 3/3 | 4.71e-05 | 1.438 |

## Exploratory paired contrasts
Residual and total-time geometric ratios below 1 favor the numerator. Average log-ratios across training seeds within task, then bootstrap tasks within each held-out family (5000 draws). Six task clusters only; unadjusted exploratory 95% intervals, not confirmatory inference. Residual floor 1e-6; all successes and budget failures included. Residuals are affected by stopping thresholds and do not directly measure solution quality.
- reptile_adapted/pooled_adapted:base: residual 1.088 [0.689, 2.039]; online time 0.897 [0.785, 0.997]; mean success difference 0.000.
- reptile_adapted/reptile_frozen:base: residual 0.971 [0.901, 1.016]; online time 1.993 [1.813, 2.191]; mean success difference -0.083.
- reptile_adapted/random_adapted:base: residual 1.318 [1.000, 2.289]; online time 0.984 [0.951, 1.027]; mean success difference 0.000.
- pooled_adapted/pooled_frozen:base: residual 0.971 [0.924, 1.021]; online time 2.234 [1.807, 2.860]; mean success difference 0.000.
- reptile_adapted/pooled_adapted:regime: residual 0.711 [0.599, 0.911]; online time 1.017 [0.956, 1.075]; mean success difference 0.083.
- reptile_adapted/reptile_frozen:regime: residual 0.880 [0.684, 1.040]; online time 1.887 [1.702, 2.091]; mean success difference 0.000.
- reptile_adapted/random_adapted:regime: residual 0.972 [0.946, 0.994]; online time 1.000 [0.951, 1.033]; mean success difference 0.000.
- pooled_adapted/pooled_frozen:regime: residual 0.947 [0.575, 1.490]; online time 1.650 [1.410, 1.932]; mean success difference 0.000.
- reptile_adapted/pooled_adapted:shuffled: residual 0.792 [0.683, 0.920]; online time 0.958 [0.933, 0.984]; mean success difference 0.000.
- reptile_adapted/reptile_frozen:shuffled: residual 0.923 [0.831, 1.083]; online time 1.990 [1.571, 2.327]; mean success difference 0.000.
- reptile_adapted/random_adapted:shuffled: residual 0.936 [0.821, 1.001]; online time 0.982 [0.847, 1.138]; mean success difference 0.000.
- pooled_adapted/pooled_frozen:shuffled: residual 0.912 [0.787, 0.994]; online time 2.133 [1.807, 2.377]; mean success difference 0.000.
- regime/base:reptile_frozen: residual 0.786 [0.432, 1.327]; online time 1.952 [1.861, 2.027]; mean success difference 0.000.
- regime/shuffled:reptile_frozen: residual 1.577 [0.687, 5.364]; online time 1.053 [0.928, 1.213]; mean success difference 0.083.
- regime/base:reptile_adapted: residual 0.712 [0.449, 0.923]; online time 1.848 [1.749, 1.953]; mean success difference 0.083.
- regime/shuffled:reptile_adapted: residual 1.502 [0.614, 6.623]; online time 0.999 [0.937, 1.057]; mean success difference 0.083.
- regime/base:pooled_frozen: residual 1.119 [0.704, 1.778]; online time 2.207 [1.991, 2.469]; mean success difference 0.000.
- regime/shuffled:pooled_frozen: residual 1.614 [0.461, 10.498]; online time 1.216 [1.063, 1.390]; mean success difference 0.000.
- regime/base:pooled_adapted: residual 1.091 [0.984, 1.233]; online time 1.630 [1.422, 1.795]; mean success difference 0.000.
- regime/shuffled:pooled_adapted: residual 1.676 [0.800, 6.707]; online time 0.940 [0.851, 1.042]; mean success difference 0.000.
- regime/base:random_adapted: residual 0.966 [0.889, 1.063]; online time 1.818 [1.740, 1.882]; mean success difference 0.083.
- regime/shuffled:random_adapted: residual 1.448 [0.631, 5.716]; online time 0.981 [0.856, 1.130]; mean success difference 0.083.

## Training and checks
- Offline training wall time (including optimizer updates/model save): 111.08 s total across 12 models. Per-model times: training_times.json. Full run: elapsed.json.
- Training-seed sensitivity, projection calls, feature/adaptation times and paired objective gaps: summary.json; complete support/query records: test.jsonl; all configurations and source hashes: metadata.json.
- Verification: {"verified": 198, "max_feasibility": 4.885869486770389e-12, "backtracks": 75, "rejected_steps": 0, "projection_retries": 0}
- Four meta-specific automated tests cover independent clone/reset, exact Reptile interpolation, inner learning/query freeze, task reproducibility/family holdout. Existing BCD/RL tests cover derivatives, projection certification, gradients and numerical safeguards.
- Equal interaction budgets do not imply equal optimizer behavior or converged training. Tiny training budgets, two seeds, one query per task, and no hyperparameter search limit all conclusions. Weak results cannot reject meta-RL in general. Positive differences cannot establish general regime utility.

## Interpretation
- Reptile adapted/regime succeeds on 10/12 repeated-seed evaluations versus 9/12 for pooled adapted/regime, but random-initialization adapted/regime also reaches 10/12 and the same median residual. This is not a compelling meta-initialization advantage.
- Local exploratory residual improvements are present: Reptile adapted/regime versus pooled adapted/regime has ratio 0.711 [0.599, 0.911], and versus Reptile adapted/base 0.712 [0.449, 0.923]. However, the comparison against shuffled is inconclusive, random-adapted performance is nearly identical, and many residuals are already below the stopping threshold.
- Adaptation leaves regime success unchanged versus frozen Reptile (10/12 each), while median online time rises from 1.363 to 2.440 seconds. GS succeeds on all six query instances, with median 108 updates and 1.403 seconds; its results are not repeated over training seeds.
- Overall: limited local residual gains, but no convincing cost-adjusted advantage specifically attributable to meta-learning or regime information in this small fixed-budget pilot. This neither rejects meta-RL generally nor evaluates classification.

## Reproduce
```bash
pip install -r experiments/rl_bcd/requirements.txt
# Archive results/meta_rl_bcd_pilot first for a fresh run.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/meta_rl_bcd/benchmark_meta.py
python experiments/meta_rl_bcd/analyze_meta.py
python -m unittest discover -s tests -p "test_*bcd.py" -v
# If interrupted after schedule.json exists, finish only missing evaluations:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/meta_rl_bcd/benchmark_meta.py --resume
```
