"""Paired ADMM pilot: calibration-only tuning; held-out cases; JSONL audit trail.

Run from any directory with --config and --output. Dependencies: numpy, scipy,
cvxpy, clarabel, threadpoolctl. No plotting dependency and no work on import.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import scipy
    import cvxpy
    import clarabel
    from threadpoolctl import threadpool_limits, threadpool_info
    from operational_regimes.optimization.admm_mimo import Prob, channels, run, solve_socp

    cfg = json.loads(args.config.read_text())
    if set(cfg["calibration_seeds"]) & set(cfg["test_seeds"]):
        raise ValueError("Calibration and test seeds must be disjoint")
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    methods = cfg["methods"]
    if "shuffled" in methods and "signature" not in methods:
        raise ValueError("Shuffled pool must come from calibration signature traces")
    source_paths = [Path(__file__).resolve(), *sorted((ROOT / "src").rglob("*.py"))]
    provenance = dict(config=cfg, python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, cvxpy=cvxpy.__version__,
                      clarabel=getattr(clarabel, "__version__", "unknown"),
                      platform=platform.platform(),
                      repository_commit=subprocess.check_output(
                          ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in source_paths},
                      calibration_objective="minimize (failed common certificates, mean capped iterations, rho)",
                      accuracy="ADMM r,s,stationarity thresholds plus relative objective error <=1e-4, cone violation <=1e-5, phase equality <=1e-7",
                      time_definition="run() includes initial/update factorization, controller, feature extraction, residual tests; excludes shared Prob construction and reference solves; extra setup timings retained separately",
                      caveat="Small exploratory pilot; no out-of-distribution validation or nonconvex convergence claim")
    (outdir / "metadata.json").write_text(json.dumps(provenance, indent=2))
    raw_file = (outdir / "runs.jsonl").open("w", buffering=1)
    cases_file = (outdir / "cases.jsonl").open("w", buffering=1)
    rows = []
    pools = {}
    selected = {}
    rng = np.random.default_rng(cfg["order_seed"])
    all_seeds = cfg["calibration_seeds"] + cfg["test_seeds"]
    started = perf_counter()

    with threadpool_limits(limits=cfg["threads"]):
        provenance["threadpools"] = threadpool_info()
        (outdir / "metadata.json").write_text(json.dumps(provenance, indent=2))
        for ci, (gam_db, delta_db) in enumerate(cfg["conditions"]):
            cases = {}
            oracle_lam = {}
            for seed in all_seeds:
                channel_rng = np.random.default_rng(seed + ci * 100000)
                h0 = channels(channel_rng, 0.)
                h = h0.copy()
                h[0] *= 10 ** (-delta_db / 20)
                start = perf_counter()
                base = solve_socp(h0, 0.)
                gref_time = perf_counter() - start
                if base is None:
                    raise RuntimeError(f"gref reference failure: condition={ci}, seed={seed}")
                gref = 2 * np.linalg.norm(base["w"])
                start = perf_counter()
                oracle = solve_socp(h, gam_db)
                reference_time = perf_counter() - start
                if oracle is None:
                    raise RuntimeError(f"Evaluation reference failure: condition={ci}, seed={seed}")
                start = perf_counter()
                problem = Prob(h, gam_db, gref)
                build_time = perf_counter() - start
                ref_w = oracle["w"]
                oracle_lam[seed] = oracle["lam"]
                cone_values = [mk @ ref_w for mk in problem.Mk]
                ref_feas = max(0., max(np.linalg.norm(np.r_[v[1:], 1.]) - v[0]
                                      for v in cone_values))
                if ref_feas > 1e-7 or np.linalg.norm(problem.E @ ref_w) > 1e-7:
                    raise RuntimeError(f"Reference accuracy failure: condition={ci}, seed={seed}")
                details = dict(condition=ci, gam_db=gam_db, delta_db=delta_db, seed=seed,
                               channel_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
                               gref=float(gref), gref_time=gref_time,
                               problem_build_time=build_time, reference_time=reference_time,
                               reference_objective=float(ref_w @ ref_w), reference_feasibility=ref_feas)
                cases_file.write(json.dumps(details) + "\n")
                cases[seed] = problem, ref_w, details
            pool = []

            def execute(seed, method, rho, split):
                definition = methods[method]
                problem, ref_w, details = cases[seed]
                kwargs = dict(cfg["solver"])
                kwargs.update(definition.get("settings", {}))
                if definition["strategy"] == "shuffled":
                    kwargs["signature_pool"] = pool
                result = run(problem, definition["strategy"], rho,
                             seed=seed + ci * 100000 + 700000, **kwargs)
                features = result.pop("features")
                if split == "calibration" and definition["strategy"] == "signature":
                    pool.extend(dict(x=entry["x"], lab=entry["lab"]) for entry in features)
                ref_obj = details["reference_objective"]
                result["objective_relative_error"] = abs(result["objective"] - ref_obj) / max(ref_obj, 1e-30)
                result["w_relative_error"] = float(np.linalg.norm(result.pop("w") - ref_w) / np.linalg.norm(ref_w))
                result["lam_relative_error"] = float(np.linalg.norm(result.pop("lam") - oracle_lam[seed])
                                                     / max(np.linalg.norm(oracle_lam[seed]), 1e-30))
                result["labels"] = dict(Counter(result.pop("labs")))
                result.pop("history")
                result["success"] = bool(result["conv"] and result["objective_relative_error"] <= 1e-4
                                         and result["feasibility"] <= 1e-5 and result["equality"] <= 1e-7)
                result["online_with_build_time"] = result["time"] + details["problem_build_time"]
                needs_gref = definition["strategy"] in ("signature", "label", "shuffled", "regime")
                result["uncached_setup_time"] = result["online_with_build_time"] + (details["gref_time"] if needs_gref else 0.)
                row = dict(split=split, condition=ci, seed=seed, method=method, rho0=rho, **result)
                raw_file.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                return row


            calibration_tasks = [(seed, method, rho)
                                 for seed in cfg["calibration_seeds"]
                                 for method in methods if methods[method]["strategy"] != "shuffled"
                                 for rho in cfg["rho_grid"]]
            rng.shuffle(calibration_tasks)
            for ti, (seed, method, rho) in enumerate(calibration_tasks):
                result = execute(seed, method, rho, "calibration")
                if (ti + 1) % 20 == 0:
                    print(f"condition {ci}: calibration {ti+1}/{len(calibration_tasks)}; elapsed {perf_counter()-started:.1f}s", flush=True)
            if "shuffled" in methods:
                if not pool:
                    raise RuntimeError("No calibration signatures available")
                shuffled_tasks = [(seed, "shuffled", rho) for seed in cfg["calibration_seeds"] for rho in cfg["rho_grid"]]
                rng.shuffle(shuffled_tasks)
                for seed, method, rho in shuffled_tasks:
                    execute(seed, method, rho, "calibration")
            pools[str(ci)] = pool
            (outdir / "calibration_signature_pools.json").write_text(json.dumps(pools))
            for method in methods:
                scores = []
                for rho in cfg["rho_grid"]:
                    rr = [r for r in rows if r["split"] == "calibration" and r["condition"] == ci
                          and r["method"] == method and r["rho0"] == rho]
                    failures = sum(not r["success"] for r in rr)
                    mean_it = float(np.mean([r["it"] if r["success"] else cfg["solver"]["maxit"] for r in rr]))
                    scores.append((failures, mean_it, rho))
                selected[f"{ci}:{method}"] = min(scores)[2]
            (outdir / "selected_rho.json").write_text(json.dumps(selected, indent=2))
            test_tasks = [(seed, method) for seed in cfg["test_seeds"] for method in methods]
            rng.shuffle(test_tasks)
            for ti, (seed, method) in enumerate(test_tasks):
                execute(seed, method, selected[f"{ci}:{method}"], "test")
                if (ti + 1) % 20 == 0:
                    print(f"condition {ci}: test {ti+1}/{len(test_tasks)}; elapsed {perf_counter()-started:.1f}s", flush=True)
    raw_file.close()
    cases_file.close()
    provenance["wall_time"] = perf_counter() - started
    provenance["completed"] = True
    (outdir / "metadata.json").write_text(json.dumps(provenance, indent=2))
    summarize(rows, methods, outdir, cfg)
    print(f"Completed: {outdir}; {provenance['wall_time']:.1f}s", flush=True)


def summarize(rows, methods, outdir, cfg):
    import numpy as np
    test = [r for r in rows if r["split"] == "test"]
    summary = {}
    for method in methods:
        rr = [r for r in test if r["method"] == method]
        success = [r for r in rr if r["success"]]
        summary[method] = dict(n=len(rr), success=len(success),
            median_capped_iterations=float(np.median([r["it"] for r in rr])),
            median_success_iterations=float(np.median([r["it"] for r in success])) if success else None,
            median_online_time=float(np.median([r["time"] for r in rr])),
            mean_online_time=float(np.mean([r["time"] for r in rr])),
            median_uncached_time=float(np.median([r["uncached_setup_time"] for r in rr])),
            median_factor_count=float(np.median([r["factor_count"] for r in rr])),
            mean_feature_time=float(np.mean([r["feature_time"] for r in rr])),
            total_signature_actions=sum(r["signature_actions"] for r in rr),
            max_feasibility=float(max(r["feasibility"] for r in rr)))
    comparisons = []
    bootstrap_rng = np.random.default_rng(987654)
    for candidate in ["history", "signature", "label", "shuffled"]:
        for baseline in ["balance_normalized", "history"]:
            if candidate == baseline or candidate not in methods or baseline not in methods:
                continue
            cc = {(r["condition"], r["seed"]): r for r in test if r["method"] == candidate}
            bb = {(r["condition"], r["seed"]): r for r in test if r["method"] == baseline}
            keys = sorted(cc.keys() & bb.keys())
            both = [k for k in keys if cc[k]["success"] and bb[k]["success"]]
            ratios = np.array([cc[k]["time"] / bb[k]["time"] for k in both])
            ci = None
            if len(ratios):
                # Stratified by condition; uncertainty concerns the paired seed
                # samples in these fixed conditions, not unseen distributions.
                draws = []
                for _ in range(2000):
                    sampled = []
                    for condition in sorted({k[0] for k in both}):
                        kk = [k for k in both if k[0] == condition]
                        selected_keys = [kk[i] for i in bootstrap_rng.integers(len(kk), size=len(kk))]
                        sampled.extend(np.log(cc[k]["time"] / bb[k]["time"]) for k in selected_keys)
                    draws.append(np.exp(np.mean(sampled)))
                ci = np.quantile(draws, [.025, .975]).tolist()
            comparisons.append(dict(candidate=candidate, baseline=baseline, paired_n=len(keys), both_success=len(both),
                candidate_only_success=sum(cc[k]["success"] and not bb[k]["success"] for k in keys),
                baseline_only_success=sum(bb[k]["success"] and not cc[k]["success"] for k in keys),
                paired_geomean_time_ratio=float(np.exp(np.mean(np.log(ratios)))) if len(ratios) else None,
                conditional_bootstrap_95=ci,
                iteration_wins=sum(cc[k]["it"] < bb[k]["it"] for k in both),
                iteration_ties=sum(cc[k]["it"] == bb[k]["it"] for k in both),
                iteration_losses=sum(cc[k]["it"] > bb[k]["it"] for k in both)))
    result = dict(methods=summary, paired_comparisons=comparisons)
    (outdir / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
