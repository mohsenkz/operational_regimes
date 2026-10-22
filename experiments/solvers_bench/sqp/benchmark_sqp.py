"""Exploratory paired SQP benchmark; no work on import. g(x)<=0.
Convexified exact Lagrangian Hessian, elastic QP, merit backtracking.
Signature uses estimated nonnegative multipliers, not certified KKT regimes.
"""

import json
import sys
from pathlib import Path
from time import perf_counter
import hashlib
import platform
import numpy as np
import scipy
from scipy.optimize import minimize, nnls

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from operational_regimes.regimes.core import (
    features_from_reaction,
    classify_features,
    THETA0,
    kappa_sat_rule,
)


def problem(seed, family):
    rng = np.random.default_rng(seed)
    n, m = 6, 8
    A = rng.normal(size=(m, n))
    if family == "correlated":
        A = rng.normal(size=(1, n)) + 0.015 * A
    elif family == "opposed":
        v = rng.normal(size=n)
        A = np.array([(-1) ** i * v + 0.04 * A[i] for i in range(m)])
    A /= np.linalg.norm(A, axis=1)[:, None]
    curvature = rng.uniform(0.05, 0.3, size=m)
    slack = rng.uniform(0.03, 0.3, size=m)
    target = rng.normal(size=n) * 1.5
    weights = np.geomspace(0.1, 30, n)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    H = Q @ np.diag(weights) @ Q.T
    x0 = rng.normal(size=n) * 0.5

    def evaluate(x):
        d = x - target
        f = 0.5 * d @ H @ d + 0.4 * np.sum(np.sin(3 * x)) + 0.02 * np.sum(x**4)
        grad = H @ d + 1.2 * np.cos(3 * x) + 0.08 * x**3
        hess = H + np.diag(-3.6 * np.sin(3 * x) + 0.24 * x * x)
        g = A @ x + curvature * (x @ x) - slack
        J = A + 2 * curvature[:, None] * x
        return f, grad, hess, g, J

    return evaluate, x0, curvature


def solve(evaluate, x0, curvature, method, strength, seed, maxit=100):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    start = perf_counter()
    qp_count = 0
    feature_time = 0.0
    actions = 0
    labels = {}
    lam = np.zeros(8)
    trace = []
    reason = "budget"
    for it in range(maxit + 1):
        f, grad, H, g, J = evaluate(x)
        active = g >= -1e-4
        lam = np.zeros(len(g))
        if np.any(active):
            lam[active] = nnls(J[active].T, -grad)[0]
        stat = float(np.linalg.norm(grad + J.T @ lam, np.inf))
        feas = float(max(0.0, g.max()))
        comp = float(np.max(np.abs(lam * g)))
        if feas <= 1e-6 and stat <= 1e-4 and comp <= 1e-5:
            reason = "kkt"
            break
        if it == maxit:
            break
        H = H + 2 * np.dot(lam, curvature) * np.eye(len(x))
        eig, V = np.linalg.eigh(H)
        B = (V * np.maximum(eig, 1e-4)) @ V.T
        cond = float(np.linalg.cond(B))
        gate = 0.0
        if method == "conditioning":
            gate = float(cond > 1e4)
        if method in ("signature", "shuffled"):
            t = perf_counter()
            reactions = lam * np.linalg.norm(J, axis=1)
            L, theta, I = features_from_reaction(
                reactions[reactions > 0], np.linalg.norm(J.T @ lam)
            )
            thresholds = dict(THETA0, ksat=kappa_sat_rule(int(np.sum(lam > 0))))
            label = classify_features(
                L / max(np.linalg.norm(grad), 1e-12), theta, I, thresholds
            )
            labels[label] = labels.get(label, 0) + 1
            # Hypothesis: strong reaction cancellation may warrant extra damping.
            gate = float(L > 0 and I > 1.3)
            if method == "shuffled":
                gate = float(rng.random() < 0.5)
            feature_time += perf_counter() - t
        reg = strength * gate * max(1.0, np.linalg.norm(B, 2))
        if reg > 0:
            actions += 1
        B += reg * np.eye(len(x))
        penalty = max(10.0, 2 * float(lam.max()) + 1)
        # Elastic QP avoids deleting infeasible linearizations.
        n = len(x)
        m = len(g)
        z0 = np.r_[np.zeros(n), np.maximum(g, 0) + 1e-8]

        def obj(z):
            return 0.5 * z[:n] @ B @ z[:n] + grad @ z[:n] + penalty * np.sum(z[n:])

        def jac(z):
            return np.r_[B @ z[:n] + grad, np.full(m, penalty)]

        C = np.c_[-J, np.eye(m)]
        qp = minimize(
            obj,
            z0,
            jac=jac,
            method="SLSQP",
            bounds=[(None, None)] * n + [(0, None)] * m,
            constraints=[
                {"type": "ineq", "fun": lambda z: C @ z - g, "jac": lambda z: C}
            ],
            options={"maxiter": 150, "ftol": 1e-10},
        )
        qp_count += 1
        if not qp.success and np.min(C @ qp.x - g) < -1e-6:
            reason = "qp_failure"
            break
        p = qp.x[:n]
        merit = f + penalty * np.maximum(g, 0).sum()
        slope = grad @ p - penalty * np.maximum(g, 0).sum()
        accepted = False
        for backtrack in range(24):
            step = 0.5**backtrack
            trial = x + step * p
            ft, _, _, gt, _ = evaluate(trial)
            if ft + penalty * np.maximum(gt, 0).sum() <= merit + 1e-4 * step * min(
                slope, -1e-12
            ):
                x = trial
                accepted = True
                break
        trace.append(
            dict(
                iteration=it,
                condition=cond,
                regularization=reg,
                feasibility=feas,
                stationarity=stat,
            )
        )
        if not accepted:
            reason = "line_search"
            break
    f, grad, _, g, J = evaluate(x)
    active = g >= -1e-4
    lam = np.zeros(len(g))
    if np.any(active):
        lam[active] = nnls(J[active].T, -grad)[0]
    stat = float(np.linalg.norm(grad + J.T @ lam, np.inf))
    feas = float(max(0.0, g.max()))
    comp = float(np.max(np.abs(lam * g)))
    success = feas <= 1e-6 and stat <= 1e-4 and comp <= 1e-5
    return dict(
        success=bool(success),
        objective=float(f),
        iterations=it,
        qp_count=qp_count,
        time=perf_counter() - start,
        feasibility=feas,
        stationarity=stat,
        complementarity=comp,
        reason=reason,
        actions=actions,
        feature_time=feature_time,
        labels=labels,
        x=x.tolist(),
        trace=trace,
    )


def main():
    out = ROOT / "results/sqp_pilot"
    out.mkdir(parents=True, exist_ok=False)
    methods = ["baseline", "conditioning", "signature", "shuffled"]
    families = ["generic", "correlated", "opposed"]
    cfg = dict(
        calibration_seeds=[11, 22, 33],
        test_seeds=list(range(101, 113)),
        families=families,
        strengths=[0.001, 0.01],
        maxit=100,
        methods=methods,
        caveat="Exploratory custom SQP; estimated multipliers; no global optimality or theorem claim. Shuffled is Bernoulli placebo, not distribution-matched permutation.",
    )
    (out / "metadata.json").write_text(
        json.dumps(
            dict(
                config=cfg,
                numpy=np.__version__,
                scipy=scipy.__version__,
                python=platform.python_version(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )
    rows = []
    selected = {}
    rng = np.random.default_rng(20261022)
    start = perf_counter()
    with (out / "runs.jsonl").open("w", buffering=1) as handle:
        for fi, family in enumerate(families):
            for split, seeds in [
                ("calibration", cfg["calibration_seeds"]),
                ("test", cfg["test_seeds"]),
            ]:
                tasks = []
                for seed in seeds:
                    for method in methods:
                        strengths = (
                            ([0.0] if method == "baseline" else cfg["strengths"])
                            if split == "calibration"
                            else [selected[family + ":" + method]]
                        )
                        tasks.extend((seed, method, s) for s in strengths)
                rng.shuffle(tasks)
                for seed, method, strength in tasks:
                    evaluate, x0, c = problem(seed + fi * 100000, family)
                    result = solve(evaluate, x0, c, method, strength, seed + 999)
                    row = dict(
                        family=family,
                        split=split,
                        seed=seed,
                        method=method,
                        strength=strength,
                        **result,
                    )
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                if split == "calibration":
                    for method in methods:
                        candidates = [0.0] if method == "baseline" else cfg["strengths"]

                        def score(s):
                            rr = [
                                r
                                for r in rows
                                if r["family"] == family
                                and r["split"] == split
                                and r["method"] == method
                                and r["strength"] == s
                            ]
                            return (
                                sum(not r["success"] for r in rr),
                                sum(r["qp_count"] if r["success"] else 101 for r in rr),
                                s,
                            )

                        selected[family + ":" + method] = min(candidates, key=score)
                print(f"{family} {split} done; {perf_counter()-start:.1f}s", flush=True)
    summary = {}
    for method in methods:
        rr = [r for r in rows if r["split"] == "test" and r["method"] == method]
        summary[method] = dict(
            n=len(rr),
            success=sum(r["success"] for r in rr),
            median_time=float(np.median([r["time"] for r in rr])),
            median_qps=float(np.median([r["qp_count"] for r in rr])),
            actions=sum(r["actions"] for r in rr),
        )
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                methods=summary,
                selected_strength=selected,
                wall_time=perf_counter() - start,
            ),
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
