import numpy as np, time
from src.optimization.mimo import *
import os, json as _json

TH = dict(eta=0.05, ksrd=1.4, ksat=5.5, gI=1.3)
if os.environ.get("ADMM_THETA"):
    TH = _json.load(open(os.environ["ADMM_THETA"]))


def proj_soc(v):
    t = v[0]
    x = v[1:]
    nx = np.linalg.norm(x)
    if nx <= t:
        return v
    if nx <= -t:
        return np.zeros_like(v)
    a = (t + nx) / 2
    return np.r_[a, a * x / nx]


class Prob:
    def __init__(s, H, gam_db, gref):
        s.Mk, s.E, s.t = build(H, gam_db)
        U, S, Vt = np.linalg.svd(s.E, full_matrices=True)
        s.Z = Vt[K:].T
        s.Mp = [np.vstack([Mk, np.zeros((1, Mk.shape[1]))]) for Mk in s.Mk]
        s.A = [Mk @ s.Z for Mk in s.Mp]
        s.G = [a.T @ a for a in s.A]
        s.dk = np.r_[np.zeros(2 * K + 1), SIG]
        s.gref = gref
        s.dkv = s.dk


def run(P, strategy, rho0, maxit=4000, tol=1e-6, check=10, seed=0, logdir=None):
    rng = np.random.default_rng(seed)
    nb = K
    rho = np.full(nb, rho0)
    alpha = 1.0
    z = [np.r_[0, np.zeros(2 * K), SIG] for _ in range(nb)]
    u = [np.zeros(2 * K + 2) for _ in range(nb)]
    theta = np.zeros(P.Z.shape[1])
    I_ = np.eye(len(theta))
    t0 = time.time()
    labs = []

    def factor(rho):
        Mx = 2 * I_ + sum(rho[k] * P.G[k] for k in range(nb))
        return np.linalg.cholesky(Mx)

    Lc = factor(rho)
    for it in range(1, maxit + 1):
        rhs = sum(rho[k] * P.A[k].T @ (z[k] - u[k] - P.dk) for k in range(nb))
        theta = np.linalg.solve(Lc.T, np.linalg.solve(Lc, rhs))
        r2 = 0
        s2 = 0
        znew = []
        unew = []
        for k in range(nb):
            Aw = P.A[k] @ theta + P.dk
            Aw_r = alpha * Aw + (1 - alpha) * z[k]
            zk = proj_soc(Aw_r + u[k])
            uk = u[k] + Aw_r - zk
            r2 += np.sum((Aw - zk) ** 2)
            s2 += np.sum((rho[k] * P.A[k].T @ (zk - z[k])) ** 2)
            znew.append(zk)
            unew.append(uk)
        z, u = znew, unew
        r = np.sqrt(r2)
        s = np.sqrt(s2)
        sc = max(1.0, np.linalg.norm(theta))
        if r <= tol * sc and s <= tol * sc:
            break
        if it % check == 0 and strategy != "fixed":
            old = rho.copy()
            if strategy == "5rho_or":
                rho = np.full(nb, 5 * rho0)
                alpha = 1.6
            elif strategy == "balance":
                f = 2.0 if r > 10 * s else (0.5 if s > 10 * r else 1.0)
                rho = rho * f
            elif strategy in ("regime", "random"):
                w = P.Z @ theta
                lam = np.array([max(-rho[k] * u[k][0], 0.0) for k in range(nb)])
                if strategy == "regime":
                    L, th, I, q = feats(w, lam, P.Mk, P.gref)
                    lab = classify_features(L, th, I, TH, 1e-3) if q > 0 else "unc"
                else:
                    lab = rng.choice(["srd", "int", "sat"])
                labs.append(lab)
                if lab == "sat":
                    rho = np.full(nb, 5 * rho0)
                    alpha = 1.6
                elif lab == "srd":
                    kdom = int(np.argmax(lam * np.linalg.norm(grads(w, P.Mk), axis=1)))
                    rho = np.full(nb, rho0)
                    rho[kdom] = 5 * rho0
                    alpha = 1.0
                elif lab == "int":
                    f = 2.0 if r > 10 * s else (0.5 if s > 10 * r else 1.0)
                    rho = rho * f
                    alpha = 1.0
            if np.any(old != rho):
                for k in range(nb):
                    u[k] = u[k] * old[k] / rho[k]
                Lc = factor(rho)
    return dict(
        it=it,
        time=time.time() - t0,
        conv=bool(it < maxit),
        r=float(r),
        w=P.Z @ theta,
        lam=np.array([max(-rho[k] * u[k][0], 0.0) for k in range(nb)]),
        labs=labs,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(100000)
    H0 = channels(rng, 0.0)
    ref = solve_socp(H0, 0.0)
    gref = 2 * np.linalg.norm(ref["w"])
    H = H0.copy()
    H[0] *= 10 ** (-10 / 20)
    sol = solve_socp(H, 10.0)
    P = Prob(H, 10.0, gref)
    for st in ("fixed", "5rho_or", "balance", "regime", "random"):
        r = run(P, st, 1.0)
        print(
            st,
            r["it"],
            round(r["time"], 2),
            r["conv"],
            "w err %.1e" % np.linalg.norm(r["w"] - sol["w"]),
            "lam err %.1e" % np.abs(r["lam"] - sol["lam"]).max(),
            r["labs"][-3:],
        )
