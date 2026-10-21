"""Max-min SINR beamforming (power budget active) + min-power, features for regime classification."""

import numpy as np, cvxpy as cp
from scipy.optimize import nnls
from mimo import *
from regimes_lib import *


# ---------- min-power (as before): features with g_ref from the 0 dB symmetric reference
def feats_minpower(seed, gdb, dl):
    rng = np.random.default_rng(seed)
    H0 = channels(rng, 0.0)
    ref = solve_socp(H0, 0.0)
    if ref is None:
        return None
    gref = 2 * np.linalg.norm(ref["w"])
    H = H0.copy()
    H[0] *= 10 ** (-dl / 20)
    s = solve_socp(H, gdb)
    if s is None:
        return None
    L, th, I, q = feats(s["w"], s["lam"], s["Mk"], gref)
    return dict(
        L=L,
        th=th,
        I=I,
        q=q,
        sc=bool(s["lam"].min() > 1e-6 * s["lam"].max()),
        qdim=D - K,
    )


# ---------- max-min: max t s.t. SINR_k>=t, sum||w||^2<=Pmax
def minpower_at(H, t_db):
    s = solve_socp(H, t_db)
    return s


def maxmin(H, Pmax, tol=1e-9):
    lo, hi = -40.0, 60.0  # dB
    s_lo = minpower_at(H, lo)
    if s_lo is None or np.linalg.norm(s_lo["w"]) ** 2 > Pmax:
        return None
    best = s_lo
    for _ in range(40):
        mid = (lo + hi) / 2
        s = minpower_at(H, mid)
        if s is not None and np.linalg.norm(s["w"]) ** 2 <= Pmax:
            lo = mid
            best = s
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return lo, best


def kkt_maxmin(H, t_db, sol):
    """multipliers of max t s.t. g_k(w,t)<=0, g_0=||w||^2-Pmax<=0, Im(h^H w)=0 at (w*,t*): solve stationarity by NNLS."""
    t = 10 ** (t_db / 10)
    w = sol["w"]
    Mk = sol["Mk"]
    E = sol["E"]
    G = []
    dt = []
    st = np.sqrt(1 + 1 / t)
    for k in range(K):
        c = Mk[k][0] / st  # Re row (Mk[k][0]=s*Re)
        y = np.r_[Mk[k][1:] @ w, SIG]
        ny = np.linalg.norm(y)
        Gk = Mk[k][1:].T @ (Mk[k][1:] @ w) / ny - Mk[k][0]
        G.append(Gk)
        dt.append(
            (c @ w) / (2 * st * t**2)
        )  # d g/dt = -(ds/dt) c.w,  ds/dt=-1/(2 s t^2)
    G = np.array(G)
    dt = np.array(dt)
    U, S, Vt = np.linalg.svd(E, full_matrices=True)
    Z = Vt[K:].T
    cols = [np.r_[Z.T @ G[k], dt[k]] for k in range(K)] + [np.r_[Z.T @ (2 * w), 0.0]]
    A = np.array(cols).T
    b = np.r_[np.zeros(Z.shape[1]), 1.0]
    lam, res = nnls(A, b, maxiter=10000)
    return lam, res, G, dt, w


def feats_maxmin(seed, pdb, dl):
    rng = np.random.default_rng(seed)
    H = channels(rng, dl)
    Pmax = 10 ** (pdb / 10)
    out = maxmin(H, Pmax)
    if out is None:
        return None
    t_db, sol = out
    lam, res, G, dt, w = kkt_maxmin(H, t_db, sol)
    Gf = np.vstack(
        [np.c_[G, dt], np.r_[2 * w, 0.0][None, :]]
    )  # K SINR constraints + power constraint, full (w,t) gradients
    act = np.where(lam > 1e-6 * max(lam.max(), 1e-300))[0]
    nr = np.linalg.norm(Gf[act], axis=1)
    lt = lam[act] * nr
    F = Gf[act].T @ lam[act]
    L, th, I = features_from_reaction(lt, np.linalg.norm(F))  # g_ref=||grad f||=1
    return dict(
        L=L,
        th=th,
        I=I,
        q=int(len(act)),
        res=float(res),
        t_db=float(t_db),
        lam0=float(lam[-1]),
        share0=float(lt[-1] / lt.sum()) if (len(act) and act[-1] == K) else 0.0,
        sc=bool(lam.min() > 1e-6 * lam.max()),
    )


if __name__ == "__main__":
    import time

    t0 = time.time()
    r = feats_maxmin(100000, 10.0, 0.0)
    print(r, time.time() - t0)
    r = feats_maxmin(100001, 10.0, 15.0)
    print(r)
