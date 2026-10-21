"""Sum-rate maximisation with QoS floors and a total-power cap, in the w variables only (consistent units)."""

import numpy as np, time
from scipy.optimize import minimize, nnls
from mimo import *
from mimo_v2 import *
from regimes_lib import *

GMIN_DB = 0.0


def parts(Mk):
    return [Mk[k][1:] for k in range(K)]  # (2K x D): rows 2j=Re(h_k^H w_j), 2j+1=Im


def rate_fg(w, Bs):
    f = 0.0
    g = np.zeros_like(w)
    for k in range(K):
        B = Bs[k]
        r = B @ w
        p = r[0::2] ** 2 + r[1::2] ** 2  # |h_k^H w_j|^2
        a = p[k]
        b = p.sum() - a + SIG**2
        s = a / b
        f -= np.log1p(s)
        # d/dw of log(1+a/b)= (da*b - a*db)/(b (a+b)) ; da=2*(Re*BRe+Im*BIm) of j=k ; db=sum_{j!=k} same
        dp = np.zeros_like(w)
        for j in range(K):
            dpj = 2 * (r[2 * j] * B[2 * j] + r[2 * j + 1] * B[2 * j + 1])
            if j == k:
                da = dpj
            else:
                dp += dpj
        g -= (da * b - a * dp) / (b * (a + b))
    return f, g


def solve_rate(H, pdb, dl, gmin_db=GMIN_DB):
    H = H.copy()
    H[0] *= 10 ** (-dl / 20)
    Pmax = 10 ** (pdb / 10)
    s0 = solve_socp(H, gmin_db)
    if s0 is None or np.linalg.norm(s0["w"]) ** 2 > Pmax:
        return None
    Mk, E, t = build(H, gmin_db)
    Bs = parts(Mk)
    cons = [
        {
            "type": "ineq",
            "fun": (
                lambda w, k=k: -(
                    np.linalg.norm(np.r_[Mk[k][1:] @ w, SIG]) - Mk[k][0] @ w
                )
            ),
        }
        for k in range(K)
    ]
    cons.append(
        {"type": "ineq", "fun": lambda w: Pmax - w @ w, "jac": lambda w: -2 * w}
    )
    cons.append({"type": "eq", "fun": lambda w: E @ w, "jac": lambda w: E})
    res = minimize(
        lambda w: rate_fg(w, Bs)[0],
        s0["w"],
        jac=lambda w: rate_fg(w, Bs)[1],
        constraints=cons,
        method="SLSQP",
        options=dict(maxiter=300, ftol=1e-12),
    )
    return res, Mk, E, Bs, Pmax, H


def kkt_rate(res, Mk, E, Bs, Pmax):
    w = res.x
    f, gf = rate_fg(w, Bs)
    G = grads(w, Mk)
    U, S, Vt = np.linalg.svd(E, full_matrices=True)
    Z = Vt[K:].T
    A = np.array([Z.T @ G[k] for k in range(K)] + [Z.T @ (2 * w)]).T
    b = -(Z.T @ gf)
    lam, rn = nnls(A, b, maxiter=20000)
    return lam, rn / np.linalg.norm(b), G, w, gf


def feats_rate(seed, pdb, dl, gref_cache={}):
    rng = np.random.default_rng(seed)
    H0 = channels(rng, 0.0)
    out = solve_rate(H0, pdb, dl)
    # note channels(rng,0) then attenuate inside; reuse the same fading draw
    if out is None:
        return None
    res, Mk, E, Bs, Pmax, H = out
    lam, rn, G, w, gf = kkt_rate(res, Mk, E, Bs, Pmax)
    # reference gradient scale: gradient of f at the reference design (same fading, Delta=0, Pmax large, gmin) = min-power solution
    ref = solve_socp(H0, GMIN_DB)
    _, gref_vec = rate_fg(ref["w"], parts(build(H0, GMIN_DB)[0]))
    gref = np.linalg.norm(gref_vec)
    Gf = np.vstack([G, 2 * w[None, :]])
    act = np.where(lam > 1e-6 * max(lam.max(), 1e-300))[0]
    if len(act) == 0:
        return dict(L=0.0, th=1.0, I=1.0, q=0, res=float(rn), ok=bool(res.success))
    nr = np.linalg.norm(Gf[act], axis=1)
    lt = lam[act] * nr
    F = Gf[act].T @ lam[act]
    L, th, I = features_from_reaction(lt, np.linalg.norm(F))
    return dict(
        L=L / gref,
        th=th,
        I=I,
        q=int(len(act)),
        res=float(rn),
        ok=bool(res.success),
        lamP=float(lam[-1]),
        qos_active=int((lam[:K] > 1e-6 * lam.max()).sum()),
        gref=float(gref),
        rate=float(-res.fun),
    )


if __name__ == "__main__":
    for pdb, dl in ((10, 0), (20, 0), (10, 15), (25, 10)):
        t0 = time.time()
        r = feats_rate(100000, pdb, dl)
        print(pdb, dl, r, round(time.time() - t0, 1), flush=True)
