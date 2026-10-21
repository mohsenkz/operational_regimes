import numpy as np, cvxpy as cp, sys, json
from scipy.optimize import brentq

rng_master = np.random.default_rng(0)
n = 100
ETA, KS, KT, GI = 0.05, 1.4, 60.5, 1.3
EPS = 1e-3


def classify(lam, grads, gref, eps=EPS, tau_rel=1e-6):
    # lam: multipliers (m,), grads: (m,n) gradients at x; Ahat by relative threshold
    if lam.size == 0 or lam.max() <= 0:
        A = np.array([], int)
    else:
        A = np.where(lam > tau_rel * lam.max())[0]
    if len(A) == 0:
        lt = np.zeros(0)
    else:
        lt = lam[A] * np.linalg.norm(grads[A], axis=1)
    Lraw = lt.sum()
    Lh = Lraw / gref
    if Lh < ETA - eps:
        return "unc", (Lh, 1, 1)
    if abs(Lh - ETA) <= eps:
        return "hyb", (Lh, None, None)
    s = (lt**2).sum()
    th = Lraw**2 / s
    if abs(th - KS) <= eps:
        return "hyb", (Lh, th, None)
    if th < KS - eps:
        return "srd", (Lh, th, None)
    if th < KT - eps:
        return "int", (Lh, th, None)
    F = (lam[A, None] * grads[A]).sum(0)
    I = np.sqrt(s) / np.linalg.norm(F) if np.linalg.norm(F) > 0 else np.inf
    if I > GI + EPS:
        return "int", (Lh, th, I)
    if th > KT + eps and I < GI - eps:
        return "sat", (Lh, th, I)
    return "hyb", (Lh, th, I)


def feats(lt, N):
    L = lt.sum()
    th = L**2 / (lt**2).sum()
    I = np.sqrt((lt**2).sum() / (lt @ N @ lt))
    return L, th, I


def rows_with_gram(N, rng):
    k = N.shape[0]
    Lc = np.linalg.cholesky(N)
    A = np.zeros((k, n))
    A[:, :k] = Lc
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return A @ Q


def solve_qp(c, A, b, extra=None):
    x = cp.Variable(n)
    cons = [A @ x <= b]
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - c)), cons)
    try:
        prob.solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
        )
    except Exception:
        prob.solve(solver=cp.CLARABEL)
    return x.value, np.maximum(cons[0].dual_value, 0)


def targeted(N, lt, rng, extra_inactive=20):
    k = len(lt)
    c = rng.standard_normal(n)
    c /= np.linalg.norm(c)  # g_ref=||c||=1 with x_ref=0
    A = rows_with_gram(N, rng)
    b = A @ c - N @ lt
    # extra inactive constraints
    E = rng.standard_normal((extra_inactive, n))
    E /= np.linalg.norm(E, axis=1)[:, None]
    xs = c - A.T @ lt
    be = E @ xs + 1.0
    AA = np.vstack([A, E])
    bb = np.concatenate([b, be])
    x, lam = solve_qp(c, AA, bb)
    return classify(lam, AA, 1.0), feats(lt, N), lam


def fam1(rng):
    M = rng.standard_normal((n, n))
    Qm, _ = np.linalg.qr(M)
    ev = rng.uniform(1, 10, n)
    Q = Qm @ np.diag(ev) @ Qm.T
    c = rng.standard_normal(n)
    x0 = -np.linalg.solve(Q, c)
    B = np.abs(x0).max() / (1 - 0.2) * 1.0001
    x = cp.Variable(n)
    G = np.vstack([np.eye(n), -np.eye(n)])
    cons = [G @ x <= B * np.ones(2 * n)]
    cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(Q)) + c @ x), cons).solve(
        solver=cp.CLARABEL
    )
    lam = np.maximum(cons[0].dual_value, 0)
    gref = np.linalg.norm(c)  # x_ref=0
    return classify(lam, G, gref)


def fam2(rng):
    # srd family: sign constraints x>=0 must stay inactive (checked a posteriori; redraw otherwise)
    e0 = 1e-3
    rej = 0
    while True:
        c = rng.uniform(0.5, 1.5, n)
        c /= np.linalg.norm(c)
        a1 = c * (1 + 0.05 * rng.uniform(-1, 1, n))
        b1 = 0.1 * (a1 @ c) / e0
        lam1 = (a1 @ c - e0 * b1) / (a1 @ a1)
        xs = (c - lam1 * a1) / e0
        m = 20
        Aj = rng.uniform(0.5, 1.5, (m - 1, n))
        bj = 10 * np.maximum(Aj @ xs, 1)
        A = np.vstack([a1, Aj, -np.eye(n)])
        b = np.concatenate([[b1], bj, np.zeros(n)])
        x = cp.Variable(n)
        cons = [A @ x <= b]
        cp.Problem(cp.Minimize(-c @ x + e0 / 2 * cp.sum_squares(x)), cons).solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12
        )
        lam = np.maximum(cons[0].dual_value, 0)
        if x.value.min() > 1e-6:
            return classify(lam, A, np.linalg.norm(c)), rej
        rej += 1


def rand_near_orth(k, rng, tol=0.1):
    while True:
        E = rng.uniform(-tol, tol, (k, k))
        E = (E + E.T) / 2
        np.fill_diagonal(E, 0)
        # scale so |offdiag|<=tol is kept (average of two uniforms stays within tol)
        N = np.eye(k) + E
        if np.linalg.eigvalsh(N).min() > 0.05:
            return N


def twoval_theta(k, target, m1=1):
    f = lambda t: (m1 + (k - m1) * t) ** 2 / (m1 + (k - m1) * t**2) - target
    return brentq(f, 1e-9, 1.0)


def run(S=500):
    res = {}
    rng = np.random.default_rng(1)
    out = {f: {"unc": 0, "srd": 0, "int": 0, "sat": 0, "hyb": 0} for f in range(1, 7)}
    extra = {}
    Is3 = []
    th3 = []
    th4 = []
    for s in range(S):
        r = np.random.default_rng(1000 + s)
        out[1][fam1(r)[0]] += 1
        (lab, _), rj = fam2(r)
        out[2][lab] += 1
        extra.setdefault("f2_rej", []).append(rj)
        N = rand_near_orth(15, r)
        u = r.uniform(0.5, 1.5, 15)
        lt = 0.5 * u / u.sum()
        (lab, ft), tf, _ = targeted(N, lt, r)
        out[3][lab] += 1
        Is3.append(tf[2])
        th3.append(tf[1])
        N = np.eye(80)
        u = r.uniform(0.8, 1.2, 80)
        lt = 1.0 * u / u.sum()
        (lab, ft), tf, _ = targeted(N, lt, r)
        out[4][lab] += 1
        th4.append(tf[1])
        # family 5: four sub-types in rotation, features within eps/2 of threshold
        typ = s % 4
        d = EPS * r.uniform(-0.5, 0.5)
        if typ == 0:
            k = 80
            N = np.eye(k)
            lt = np.ones(k) / k * (ETA + d)
        elif typ == 1:
            k = 80
            N = np.eye(k)
            t = twoval_theta(k, KS + d)
            lt = np.r_[1, t * np.ones(k - 1)]
            lt = lt / lt.sum() * 0.5
        elif typ == 2:
            k = 80
            N = np.eye(k)
            t = twoval_theta(k, KT + d, m1=1)
            # theta=KT via m1 ones and rest t: use m1=1 gives theta in (1,80)
            lt = np.r_[1, t * np.ones(k - 1)]
            lt = lt / lt.sum() * 0.5
        else:
            k = 70
            g = GI + d
            dl = (1 - g**-2) / (k - 1)
            N = (1 + dl) * np.eye(k) - dl * np.ones((k, k))
            lt = np.ones(k) / k * 0.5
        (lab, ft), tf, _ = targeted(N, lt, r)
        out[5][lab] += 1
        k = 70
        dl = 0.6 / (k - 1)
        N = (1 + dl) * np.eye(k) - dl * np.ones((k, k))
        lt = np.ones(k) / k * 0.5
        (lab, ft), tf, _ = targeted(N, lt, r)
        out[6][lab] += 1
    return out, extra, Is3, th3, th4


if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    out, extra, Is3, th3, th4 = run(S)
    print(json.dumps(out))
    print("f2 redraws", sum(extra["f2_rej"]))
    print("F3 I range", min(Is3), np.median(Is3), max(Is3), "theta", min(th3), max(th3))
    print("F4 theta", min(th4), max(th4))
    json.dump(out, open("/home/claude/table3.json", "w"))
