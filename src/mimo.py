import numpy as np, cvxpy as cp
from regimes_lib import *

M, K = 16, 8
D = 2 * M * K
SIG = 1.0


def channels(rng, delta_db=0.0):
    H = (rng.standard_normal((K, M)) + 1j * rng.standard_normal((K, M))) / np.sqrt(2)
    H[0] *= 10 ** (-delta_db / 20)
    return H


def build(H, gam_db):
    """real data: for user k, rows mapping w=[u_1;v_1;...;u_K;v_K] (blocks of 2M) -> Re/Im of h_k^H w_j."""
    gam = 10 ** (gam_db / 10)
    t = np.sqrt(1 + 1 / gam)
    Mk = []
    ck = []
    E = []
    for k in range(K):
        hr, hi = H[k].real, H[k].imag
        rows = []
        B = np.zeros((2 * K, D))
        for j in range(K):
            o = 2 * M * j
            B[2 * j, o : o + M] = hr
            B[2 * j, o + M : o + 2 * M] = hi  # Re(h^H w_j)=hr.u+hi.v
            B[2 * j + 1, o : o + M] = -hi
            B[2 * j + 1, o + M : o + 2 * M] = hr  # Im(h^H w_j)=hr.v-hi.u
        c = t * B[2 * k]
        E.append(B[2 * k + 1])
        Mk.append(np.vstack([c, B]))
        ck.append(c)
    return Mk, np.array(E), t


def solve_socp(H, gam_db):
    Mk, E, t = build(H, gam_db)
    w = cp.Variable(D)
    cons = []
    for k in range(K):
        y = Mk[k][1:] @ w
        cons.append(cp.SOC(Mk[k][0] @ w, cp.hstack([y, SIG])))
    eq = E @ w == 0
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w)), cons + [eq])
    try:
        prob.solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
        )
    except Exception:
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    lam = np.array([max(float(np.ravel(c.dual_value[0])[0]), 0.0) for c in cons])
    return dict(w=w.value, lam=lam, Mk=Mk, E=E, t=t)


def grads(w, Mk):
    """grad g_k(w) = M_k[1:]^T y/||y|| - c_k, y=(Mk[1:] w, sigma)"""
    G = []
    for k in range(K):
        y = np.r_[Mk[k][1:] @ w, SIG]
        ny = np.linalg.norm(y)
        G.append(Mk[k][1:].T @ (Mk[k][1:] @ w) / ny - Mk[k][0])
    return np.array(G)


def feats(w, lam, Mk, gref):
    G = grads(w, Mk)
    act = np.where(lam > 1e-6 * max(lam.max(), 1e-300))[0]
    if len(act) == 0:
        return 0.0, 1.0, 1.0, 0
    nr = np.linalg.norm(G[act], axis=1)
    lt = lam[act] * nr
    F = G[act].T @ lam[act]
    L, th, I = features_from_reaction(lt, np.linalg.norm(F))
    return L / gref, th, I, len(act)
