import numpy as np, cvxpy as cp
from regimes_lib import *


# problem: min 1/2||x-c||^2  s.t. A x <= b  (A unit rows)  ; returns dict(x,lam,iters,res,hist(optional))
def kkt_res(x, lam, A, b, c):
    r1 = np.linalg.norm(x - c + A.T @ lam)
    r2 = np.linalg.norm(np.maximum(A @ x - b, 0))
    r3 = np.abs(lam * (b - A @ x)).sum()
    return r1 + r2 + r3


def _solve(prob, tol):
    try:
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol)
    except Exception:
        prob.solve(solver=cp.CLARABEL)


def ipm(A, b, c, tol=1e-9):
    x = cp.Variable(len(c))
    con = [A @ x <= b]
    _solve(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - c)), con), tol)
    lam = np.maximum(con[0].dual_value, 0)
    return dict(x=x.value, lam=lam, iters=1, res=kkt_res(x.value, lam, A, b, c))


def admm(A, b, c, rho=1.0, tol=1e-7, maxit=20000, hist=False):
    n = len(c)
    M = np.linalg.inv(np.eye(n) + rho * A.T @ A)
    z = np.minimum(A @ c, b)
    u = np.zeros(len(b))
    H = []
    for k in range(maxit):
        x = M @ (c + rho * A.T @ (z - u))
        Ax = A @ x
        z = np.minimum(Ax + u, b)
        u = u + Ax - z
        lam = np.maximum(rho * u, 0)
        if hist:
            H.append((x.copy(), lam.copy()))
        if k % 5 == 0 and kkt_res(x, lam, A, b, c) <= tol:
            break
    return dict(x=x, lam=lam, iters=k + 1, res=kkt_res(x, lam, A, b, c), hist=H)


def bcd(
    A, b, c, tol=1e-7, maxit=20000, hist=False
):  # dual coordinate ascent (Hildreth), unit rows
    m = len(b)
    lam = np.zeros(m)
    x = c.copy()
    H = []
    for k in range(maxit):
        for i in range(m):
            d = A[i] @ x - b[i]
            new = max(0.0, lam[i] + d)  # a_i^T x(lam) = a_i^T x ; step for unit row
            delta = new - lam[i]
            if delta != 0:
                x = x - delta * A[i]
                lam[i] = new
        if hist:
            H.append((x.copy(), lam.copy()))
        if kkt_res(x, lam, A, b, c) <= tol:
            break
    return dict(x=x, lam=lam, iters=k + 1, res=kkt_res(x, lam, A, b, c), hist=H)


def md(
    A, b, c, tol=1e-7, maxit=100000, lam0=1e-3, hist=False
):  # exponentiated-gradient (entropic mirror ascent) on dual
    m = len(b)
    L = np.linalg.eigvalsh(A @ A.T).max()
    step = 1.0 / L
    lam = np.full(m, lam0)
    H = []
    for k in range(maxit):
        x = c - A.T @ lam
        g = A @ x - b
        lam = lam * np.exp(np.clip(step * g, -50, 50))
        if hist and k % 10 == 0:
            H.append((x.copy(), lam.copy()))
        if k % 20 == 0:
            x = c - A.T @ lam
            if kkt_res(x, lam, A, b, c) <= tol:
                break
    x = c - A.T @ lam
    return dict(x=x, lam=lam, iters=k + 1, res=kkt_res(x, lam, A, b, c), hist=H)


def sca(
    A, b, c, tau=1.0, tol=1e-7, maxit=200
):  # proximal-point successive convex approximation
    xk = c.copy()
    lam = np.zeros(len(b))
    for k in range(maxit):
        x = cp.Variable(len(c))
        con = [A @ x <= b]
        _solve(
            cp.Problem(
                cp.Minimize(
                    0.5 * cp.sum_squares(x - c) + 0.5 * tau * cp.sum_squares(x - xk)
                ),
                con,
            ),
            1e-10,
        )
        xk = x.value
        lam = np.maximum(con[0].dual_value, 0) / (
            1.0
        )  # multiplier of subproblem; equals true lam at fixed point
        # at fixed point x-c+A^T lam=0
        if kkt_res(xk, lam, A, b, c) <= tol:
            break
    return dict(x=xk, lam=lam, iters=k + 1, res=kkt_res(xk, lam, A, b, c))


SOLVERS = dict(IPM=ipm, ADMM=admm, SCA=sca, BCD=bcd, MD=md)
