import numpy as np, scipy.io as sio, scipy.sparse as sp, json, sys, time, glob, os
from qpsolvers import Problem, solve_problem
from regimes_lib import *

MM = "/home/claude/mm"
BIG = 1e19


def load(name):
    d = sio.loadmat(f"{MM}/{name}.mat")
    P = sp.csc_matrix(d["P"])
    q = np.asarray(d["q"]).ravel().astype(float)
    A = sp.csr_matrix(d["A"])
    l = np.asarray(d["l"]).ravel().astype(float)
    u = np.asarray(d["u"]).ravel().astype(float)
    return P, q, A, l, u


def split(A, l, u):
    eq = np.where(np.abs(u - l) < 1e-12)[0]
    rows = []
    h = []
    sgn = []
    for i in range(A.shape[0]):
        if i in set(eq):
            continue
    eqset = set(eq.tolist())
    G = []
    hh = []
    Gr = []
    meta = []
    for i in range(A.shape[0]):
        if i in eqset:
            continue
        if l[i] > -BIG:
            Gr.append(-A[i])
            hh.append(-l[i])
        if u[i] < BIG:
            Gr.append(A[i])
            hh.append(u[i])
    G = sp.vstack(Gr).tocsc() if Gr else None
    return (
        A[eq].tocsc() if len(eq) else None,
        l[eq] if len(eq) else None,
        G,
        np.array(hh),
    )


def run_one(name):
    P, q, A, l, u = load(name)
    n = len(q)
    m = A.shape[0]
    rec = dict(name=name, n=n, m=m)
    Aeq, beq, G, h = split(A, l, u)
    t0 = time.time()
    try:
        sol = solve_problem(
            Problem(P, q, G, (h if G is not None else None), Aeq, beq),
            solver="clarabel",
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
            time_limit=120.0,
        )
    except Exception as e:
        rec["status"] = "solver_error"
        rec["err"] = str(e)[:80]
        return rec
    rec["solve_s"] = time.time() - t0
    if not sol.found:
        rec["status"] = "not_solved"
        return rec
    x = sol.x
    z = np.maximum(sol.z, 0) if sol.z is not None else np.zeros(0)
    grad = P @ x + q
    # active inequality rows
    slack = (G @ x - h) if G is not None else np.zeros(0)
    zmax = z.max() if z.size else 0.0
    act = np.where(z > 1e-6 * max(zmax, 1e-300))[0] if zmax > 0 else np.array([], int)
    weak = (
        np.where(
            (np.abs(slack) <= 1e-7 * (1 + np.abs(h))) & (z <= 1e-6 * max(zmax, 1e-300))
        )[0]
        if G is not None and zmax > 0
        else np.array([], int)
    )
    rec["q_act"] = int(len(act))
    rec["n_weak"] = int(len(weak))
    rec["res"] = float(
        np.linalg.norm(
            grad
            + (G.T @ z if G is not None else 0)
            + (Aeq.T @ sol.y if Aeq is not None else 0)
        )
    )
    # SC/LICQ/SOSC checks
    Ga = G[act] if len(act) else None
    blocks = [B.toarray() for B in (Ga, Aeq) if B is not None]
    Jac = np.vstack(blocks) if blocks else np.zeros((0, n))
    flags = []
    if Jac.shape[0] > n:
        flags.append("LICQ")
        Z = np.zeros((n, 0))
    elif Jac.shape[0] > 0:
        U, S, Vt = np.linalg.svd(Jac, full_matrices=True)
        r = int((S > 1e-9 * max(1, S[0])).sum())
        if r < Jac.shape[0]:
            flags.append("LICQ")
        Z = Vt[r:].T
    else:
        Z = np.eye(n)
    if len(weak) > 0:
        flags.append("SC")
    if Z.shape[1] > 0 and "LICQ" not in flags:
        H = Z.T @ (P @ Z)
        ev = np.linalg.eigvalsh((H + H.T) / 2)
        if ev.min() <= 1e-8 * max(1, np.abs(ev).max()):
            flags.append("SOSC")
    rec["flags"] = flags
    # features
    if len(act):
        Ga_d = Ga.toarray()
        nr = np.linalg.norm(Ga_d, axis=1)
        lt = z[act] * nr
        F = Ga_d.T @ z[act]
        Fn = np.linalg.norm(F)
    else:
        lt = np.zeros(0)
        Fn = 0.0
    # g_ref
    xref = np.zeros(n)
    gref = np.linalg.norm(P @ xref + q)
    if gref == 0:
        gref = np.linalg.norm(P @ np.ones(n) + q)
    if gref == 0:
        rec["status"] = "excluded_gref"
        return rec
    L, th, I = features_from_reaction(lt, Fn)
    rec.update(
        status=("classified" if not flags else "relaxed"),
        Lh=float(L / gref),
        th=float(th),
        I=float(I),
        gref=float(gref),
    )
    return rec


if __name__ == "__main__":
    names = [l.strip() for l in open(f"{MM}/names.txt")]
    sel = []
    for nm in names:
        d = sio.loadmat(f"{MM}/{nm}.mat")
        n = int(d["n"][0, 0])
        m = int(d["m"][0, 0])
        if 10 <= n <= 1500 and 5 <= m <= 3000:
            sel.append(nm)
    out = []
    for nm in sel:
        t0 = time.time()
        r = run_one(nm)
        out.append(r)
        print(nm, r.get("status"), round(time.time() - t0, 1), flush=True)
        json.dump(out, open("tier2_mm_raw.json", "w"))
