import numpy as np, json, itertools, time
from solvers import *
import tier1_families as t1
from regimes_lib import *

n = 100
TH = THETA0


def build(fam, rng):
    if fam == 3:
        N = t1.rand_near_orth(15, rng)
        u = rng.uniform(0.5, 1.5, 15)
        lt = 0.5 * u / u.sum()
    elif fam == 4:
        N = np.eye(80)
        u = rng.uniform(0.8, 1.2, 80)
        lt = u / u.sum()
    elif fam == 6:
        k = 70
        dl = 0.6 / (k - 1)
        N = (1 + dl) * np.eye(k) - dl * np.ones((k, k))
        lt = np.ones(k) / k * 0.5
    else:  # 5: four subtypes at random
        typ = rng.integers(4)
        d = 1e-3 * rng.uniform(-0.5, 0.5)
        if typ == 0:
            k = 80
            N = np.eye(k)
            lt = np.ones(k) / k * (0.05 + d)
        elif typ == 1:
            k = 80
            N = np.eye(k)
            t = t1.twoval_theta(k, 1.4 + d)
            lt = np.r_[1, t * np.ones(k - 1)]
            lt = lt / lt.sum() * 0.5
        elif typ == 2:
            k = 80
            N = np.eye(k)
            t = t1.twoval_theta(k, 60.5 + d)
            lt = np.r_[1, t * np.ones(k - 1)]
            lt = lt / lt.sum() * 0.5
        else:
            k = 70
            g = 1.3 + d
            dl = (1 - g**-2) / (k - 1)
            N = (1 + dl) * np.eye(k) - dl * np.ones((k, k))
            lt = np.ones(k) / k * 0.5
    k = len(lt)
    c = rng.standard_normal(n)
    c /= np.linalg.norm(c)
    A = t1.rows_with_gram(N, rng)
    b = A @ c - N @ lt
    E = rng.standard_normal((20, n))
    E /= np.linalg.norm(E, axis=1)[:, None]
    xs = c - A.T @ lt
    return np.vstack([A, E]), np.concatenate([b, E @ xs + 1.0]), c, k


def label(x, lam, A):
    act = (
        np.where(lam > 1e-6 * max(lam.max(), 1e-300))[0]
        if lam.max() > 0
        else np.array([], int)
    )
    if len(act) == 0:
        return "unc"
    lt = lam[act]
    F = A[act].T @ lam[act]
    L, th, I = features_from_reaction(lt, np.linalg.norm(F))
    return classify_features(L, th, I, TH, 1e-3)


S = 100
names = list(SOLVERS)
rows = []
for fam in (3, 4, 5, 6):
    for s in range(S):
        rng = np.random.default_rng(20000 + 100 * fam + s)
        A, b, c, k = build(fam, rng)
        rec = dict(fam=fam, lab={}, x={}, res={}, it={})
        for nm, f in SOLVERS.items():
            r = f(A, b, c)
            rec["lab"][nm] = label(r["x"], r["lam"], A)
            rec["x"][nm] = r["x"]
            rec["res"][nm] = r["res"]
            rec["it"][nm] = r["iters"]
        rows.append(rec)
    print("fam", fam, "done", flush=True)
agr_all = np.zeros((5, 5))
agr_core = np.zeros((5, 5))
n_core = 0
excl = 0
for rec in rows:
    ok = all(rec["res"][nm] <= 2e-7 for nm in names)
    same = ok and all(
        np.linalg.norm(rec["x"][a] - rec["x"][b]) <= 1e-5 for a in names for b in names
    )
    core = same and all(rec["lab"][nm] != "hyb" for nm in names)
    if not same:
        excl += 1
    if core:
        n_core += 1
    for i, a in enumerate(names):
        for j, bn in enumerate(names):
            agr_all[i, j] += rec["lab"][a] == rec["lab"][bn]
            if core:
                agr_core[i, j] += rec["lab"][a] == rec["lab"][bn]
agr_all /= len(rows)
agr_core /= max(n_core, 1)
conv = {nm: float(np.mean([rec["res"][nm] <= 2e-7 for rec in rows])) for nm in names}
its = {nm: float(np.median([rec["it"][nm] for rec in rows])) for nm in names}
out = dict(
    S=len(rows),
    n_core=n_core,
    n_not_same_point=excl,
    names=names,
    agr_all=agr_all.tolist(),
    agr_core=agr_core.tolist(),
    conv_frac=conv,
    median_iters=its,
)
json.dump(out, open("res_522.json", "w"))
print(json.dumps(out, indent=1))
