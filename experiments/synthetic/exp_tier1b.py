import numpy as np, json, cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.optimization.solvers import _solve
import experiments.synthetic.tier1_families as t1
from src.regimes.core import *

TH = THETA0
n = 100
EPS = 1e-3


def inst(fam, rng):
    if fam == 3:
        N = t1.rand_near_orth(15, rng)
        u = rng.uniform(0.5, 1.5, 15)
        lt = 0.5 * u / u.sum()
    elif fam == 4:
        N = np.eye(80)
        u = rng.uniform(0.8, 1.2, 80)
        lt = u / u.sum()
    else:
        k = 70
        dl = 0.6 / (k - 1)
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
    return np.vstack([A, E]), np.concatenate([b, E @ xs + 1.0]), c


def solve_gen(Wd, aobj, y0, R, r, yref):
    y = cp.Variable(n)
    con = [R @ y <= r]
    _solve(
        cp.Problem(
            cp.Minimize(0.5 * aobj * cp.sum(cp.multiply(Wd, cp.square(y - y0)))), con
        ),
        1e-10,
    )
    lam = np.maximum(con[0].dual_value, 0)
    yv = y.value
    gref = np.linalg.norm(aobj * Wd * (yref - y0))
    act = np.where(lam > 1e-6 * lam.max())[0]
    nr = np.linalg.norm(R[act], axis=1)
    lt = lam[act] * nr
    F = R[act].T @ lam[act]
    L, th, I = features_from_reaction(lt, np.linalg.norm(F))
    return np.array([L / gref, th, I])


def transformed(A, b, c, kind, rng):
    m = len(b)
    ones = np.ones(n)
    if kind == "identity":
        return solve_gen(ones, 1.0, c, A, b, np.zeros(n))
    if kind == "permutation":
        p = rng.permutation(m)
        return solve_gen(ones, 1.0, c, A[p], b[p], np.zeros(n))
    if kind == "scaling":
        ci = rng.uniform(0.2, 5, m)
        a = rng.uniform(0.3, 3)
        return solve_gen(ones, a, c, A * ci[:, None], b * ci, np.zeros(n))
    if kind in ("rigid", "similarity"):
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        d = rng.standard_normal(n)
        cc = 1.0 if kind == "rigid" else rng.uniform(0.3, 3)
        return solve_gen(
            ones / cc**2, 1.0, d + cc * Q @ c, (A @ Q.T), cc * b + (A @ Q.T) @ d, d
        )
    if kind == "anisotropic":
        s = rng.uniform(0.5, 2, n)
        return solve_gen(1 / s**2, 1.0, s * c, A / s[None, :], b, np.zeros(n))


res = {}
kinds = ["permutation", "scaling", "rigid", "similarity", "anisotropic"]
S = 50
for kd in kinds:
    res[kd] = dict(maxd=[0, 0, 0], agree=0, tot=0)
for fam in (3, 4, 6):
    for s in range(S):
        rng = np.random.default_rng(30000 + 100 * fam + s)
        A, b, c = inst(fam, rng)
        base = transformed(A, b, c, "identity", rng)
        lb = classify_features(*base, TH, EPS)
        for kd in kinds:
            f = transformed(A, b, c, kd, rng)
            d = np.abs(f - base)
            res[kd]["maxd"] = [max(x, y) for x, y in zip(res[kd]["maxd"], d)]
            res[kd]["agree"] += classify_features(*f, TH, EPS) == lb
            res[kd]["tot"] += 1
json.dump(res, open("res_523.json", "w"))
print("523", json.dumps(res), flush=True)


# ---- 5.2.4 margin test: N=I_k, k=80, lambda~ two-valued with vartheta=kappa_sat+Delta
def lam_two(k, target):
    t = t1.twoval_theta(k, target)
    lt = np.r_[1, t * np.ones(k - 1)]
    return 0.5 * lt / lt.sum()


def feats_lt(lt):
    return np.array(
        [lt.sum(), lt.sum() ** 2 / (lt**2).sum(), 1.0]
    )  # Lambda/gref (gref=1), N=I => I=1


def grad_C(lt, eps=1e-6):
    G = []
    for i in range(len(lt)):
        e = np.zeros(len(lt))
        e[i] = eps
        G.append(
            (feats_lt(np.maximum(lt + e, 0)) - feats_lt(np.maximum(lt - e, 0)))
            / (2 * eps)
        )
    return np.abs(np.array(G)).max()  # ||.||_{inf<-l1} bound = max entry


out = []
rng = np.random.default_rng(777)
k = 80
for Delta in (0.1, 0.3, 1.0, 3.0):
    for rep in range(25):
        tgt = KS = (
            TH["ksat"] + Delta * rng.choice([-1, 1]) if False else TH["ksat"] + Delta
        )
        lt = lam_two(k, tgt)
        lab0 = classify_exact(*feats_lt(lt), TH)
        m = margin(*feats_lt(lt), TH, lab0)
        C = grad_C(lt)
        Lz = 1 + np.sqrt(k)
        eta = m / (C * (1 + Lz))
        u = rng.standard_normal(k)
        u /= np.linalg.norm(
            u
        )  # perturb active rhs: lt(delta)=max(lt - delta*u,0) (orthonormal rows, lam=max(Ac-b,0))

        def lab_at(d):
            return classify_exact(*feats_lt(np.maximum(lt - d * u, 1e-300)), TH)

        below = all(lab_at(d * eta) == lab0 for d in np.linspace(0.05, 0.999, 8))
        lo, hi = 0.0, eta
        while lab_at(hi) == lab0 and hi < 1e3 * eta and hi < 0.5 * lt.min() * 1e3:
            hi *= 2
        flip = lab_at(hi) != lab0
        if flip:
            lo = hi / 2
            for _ in range(40):
                mid = (lo + hi) / 2
                if lab_at(mid) == lab0:
                    lo = mid
                else:
                    hi = mid
        out.append(
            dict(
                Delta=Delta,
                m=m,
                C=C,
                eta=eta,
                below_ok=bool(below),
                flip=bool(flip),
                ratio=(hi / eta if flip else None),
            )
        )
r2 = dict(
    n=len(out),
    below_ok=float(np.mean([o["below_ok"] for o in out])),
    flipped=float(np.mean([o["flip"] for o in out])),
    ratio_min=float(min(o["ratio"] for o in out if o["ratio"])),
    ratio_med=float(np.median([o["ratio"] for o in out if o["ratio"]])),
    eta_range=[min(o["eta"] for o in out), max(o["eta"] for o in out)],
)
json.dump(r2, open("res_524.json", "w"))
print("524", r2, flush=True)


# ---- 5.4 paths
def p1(s, k=80):
    phi = (max(s - 0.3, 0) / 0.7) ** 2
    g = 10 ** (-3 + 3 * phi)
    lt = (s / 2) * np.r_[1, g * np.ones(k - 1)]
    return lt


ss = np.linspace(1e-3, 1, 20001)
F1 = np.array([feats_lt(p1(s)) for s in ss])
labs1 = [classify_features(*f, TH, EPS) for f in F1]


def crossings(vals, thr):
    sg = np.sign(vals - thr)
    idx = np.where(sg[:-1] * sg[1:] < 0)[0]
    return ss[idx]


c_L = crossings(F1[:, 0], TH["eta"])
c_s = crossings(F1[:, 1], TH["ksrd"])
c_t = crossings(F1[:, 1], TH["ksat"])
seq = [labs1[0]] + [labs1[i] for i in range(1, len(ss)) if labs1[i] != labs1[i - 1]]
corr = []
i = 0
while i < len(ss):
    if labs1[i] == "hyb":
        j = i
        while j < len(ss) and labs1[j] == "hyb":
            j += 1
        corr.append((ss[i], ss[j - 1]))
        i = j
    else:
        i += 1
# solver check at 12 points
rng = np.random.default_rng(5)
k = 80
c = rng.standard_normal(n)
c /= np.linalg.norm(c)
A = t1.rows_with_gram(np.eye(k), rng)
chk = []
for s in np.linspace(0.05, 1, 12):
    lt = p1(s)
    b = A @ c - lt
    x = cp.Variable(n)
    con = [A @ x <= b]
    _solve(cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - c)), con), 1e-10)
    lam = np.maximum(con[0].dual_value, 0)
    F = A.T @ lam
    f = features_from_reaction(lam[lam > 1e-8 * lam.max()], np.linalg.norm(F))
    chk.append(float(np.abs(np.array(f) - feats_lt(lt)).max()))


# path 2
def p2(s, k=70):
    dl = 0.6 / (k - 1)
    NF = (1 + dl) * np.eye(k) - dl * np.ones((k, k))
    N = (1 - s) * np.eye(k) + s * NF
    lt = np.ones(k) * 0.5 / k
    return lt, N


F2 = []
for s in ss:
    lt, N = p2(s)
    F2.append(
        [lt.sum(), lt.sum() ** 2 / (lt**2).sum(), np.sqrt((lt @ lt) / (lt @ N @ lt))]
    )
F2 = np.array(F2)
labs2 = [classify_features(*f, TH, EPS) for f in F2]
c_I = crossings(F2[:, 2], TH["gI"])
res54 = dict(
    p1_cross_L=c_L.tolist(),
    p1_cross_srd=c_s.tolist(),
    p1_cross_sat=c_t.tolist(),
    p1_sequence=seq,
    p1_hyb_corridors=corr,
    solver_check_maxdiff=max(chk),
    p2_cross_I=c_I.tolist(),
    p2_seq=[labs2[0]]
    + [labs2[i] for i in range(1, len(ss)) if labs2[i] != labs2[i - 1]],
)
json.dump(res54, open("res_54.json", "w"))
print("54", res54, flush=True)
col = dict(unc="tab:gray", srd="tab:red", int="tab:orange", sat="tab:green", hyb="k")
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(ss, F1[:, 0], label=r"$\Lambda$")
ax[0].axhline(TH["eta"], ls="--", c="C0")
ax[0].set_yscale("log")
ax[0].set_xlabel("s")
ax[0].set_ylabel(r"$\Lambda/g_{ref}$")
a2 = ax[0].twinx()
a2.plot(ss, F1[:, 1], c="C1")
a2.axhline(TH["ksrd"], ls="--", c="C1")
a2.axhline(TH["ksat"], ls="--", c="C1")
a2.set_yscale("log")
a2.set_ylabel(r"$\vartheta$")
ax[0].set_title("(a) features along Path 1 ($\mathcal{I}\\equiv1$)")
for lab in col:
    idx = [i for i in range(len(ss)) if labs1[i] == lab]
    if idx:
        ax[1].scatter(
            ss[idx], np.zeros(len(idx)), c=col[lab], s=40, marker="|", label=lab
        )
ax[1].set_yticks([])
ax[1].set_xlabel("s")
ax[1].legend(ncol=5, loc="upper center")
ax[1].set_title("(b) labels (Algorithm 1, eps=1e-3)")
plt.tight_layout()
plt.savefig("figs/fig_path.pdf")
