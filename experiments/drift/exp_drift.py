import numpy as np, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import experiments.synthetic.tier1_families as t1
from src.regimes.core import *

TH = THETA0
k = 80
ALPHA = 0.05
Lg = np.log(2 / ALPHA)


def fe(lt):
    s = (lt**2).sum()
    L = lt.sum()
    return np.array([L, L * L / s, 1.0])


def lam_two(target):
    t = t1.twoval_theta(k, target)
    lt = np.r_[1, t * np.ones(k - 1)]
    return 0.5 * lt / lt.sum()


# path u in [55,66] (vartheta target) in omega-space: omega = Ac - lam~(u); N=Id so we work with lam~ coordinates (isometric up to sign)
us = np.linspace(55, 66, 4001)
P = np.array([lam_two(u) for u in us])
seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
arc = np.r_[0, np.cumsum(seg)]
Lpath = arc[-1]


def pos(sarc):  # bounce back and forth along arc length
    s = np.mod(sarc, 2 * Lpath)
    s = np.where(s > Lpath, 2 * Lpath - s, s)
    i = np.searchsorted(arc, s).clip(1, len(arc) - 1)
    w = (s - arc[i - 1]) / (arc[i] - arc[i - 1] + 1e-300)
    return P[i - 1] * (1 - w)[:, None] + P[i] * w[:, None]


# constants: margin>=m_min on eligible times |vartheta-kappa_sat|>=0.3 ; C = sup Jacobian bound over path points; L_z=1+sqrt(k)
def jac_bound(lt, eps=1e-7):
    J = []
    for i in range(len(lt)):
        e = np.zeros(len(lt))
        e[i] = eps
        J.append((fe(lt + e) - fe(np.maximum(lt - e, 0))) / (2 * eps))
    return np.abs(np.array(J)).max()


C = max(jac_bound(P[i]) for i in range(0, len(us), 200))
Lz = 1 + np.sqrt(k)
mmin = 0.3
eta = mmin / (C * (1 + Lz))
import sys

sig = float(sys.argv[1])
tag = sys.argv[2]
D = k
sth2 = 8 * D * sig**2
numax = 4 * eta**3 / (27 * sth2 * Lg)
print("Lpath", Lpath, "C", C, "eta", eta, "numax", numax, flush=True)
T = 2000
trials = 20
res = []
rng = np.random.default_rng(3)
for fac in np.logspace(-1, 4, 11):
    nu = fac * numax
    Wst = int(np.ceil((2 * sth2 * Lg / nu**2) ** (1 / 3)))
    Wst = max(1, Wst)
    corr = 0
    tot = 0
    for tr in range(trials):
        s0 = rng.uniform(0, 2 * Lpath)
        t = np.arange(1, T + 1)
        om = pos(s0 + nu * t)  # arc-length speed nu  => ||omega_{t+1}-omega_t||<=nu
        y = om + rng.standard_normal(om.shape) * sig
        cs = np.cumsum(np.r_[np.zeros((1, k)), y], axis=0)
        # vectorised: true labels and eligibility
        th_true = np.array([fe(o)[1] for o in om])
        elig = np.abs(th_true - TH["ksat"]) >= mmin
        what = (cs[Wst:] - cs[:-Wst]) / Wst
        idx = np.arange(Wst, T + 1) - 1  # estimates at times Wst..T
        lab_hat = np.array(
            [classify_exact(*fe(np.maximum(w, 1e-300)), TH) for w in what]
        )
        lab_true = np.array([classify_exact(*fe(o), TH) for o in om[idx]])
        e = elig[idx]
        corr += np.sum((lab_hat == lab_true) & e)
        tot += np.sum(e)
    res.append(
        dict(
            fac=float(fac),
            nu=float(nu),
            W=Wst,
            acc=float(corr / max(tot, 1)),
            n=int(tot),
        )
    )
    print(res[-1], flush=True)
json.dump(
    dict(C=C, Lz=Lz, eta=eta, numax=numax, sigma=sig, Lpath=Lpath, res=res),
    open(f"res_533_{tag}.json", "w"),
)
f = np.array([r["fac"] for r in res])
a = np.array([r["acc"] for r in res])
plt.figure(figsize=(5.2, 3.6))
plt.semilogx(f, a, "o-")
plt.axvline(1, ls="--", c="k")
plt.axhline(1 - ALPHA, c="gray", ls=":")
plt.xlabel(r"$\nu/\nu_{max}$")
plt.ylabel("tracking accuracy (eligible times)")
plt.tight_layout()
plt.savefig(f"figs/fig_drift_{tag}.pdf")
