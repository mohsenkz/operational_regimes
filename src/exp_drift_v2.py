import numpy as np, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tier1_families as t1
from regimes_lib import *

TH = THETA0
k = 80
ALPHA = 0.05
Lg = np.log(2 / ALPHA)
KAP = TH["ksat"]


def lam_two(target):
    t = t1.twoval_theta(k, target)
    lt = np.r_[1, t * np.ones(k - 1)]
    return 0.5 * lt / lt.sum()


def theta_of(X):
    L = X.sum(-1)
    return L * L / (X**2).sum(-1)


us = np.linspace(55, 66, 4001)
P = np.array([lam_two(u) for u in us])
arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
Lpath = arc[-1]


def pos(sarc):
    s = np.mod(sarc, 2 * Lpath)
    s = np.where(s > Lpath, 2 * Lpath - s, s)
    i = np.searchsorted(arc, s).clip(1, len(arc) - 1)
    w = (s - arc[i - 1]) / (arc[i] - arc[i - 1] + 1e-300)
    return P[i - 1] * (1 - w)[:, None] + P[i] * w[:, None]


def jac_bound(lt, eps=1e-7):
    f = lambda x: np.array([x.sum(), x.sum() ** 2 / (x**2).sum(), 1.0])
    J = []
    for i in range(len(lt)):
        e = np.zeros(len(lt))
        e[i] = eps
        J.append((f(lt + e) - f(np.maximum(lt - e, 0))) / (2 * eps))
    return np.abs(np.array(J)).max()


C = max(jac_bound(P[i]) for i in range(0, len(us), 200))
Lz = 1 + np.sqrt(k)
eta = 0.3 / (C * (1 + Lz))
c1 = 3 / 2 ** (2 / 3)
T = 4000
TR = 20
rng = np.random.default_rng(42)
bins = np.logspace(-3, 1, 17)
res = {}


def run(sig, nu, W, TR=TR):
    pooled_d = []
    pooled_ok = []
    for tr in range(TR):
        s0 = rng.uniform(0, 2 * Lpath)
        om = pos(s0 + nu * np.arange(1, T + 1))
        y = om + rng.standard_normal(om.shape) * sig
        cs = np.cumsum(np.r_[np.zeros((1, k)), y], axis=0)
        what = (cs[W:] - cs[:-W]) / W
        idx = np.arange(W, T + 1) - 1
        th_hat = theta_of(np.maximum(what, 1e-300))
        th_true = theta_of(om[idx])
        pooled_d.append(np.abs(th_true - KAP))
        pooled_ok.append((th_hat > KAP) == (th_true > KAP))
    return np.concatenate(pooled_d), np.concatenate(pooled_ok)


def d99(d, ok):
    """smallest bin edge x such that accuracy among points with distance >= x is >= 99% (needs >200 points)."""
    for x in bins:
        m = d >= x
        if m.sum() > 200 and ok[m].mean() >= 0.99:
            return float(x)
    return None


for sig in (3e-6, 1.2e-5, 5e-5):
    sth2 = 8 * k * sig**2
    numax = 4 * eta**3 / (27 * sth2 * Lg)
    out = dict(numax=numax, rows=[])
    for fac in np.logspace(-1, 5, 13):
        nu = fac * numax
        Ws = min(max(1, int(np.ceil((2 * sth2 * Lg / nu**2) ** (1 / 3)))), T // 4)
        row = dict(
            fac=fac,
            nu=nu,
            Wstar=Ws,
            clipped=bool(int(np.ceil((2 * sth2 * Lg / nu**2) ** (1 / 3))) > T // 4),
        )
        for name, W in (("star", Ws), ("W1", 1), ("W16", 16), ("W256", 256)):
            d, ok = run(sig, nu, W)
            row[name] = dict(
                acc=float(ok.mean()),
                d99=d99(d, ok),
                curve=[
                    (
                        float(ok[(d >= bins[i]) & (d < bins[i + 1])].mean())
                        if ((d >= bins[i]) & (d < bins[i + 1])).sum() > 50
                        else None
                    )
                    for i in range(len(bins) - 1)
                ],
            )
        row["d_theory"] = float(
            C * (1 + Lz) * c1 * sth2 ** (1 / 3) * (nu ** (1 / 3)) * Lg ** (1 / 3)
        )
        out["rows"].append(row)
    res[str(sig)] = out
    print("sigma", sig, "done", flush=True)
json.dump(
    dict(C=C, Lz=Lz, eta=eta, res=res, bins=bins.tolist()),
    open("res_drift_v2.json", "w"),
    default=float,
)
# figures
sig = "1.2e-05"
R = res[sig]["rows"]
fac = np.array([r["fac"] for r in R])
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
bc = np.sqrt(bins[:-1] * bins[1:])
for r in R[::3]:
    cv = np.array([np.nan if v is None else v for v in r["star"]["curve"]])
    ax[0].semilogx(bc, cv, "o-", ms=3, label=f"ν/ν_max={r['fac']:.0e}, W*={r['Wstar']}")
ax[0].set_xlabel(r"distance to boundary $|\vartheta_t-\kappa_{sat}|$")
ax[0].set_ylabel("tracking accuracy")
ax[0].axhline(0.95, c="gray", ls=":")
ax[0].legend(fontsize=6)
ax[0].set_title(r"(a) accuracy vs distance, $\sigma$=1.2e-5")
for sg, mk in (("3e-06", "o"), ("1.2e-05", "s"), ("5e-05", "^")):
    RR = res[sg]["rows"]
    f = np.array([r["fac"] for r in RR])
    ax[1].loglog(
        f,
        [r["star"]["d99"] or np.nan for r in RR],
        mk + "-",
        label=f"empirical d99 (W*), σ={sg}",
    )
    ax[1].loglog(
        f, [r["d_theory"] for r in RR], mk + "--", alpha=0.5, label=f"theory, σ={sg}"
    )
ax[1].axhline(0.3, c="k", ls=":")
ax[1].set_xlabel(r"$\nu/\nu_{max}$")
ax[1].set_ylabel("distance for 99% accuracy")
ax[1].legend(fontsize=5)
ax[1].set_title("(b) empirical vs sufficient distance")
for nm, mk in (("star", "o"), ("W1", "s"), ("W16", "^"), ("W256", "d")):
    ax[2].semilogx(
        fac,
        [r[nm]["acc"] for r in R],
        mk + "-",
        label={"star": "W*", "W1": "W=1", "W16": "W=16", "W256": "W=256"}[nm],
    )
ax[2].set_xlabel(r"$\nu/\nu_{max}$")
ax[2].set_ylabel("overall accuracy (all times)")
ax[2].legend()
ax[2].set_title("(c) window choice, σ=1.2e-5")
plt.tight_layout()
plt.savefig("figs/fig_drift_v2.pdf")
print("fig ok")
