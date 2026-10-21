import numpy as np, json, cvxpy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import solvers as sv, tier1_families as t1
from regimes_lib import *

TH = THETA0
n = 100
k = 80
ALPHA = 0.05


def fe(lt, N=None):
    s = (lt**2).sum()
    L = lt.sum()
    I = 1.0 if N is None else np.sqrt(s / (lt @ N @ lt))
    return np.array([L, L * L / s, I])


def lam_two(target, Lam=0.5):
    t = t1.twoval_theta(k, target)
    lt = np.r_[1, t * np.ones(k - 1)]
    return Lam * lt / lt.sum()


def jac_bound(lt, N=None, eps=1e-7):
    J = []
    for i in range(len(lt)):
        e = np.zeros(len(lt))
        e[i] = eps
        J.append((fe(lt + e, N) - fe(np.maximum(lt - e, 0), N)) / (2 * eps))
    return np.abs(np.array(J)).max()  # sup norm of l1->linf Lipschitz at lt


# ------------- 5.3.1 stabilisation
def sca_hist(A, b, c, tau=1.0, maxit=60):
    H = []
    xk = c.copy()
    for it in range(maxit):
        x = cp.Variable(len(c))
        con = [A @ x <= b]
        sv._solve(
            cp.Problem(
                cp.Minimize(
                    0.5 * cp.sum_squares(x - c) + 0.5 * tau * cp.sum_squares(x - xk)
                ),
                con,
            ),
            1e-10,
        )
        xk = x.value
        H.append((xk.copy(), np.maximum(con[0].dual_value, 0)))
        if sv.kkt_res(xk, H[-1][1], A, b, c) < 1e-8:
            break
    return H


rows = []
for Delta in (0.5, 2.0, 10.0, 18.0):
    for rep in range(6):
        rng = np.random.default_rng(int(1000 * Delta) + rep)
        lt = lam_two(TH["ksat"] + Delta)
        c = rng.standard_normal(n)
        c /= np.linalg.norm(c)
        A = t1.rows_with_gram(np.eye(k), rng)
        b = A @ c - lt
        E = rng.standard_normal((20, n))
        E /= np.linalg.norm(E, axis=1)[:, None]
        xs = c - A.T @ lt
        A2 = np.vstack([A, E])
        b2 = np.concatenate([b, E @ xs + 1.0])
        lam_true = np.r_[lt, np.zeros(20)]
        f0 = fe(lt)
        lab0 = classify_exact(*f0, TH)
        m = margin(*f0, TH, lab0)
        C = jac_bound(lt)
        eta = m / C
        for nm, fn in dict(
            ADMM=lambda: sv.admm(A2, b2, c, hist=True, tol=1e-9)["hist"],
            BCD=lambda: sv.bcd(A2, b2, c, hist=True, tol=1e-9)["hist"],
            MD=lambda: sv.md(A2, b2, c, hist=True, tol=1e-9, maxit=60000)["hist"],
            SCA=lambda: sca_hist(A2, b2, c),
        ).items():
            H = fn()
            phi = []
            labs = []
            for x, lam in H:
                act = np.where(lam > 1e-6 * max(lam.max(), 1e-300))[0]
                L, th, I = (
                    features_from_reaction(
                        lam[act], np.linalg.norm(A2[act].T @ lam[act])
                    )
                    if len(act)
                    else (0, 1, 1)
                )
                labs.append(classify_exact(L, th, I, TH))
                phi.append(
                    np.linalg.norm(x - (c - A.T @ lt)) + np.abs(lam[:k] - lt).sum()
                )
            phi = np.array(phi)
            supf = np.maximum.accumulate(phi[::-1])[::-1]
            bad = [i for i, l in enumerate(labs) if l != lab0]
            kemp = (bad[-1] + 2) if bad else 1  # 1-based first K with all later correct
            ka = int(np.argmax(supf < eta)) + 1 if (supf < eta).any() else None
            rows.append(
                dict(Delta=Delta, solver=nm, kemp=kemp, kA=ka, m=m, eta=eta, len=len(H))
            )
        print("stab", Delta, rep, flush=True)
json.dump(rows, open("res_531.json", "w"))
sm = {}
for nm in ("ADMM", "BCD", "MD", "SCA"):
    r = [x for x in rows if x["solver"] == nm and x["kA"] is not None]
    sm[nm] = dict(
        n=len(r),
        frac_ok=float(np.mean([x["kemp"] <= x["kA"] for x in r])),
        median_ratio=float(np.median([x["kemp"] / x["kA"] for x in r])),
        median_kemp=float(np.median([x["kemp"] for x in r])),
        median_kA=float(np.median([x["kA"] for x in r])),
    )
print("531", sm, flush=True)
json.dump(sm, open("res_531_summary.json", "w"))
# ------------- 5.3.2 sample complexity (Models A and B), exact distribution of the sample mean
rng = np.random.default_rng(11)
T = 2000
out = []
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for Delta in (0.5, 2.0, 5.0):
    for sig in (3e-2, 1e-1):
        lt = lam_two(TH["ksat"] + Delta)
        f0 = fe(lt)
        lab0 = classify_exact(*f0, TH)
        m = margin(*f0, TH, lab0)
        C = jac_bound(lt)
        es = m / C
        a = k
        Nb = 2 * sig**2 * a**2 * np.log(2 * a / ALPHA) / es**2
        Ns = np.unique(np.round(np.logspace(0, np.log10(Nb * 30), 40)).astype(int))
        acc = []
        for N in Ns:
            z = rng.standard_normal((T, k)) * sig / np.sqrt(N)
            lh = np.maximum(lt + z, 0)
            s = (lh**2).sum(1)
            L = lh.sum(1)
            th = L * L / s
            ok = 0
            for i in range(T):
                ok += classify_exact(L[i], th[i], 1.0, TH) == lab0
            acc.append(ok / T)
        acc = np.array(acc)
        Nemp = Ns[np.argmax(acc >= 1 - ALPHA)] if (acc >= 1 - ALPHA).any() else None
        out.append(
            dict(
                model="A",
                Delta=Delta,
                sigma=sig,
                m=m,
                C=C,
                eta_star=es,
                N_bound=Nb,
                N_emp=int(Nemp) if Nemp else None,
                ratio=(Nemp / Nb if Nemp else None),
                acc_at_bound=float(np.interp(Nb, Ns, acc)),
            )
        )
        ax[0].semilogx(Ns, acc, label=f"Δ={Delta},σ={sig}")
        ax[1].semilogx(Ns * es**2 / (sig**2 * a**2 * np.log(2 * a / ALPHA)), acc)
ax[0].set_xlabel("N")
ax[0].set_ylabel("accuracy")
ax[0].legend(fontsize=6)
ax[1].axvline(2.0, c="k", ls="--")
ax[1].set_xlabel(r"$N\eta_*^2/(\sigma^2a^2\log(2a/\alpha))$")
ax[0].axhline(1 - ALPHA, c="gray", ls=":")
ax[1].axhline(1 - ALPHA, c="gray", ls=":")
plt.tight_layout()
plt.savefig("figs/fig_sample.pdf")
# Model B: noise on omega=b (D=k+20 coordinates), lam~ = max(Ac-b_hat,0) on active coords; Lz=1+sqrt(k); sigma_th^2=8 D sigma^2, c0=1
D = k + 20
outB = []
for Delta in (0.5, 2.0, 5.0):
    for sig in (3e-2,):
        lt = lam_two(TH["ksat"] + Delta)
        f0 = fe(lt)
        lab0 = classify_exact(*f0, TH)
        m = margin(*f0, TH, lab0)
        C = jac_bound(lt)
        Lz = 1 + np.sqrt(k)
        eta = m / (C * (1 + Lz))
        N0 = 8 * D * sig**2 * np.log(2 / ALPHA) / eta**2
        Ns = np.unique(np.round(np.logspace(0, np.log10(N0 * 30), 40)).astype(int))
        acc = []
        for N in Ns:
            z = rng.standard_normal((T, k)) * sig / np.sqrt(N)
            lh = np.maximum(lt - z, 0)
            s = (lh**2).sum(1)
            L = lh.sum(1)
            th = L * L / s
            acc.append(
                np.mean(
                    [classify_exact(L[i], th[i], 1.0, TH) == lab0 for i in range(T)]
                )
            )
        acc = np.array(acc)
        Ne = Ns[np.argmax(acc >= 1 - ALPHA)] if (acc >= 1 - ALPHA).any() else None
        outB.append(
            dict(
                model="B",
                Delta=Delta,
                sigma=sig,
                eta=eta,
                N0=N0,
                N_emp=int(Ne) if Ne else None,
                ratio=(Ne / N0 if Ne else None),
            )
        )
json.dump(dict(A=out, B=outB), open("res_532.json", "w"))
print("532", json.dumps(dict(A=out, B=outB))[:1500], flush=True)
