"""Regenerates figs/fig_drift_v2.pdf (3 panels) with the new Theorem 19. Run next to drift_experiment.py."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import drift_experiment as de

rng = np.random.default_rng(0)  # same seed/order as drift_experiment.run
store = {}  # (sigma, ratio) -> dict(dist, ok, acc per window)
for sigma in de.SIGMAS:
    numax = de.nu_max(sigma)
    noise = [rng.standard_normal((de.T, de.D)) for _ in range(de.TRIALS)]
    phases = rng.uniform(0, 2 * de.S_TOT, de.TRIALS)
    for r in de.SPEEDS:
        nu = r * numax
        Ws = {"W*": de.w_star(sigma, nu), **{f"W={w}": w for w in de.FIXED_W}}
        acc = {k: [] for k in Ws}
        dist_l, ok_l = [], []
        for tr in range(de.TRIALS):
            om, tp = de.path_at(nu, phases[tr])
            true = de.classify(om)
            dist = np.abs(de.theta_of_t(tp) - de.K_SAT)
            y = om + sigma * noise[tr]
            for k, W in Ws.items():
                ok = de.classify(de.window_means(y, W)) == true
                acc[k].append(ok[de.BURN :].mean())
                if k == "W*":
                    dist_l.append(dist[de.BURN :])
                    ok_l.append(ok[de.BURN :])
        store[(sigma, r)] = dict(
            dist=np.concatenate(dist_l),
            ok=np.concatenate(ok_l),
            acc={k: np.mean(v) for k, v in acc.items()},
            W=Ws["W*"],
        )

# number needed in the text: largest distance of an error, sigma = 3e-6, W*, all speeds
s0 = de.SIGMAS[0]
errd = [store[(s0, r)]["dist"][~store[(s0, r)]["ok"]] for r in de.SPEEDS]
errd = np.concatenate(errd)
print(
    f"sigma=3e-6, W*: #errors={errd.size}, max error distance d_t = {errd.max():.4f}, "
    f"99th pct = {np.percentile(errd, 99):.4f}"
)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) accuracy vs distance, sigma = 1.2e-5, W*
s1 = de.SIGMAS[1]
edges = np.logspace(-3, np.log10(8), 16)
mid = np.sqrt(edges[1:] * edges[:-1])
for r in de.SPEEDS:
    d, ok = store[(s1, r)]["dist"], store[(s1, r)]["ok"]
    a = [
        (
            ok[(d >= lo) & (d < hi)].mean()
            if ((d >= lo) & (d < hi)).sum() >= 20
            else np.nan
        )
        for lo, hi in zip(edges[:-1], edges[1:])
    ]
    ax[0].semilogx(
        mid, a, "o-", ms=3, label=rf"$\nu/\nu_{{\max}}=10^{{{int(np.log10(r))}}}$"
    )
ax[0].axhline(0.99, color="k", ls=":", lw=1)
ax[0].set(
    xlabel=r"distance $d_t=|\vartheta_t-\kappa_{\rm sat}|$",
    ylabel="accuracy",
    title=r"(a) accuracy vs distance, $\sigma=1.2\cdot10^{-5}$, $W^\star$",
    ylim=(0.5, 1.02),
)
ax[0].legend(fontsize=7, loc="lower right")

# (b) d99 vs d_theory
rr = np.array(de.SPEEDS)
for i, s in enumerate(de.SIGMAS):
    d99 = [de.d99(store[(s, r)]["dist"], store[(s, r)]["ok"]) for r in de.SPEEDS]
    ax[1].loglog(rr, d99, "o-", color=f"C{i}", label=rf"$d_{{99}}$, $\sigma={s:.1e}$")
ax[1].loglog(rr, de.d_theory(rr), "k--", label=r"$d_{\rm theory}$ (all $\sigma$)")
ax[1].axhline(0.3, color="gray", ls=":", lw=1)
ax[1].set(
    xlabel=r"$\nu/\nu_{\max}$",
    ylabel="distance",
    title=r"(b) empirical $d_{99}$ vs sufficient distance",
)
ax[1].legend(fontsize=7)

# (c) window comparison, sigma = 3e-6
x = np.arange(len(de.SPEEDS))
w = 0.2
for j, k in enumerate(["W*", "W=1", "W=16", "W=256"]):
    ax[2].bar(
        x + (j - 1.5) * w,
        [store[(s0, r)]["acc"][k] for r in de.SPEEDS],
        w,
        label=k.replace("W*", r"$W^\star$"),
    )
ax[2].set_xticks(x, [rf"$10^{{{int(np.log10(r))}}}$" for r in de.SPEEDS])
ax[2].set(
    xlabel=r"$\nu/\nu_{\max}$",
    ylabel="overall accuracy",
    ylim=(0, 1.05),
    title=r"(c) window comparison, $\sigma=3\cdot10^{-6}$",
)
ax[2].legend(fontsize=7, loc="lower left")

plt.tight_layout()
os.makedirs("figs", exist_ok=True)
plt.savefig("figs/fig_drift_v2.pdf")
print("saved figs/fig_drift_v2.pdf")
