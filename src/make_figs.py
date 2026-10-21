import json, numpy as np, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# stabilisation
rows = json.load(open("res_531.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
mk = dict(ADMM="o", BCD="s", MD="^", SCA="d")
for nm in mk:
    r = [x for x in rows if x["solver"] == nm and x["kA"]]
    ax[0].loglog(
        [x["kA"] for x in r], [x["kemp"] for x in r], mk[nm], label=nm, alpha=0.7
    )
lo = 1
hi = max(x["kA"] for x in rows if x["kA"]) * 1.5
ax[0].plot([lo, hi], [lo, hi], "k-")
ax[0].set_xlabel(r"sufficient index $k^\star_A$")
ax[0].set_ylabel(r"empirical $k^\star_{emp}$")
ax[0].legend()
ax[1].bar(
    list(mk),
    [
        np.median([x["kemp"] / x["kA"] for x in rows if x["solver"] == nm and x["kA"]])
        for nm in mk
    ],
)
ax[1].set_ylabel(r"median $k^\star_{emp}/k^\star_A$")
ax[1].axhline(1, c="k", ls="--")
plt.tight_layout()
plt.savefig("figs/fig_stabilisation.pdf")
plt.close()
# MIMO map
m = json.load(open("res_mimo_map.json"))
gams = [0, 5, 10, 15, 20, 25]
dels = [0, 5, 10, 15, 20]
code = dict(unc=0, srd=1, int=2, sat=3, hyb=4)
Z = np.full((len(dels), len(gams)), np.nan)
Iv = np.zeros_like(Z)
frac = np.zeros_like(Z)
for i, d in enumerate(dels):
    for j, g in enumerate(gams):
        c = m[f"{g}_{d}"]["counts"]
        tot = sum(c.values())
        lab = max((k for k in c if k in code), key=lambda k: c[k])
        Z[i, j] = code[lab]
        frac[i, j] = c[lab] / tot
        Iv[i, j] = m[f"{g}_{d}"]["med"][2]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
cm = matplotlib.colors.ListedColormap(
    ["tab:gray", "tab:red", "tab:orange", "tab:green", "k"]
)
ax[0].imshow(Z, origin="lower", cmap=cm, vmin=0, vmax=4, aspect="auto")
ax[0].set_xticks(range(len(gams)), gams)
ax[0].set_yticks(range(len(dels)), dels)
ax[0].set_xlabel("SINR target (dB)")
ax[0].set_ylabel("extra path loss of user 1 (dB)")
for i in range(len(dels)):
    for j in range(len(gams)):
        ax[0].text(
            j,
            i,
            ["unc", "srd", "int", "sat", "hyb"][int(Z[i, j])]
            + f"\n{100*frac[i,j]:.0f}%",
            ha="center",
            va="center",
            fontsize=6,
            color="w",
        )
im = ax[1].imshow(Iv, origin="lower", aspect="auto")
ax[1].set_xticks(range(len(gams)), gams)
ax[1].set_yticks(range(len(dels)), dels)
ax[1].set_xlabel("SINR target (dB)")
ax[1].set_title("median interference index $\\mathcal{I}$")
plt.colorbar(im, ax=ax[1])
ax[0].set_title("majority label (share of 20 realisations)")
plt.tight_layout()
plt.savefig("figs/fig_mimo_map.pdf")
plt.close()
# ADMM
if os.path.exists("res_admm.json"):
    a = json.load(open("res_admm.json"))
    sts = ["fixed", "5rho_or", "balance", "regime", "random"]
    fig, ax = plt.subplots(1, len(a), figsize=(4 * len(a), 3.6), squeeze=False)
    for c, (k, v) in enumerate(a.items()):
        ax[0, c].boxplot(
            [[r["it"] for r in v["res"][s]] for s in sts],
            tick_labels=["fixed", "5ρ+OR", "balance", "regime", "random"],
        )
        ax[0, c].set_yscale("log")
        ax[0, c].set_title(
            f'SINR/Δ = {k.replace("_"," dB / ")} dB  (ρ0={v["rho0"]})', fontsize=8
        )
        ax[0, c].set_ylabel("iterations to 1e-6")
    plt.tight_layout()
    plt.savefig("figs/fig_admm.pdf")
    plt.close()
print("ok")
