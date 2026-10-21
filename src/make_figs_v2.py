import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

code = dict(unc=0, srd=1, int=2, sat=3, hyb=4)
names = ["unc", "srd", "int", "sat", "hyb"]
cm = matplotlib.colors.ListedColormap(
    ["tab:gray", "tab:red", "tab:orange", "tab:green", "k"]
)


def panel(ax, cells, levels, dels, xlabel, title, th):
    Z = np.full((len(dels), len(levels)), np.nan)
    txt = {}
    for i, d in enumerate(dels):
        for j, g in enumerate(levels):
            c = cells.get(f"{g}_{d}", {})
            tot = sum(c.values())
            if tot == 0:
                continue
            lab = max(c, key=c.get)
            Z[i, j] = code[lab]
            txt[(i, j)] = f"{lab}\n{100*c[lab]//tot}%"
    ax.imshow(
        np.ma.masked_invalid(Z), origin="lower", cmap=cm, vmin=0, vmax=4, aspect="auto"
    )
    ax.set_xticks(range(len(levels)), levels)
    ax.set_yticks(range(len(dels)), dels)
    for (i, j), t in txt.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=6, color="w")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("extra path loss of user 1 (dB)")
    ax.set_title(
        title
        + "\n"
        + r"$\Theta$=(%.2f, %.1f, %.1f, %.2f)"
        % (th["eta"], th["ksrd"], th["ksat"], th["gI"]),
        fontsize=8,
    )


a = json.load(open("res_mimo_v2_summary.json"))["minpower"]
b = json.load(open("res_mimo_v3_summary.json"))["rate"]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
panel(
    ax[0],
    a["cells"],
    [0, 5, 10, 15, 20, 25],
    [0, 5, 10, 15, 20],
    "SINR target (dB)",
    "(a) min-power SOCP, calibrated",
    a["theta"],
)
panel(
    ax[1],
    b["cells"],
    [5, 10, 15, 20, 25],
    [0, 5, 10, 15, 20],
    "power cap $P_{max}$ (dB)",
    "(b) rate maximisation, QoS floor 0 dB, calibrated",
    b["theta"],
)
plt.tight_layout()
plt.savefig("figs/fig_mimo_map_v2.pdf")
print("ok")
