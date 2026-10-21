#!/usr/bin/env python3
"""Recompute Fig. 1 using Definition 1, equations (19)-(22), (35).
Run: python plot_regimes.py
Dependencies: numpy, matplotlib

The old polygonal boundaries cannot equal participation-ratio level sets.
Default thresholds are illustrative admissible choices, NOT values extracted
from the raster figure. Unit orthogonal gradients imply tilde(lambda)=lambda.
Panel (b) shows the section Lambda=1 > delta, not a projection of unc.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS = ["#1d6ff2", "#e41717", "#f2b751", "#3ceb99", "#1c262a"]
NAMES = [
    r"$R^\circ_{\rm unc}$  Negligible reaction",
    r"$R^\circ_{\rm srd}$  Single-resource dominant",
    r"$R^\circ_{\rm int}$  Intermediate",
    r"$R^\circ_{\rm sat}$  Saturated / distributed",
    r"$R_{\rm hyb}$  Hybrid interface",
]


def classify(lam, delta, ksrd, ksat, gamma, gradients=np.eye(3)):
    """Exact strict predicates of Definition 1 (no artificial hybrid band).

    gradients[i] is grad(g_i). Coordinates are RAW multipliers.
    Return labels, normalized reaction profiles, mass, participation, interference.
    Numerical equality is ordinary floating-point equality, not an epsilon band.
    Zero coordinates use the natural lower-face extension of the definitions.
    """
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    G = np.asarray(gradients, dtype=float)
    if np.any(lam < 0):
        raise ValueError("Multipliers must be nonnegative")
    norms = np.linalg.norm(G, axis=1)
    if np.any(norms == 0):
        raise ValueError("Nonzero constraint gradients required")
    reaction = lam * norms
    mass = reaction.sum(axis=1)
    p = np.divide(
        reaction, mass[:, None], out=np.zeros_like(reaction), where=mass[:, None] > 0
    )
    q = np.sum(p * p, axis=1)
    theta = np.divide(1.0, q, out=np.zeros_like(q), where=q > 0)
    normals = G / norms[:, None]
    gram = normals @ normals.T
    net2 = np.einsum("bi,ij,bj->b", p, gram, p)
    interference = np.sqrt(
        np.divide(q, net2, out=np.full_like(q, np.inf), where=net2 > 0)
    )
    labels = np.full(len(lam), 4, dtype=np.int8)
    active = mass > delta
    labels[mass < delta] = 0
    labels[active & (theta < ksrd)] = 1
    labels[active & (theta > ksrd) & ((theta < ksat) | (interference > gamma))] = 2
    labels[active & (theta > ksat) & (interference < gamma)] = 3
    return labels, p, mass, theta, interference


def circle_on_simplex(k, count=4000):
    """Exact theta=k boundary, clipped to nonnegative simplex."""
    t = np.linspace(0, 2 * np.pi, count)
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
    v = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    r = np.sqrt(1 / k - 1 / 3)
    p = np.ones((count, 3)) / 3 + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    p[np.any(p < 0, axis=1)] = np.nan
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--ksrd", type=float, default=1.6)
    ap.add_argument("--ksat", type=float, default=2.6)
    ap.add_argument("--gamma", type=float, default=1.2)
    ap.add_argument("--points", type=int, default=130000)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--output", default="dual_regimes")
    args = ap.parse_args()
    d, a, b, g = args.delta, args.ksrd, args.ksat, args.gamma
    if not (0 < d < 1 and 1 < a < 2 < b < 3 and g > 1):
        ap.error("Require 0<delta<1, 1<ksrd<2<ksat<3, gamma>1")
    rng = np.random.default_rng(args.seed)
    raw = rng.uniform(0, 2, (args.points, 3))
    # Supplemental near-origin samples make the small unc tetrahedron visible.
    near = rng.dirichlet(np.ones(3), 5000) * rng.uniform(0, d, (5000, 1))
    raw = np.vstack([raw, near])
    lab, profiles, mass, theta, interference = classify(raw, d, a, b, g)
    simplex = rng.dirichlet(np.ones(3), args.points // 2)
    slabels = classify(simplex, d, a, b, g)[0]
    assert np.allclose(interference, 1)
    # Normalization preserves shape labels above the mass cutoff.
    active = mass > d
    assert np.array_equal(lab[active], classify(profiles[active], d, a, b, g)[0])
    assert set(np.unique(lab)) >= {0, 1, 2, 3}

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "mathtext.fontset": "stix"}
    )
    fig = plt.figure(figsize=(18, 9), facecolor="#fefefe")
    left = fig.add_subplot(121, projection="3d")
    right = fig.add_subplot(122, projection="3d")
    for ax in (left, right):
        ax.set_facecolor("#fefefe")
        ax.view_init(elev=24, azim=45)
        ax.set_box_aspect((1, 1, 1))
        for setlim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setlim(0, 1)
        ax.tick_params(labelsize=9)
    left.set(xlim=(0, 2), ylim=(0, 2), zlim=(0, 2))
    ticks = np.linspace(0, 2, 6)  # 0، 0.4، 0.8، 1.2، 1.6، 2
    left.set_xticks(ticks)
    left.set_yticks(ticks)
    left.set_zticks(ticks)
    for i in range(4):
        xyz = raw[lab == i]
        left.scatter(
            *xyz.T,
            c=COLORS[i],
            s=0.7 if i else 2,
            alpha=1.0 if i else 0.65,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
    # for j in range(3):
    #     line = np.zeros((2, 3))
    #     line[1, j] = 1
    #     left.plot(*line.T, color="#a83736", lw=2)
    # # Actual hybrid surfaces: conical theta levels outside the unc tetrahedron.
    # for k in (a, b):
    #     curve = circle_on_simplex(k, 1000)
    #     for scale in (d, 0.35, 0.65, 1.0, 1.4, 2.0):
    #         surface_curve = curve * scale
    #         surface_curve[np.any(surface_curve > 1, axis=1)] = np.nan
    #         left.plot(*surface_curve.T, color=COLORS[4], lw=0.55, alpha=0.50)
    # # Lambda=delta is the entire near-origin hybrid triangle.
    # edge = d * np.eye(3)[[0, 1, 2, 0]]
    # left.plot(*edge.T, color=COLORS[4], lw=1.1)
    for j in range(3):
        line = np.zeros((2, 3))
        line[1, j] = 2
        left.plot(*line.T, color="#a83736", lw=2)
    for i in (1, 2, 3):
        xyz = simplex[slabels == i]
        right.scatter(
            *xyz.T,
            c=COLORS[i],
            s=3,
            alpha=0.65,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
    border = np.eye(3)[[0, 1, 2, 0]]
    right.plot(*border.T, color="black", lw=1.7)
    for k in (a, b):
        right.plot(*circle_on_simplex(k).T, color=COLORS[4], lw=1.3)
    left.set_xlabel(r"Dual multiplier $\lambda_1$", labelpad=22, fontsize=18)
    left.set_ylabel(r"Dual multiplier $\lambda_2$", labelpad=22, fontsize=18)
    left.set_zlabel(r"Dual multiplier $\lambda_3$", labelpad=22, fontsize=18)
    right.set_xlabel(r"$\bar\lambda_1$", labelpad=32, fontsize=18)
    right.set_ylabel(r"$\bar\lambda_2$", labelpad=32, fontsize=18)
    right.set_zlabel(r"$\bar\lambda_3$", labelpad=32, fontsize=18)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=COLORS[i],
            markersize=7,
            label=NAMES[i],
        )
        for i in range(5)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=5,
        fontsize=11,
        frameon=True,
        framealpha=0.95,
    )
    left.text2D(0.5, -0.06, "(a)", transform=left.transAxes, ha="center", fontsize=18)
    right.text2D(0.5, -0.06, "(b)", transform=right.transAxes, ha="center", fontsize=18)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.10, top=0.89, wspace=0.04)
    out = Path(args.output)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            str(out) + "." + ext,
            dpi=230,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)
    print("Saved", ", ".join(str(out) + "." + ext for ext in ("png", "pdf", "svg")))
    print(f"Illustrative thresholds: delta={d}, ksrd={a}, ksat={b}, gamma={g}; N=I3.")


if __name__ == "__main__":
    main()
