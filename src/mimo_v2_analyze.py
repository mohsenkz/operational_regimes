import json, numpy as np, collections
from regimes_lib import *


def load(w):
    d = json.load(open(f"res_mimo_v2_{w}.json"))
    return d


def calibrate(recs, qmax):
    L = np.array([r["L"] for r in recs])
    th = np.array([r["th"] for r in recs])
    I = np.array([r["I"] for r in recs])
    ks = np.floor(np.percentile(th, 75)) + 0.5
    ks = min(max(ks, 2.5), qmax - 0.5)
    return dict(
        eta=float(np.percentile(L, 10)),
        ksrd=1.4,
        ksat=float(ks),
        gI=float(max(1.05, np.percentile(I, 75))),
    )


out = {}
import sys

for w, qmax in (
    (("rate", 9),) if len(sys.argv) > 1 else (("maxmin", 9), ("minpower", 8))
):
    d = json.load(open(f"res_mimo_v3_{w}.json")) if w == "rate" else load(w)
    ok = lambda r: r.get("res", 0) < (1e-4 if w == "rate" else 1e-3)
    cal = [r for c in d["calib"].values() for r in c if ok(r)]
    TH = calibrate(cal, qmax)
    test = {k: [r for r in v if ok(r)] for k, v in d["test"].items()}
    ntest = sum(len(v) for v in d["test"].values())
    nok = sum(len(v) for v in test.values())
    print(
        "=====",
        w,
        "calibrated Theta",
        {k: round(v, 3) for k, v in TH.items()},
        "| calib n",
        len(cal),
        "| test kept",
        nok,
        "of",
        ntest,
    )
    allr = [r for v in test.values() for r in v]
    print(
        " feature ranges: Lhat %.2f-%.2f  theta %.2f-%.2f  I %.2f-%.2f"
        % (
            min(r["L"] for r in allr),
            max(r["L"] for r in allr),
            min(r["th"] for r in allr),
            max(r["th"] for r in allr),
            min(r["I"] for r in allr),
            max(r["I"] for r in allr),
        )
    )
    if w == "maxmin":
        print(
            " power-constraint share of reaction mass: median %.2f, range %.2f-%.2f; max-min t* (dB) range %.1f-%.1f"
            % (
                np.median([r["share0"] for r in allr]),
                min(r["share0"] for r in allr),
                max(r["share0"] for r in allr),
                min(r["t_db"] for r in allr),
                max(r["t_db"] for r in allr),
            )
        )
    lv = d["levels"]
    dl = d["dels"]
    tot = collections.Counter()
    cellres = {}
    for g in lv:
        row = []
        for dd in dl:
            c = collections.Counter(
                classify_features(r["L"], r["th"], r["I"], TH, 1e-3)
                for r in test[f"{g}_{dd}"]
            )
            tot.update(c)
            n = sum(c.values())
            cellres[f"{g}_{dd}"] = {k: v for k, v in c.items()}
            row.append(
                "%s(%d%%)" % (max(c, key=c.get), 100 * max(c.values()) // max(n, 1))
                if n
                else "--"
            )
        print("  level %2d dB:" % g, "  ".join(row))
    print(" overall label counts:", dict(tot))
    out[w] = dict(
        theta=TH, cells=cellres, overall=dict(tot), n_test=nok, n_calib=len(cal)
    )
json.dump(
    out,
    open(
        "res_mimo_v3_summary.json" if len(sys.argv) > 1 else "res_mimo_v2_summary.json",
        "w",
    ),
    default=float,
)
