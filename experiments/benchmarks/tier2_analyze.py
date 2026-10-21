import json, collections, itertools, numpy as np
from src.regimes.core import *

R = json.load(open("tier2_mm_raw.json"))


def lab(x, theta=None, csat=0.6, eps=1e-3, eta=0.05, ks=1.4, gI=1.3):
    th = dict(eta=eta, ksrd=ks, ksat=kappa_sat_rule(x["q_act"], csat), gI=gI)
    return classify_features(x["Lh"], x["th"], x["I"], th, eps), th


def table(sub):
    c = collections.Counter(lab(x)[0] for x in sub)
    return c


strict = [x for x in R if x["status"] == "classified"]
relaxed = [x for x in R if x["status"] == "relaxed"]
print("total", len(R), collections.Counter(x["status"] for x in R))
print("strict", dict(table(strict)), len(strict))
print("relaxed(non-strict)", dict(table(relaxed)), len(relaxed))
print("all solved & featured", dict(table(strict + relaxed)))
# sat reachable?
print("q_act strict", sorted(x["q_act"] for x in strict))
# margins
ms = []
for x in strict + relaxed:
    l, th = lab(x)
    ms.append(
        margin(x["Lh"], x["th"], x["I"], th, l)
        if l in ("unc", "srd", "int", "sat")
        else 0.0
    )
ms = np.array(ms)
print("m_e<eps share", np.mean(ms < 1e-3), "median m", np.median(ms))


# sensitivity (strict set)
def sens(sub):
    res = collections.defaultdict(list)
    for eta in (1e-3, 1e-2, 5e-2):
        for ks in (1.2, 1.4, 1.6, 1.8):
            for gI in (1.15, 1.3, 1.5):
                for cs in (0.6, 0.75, 0.9):
                    c = collections.Counter(
                        lab(x, eta=eta, ks=ks, gI=gI, csat=cs)[0] for x in sub
                    )
                    for k in ("unc", "srd", "int", "sat", "hyb"):
                        res[k].append(100 * c[k] / len(sub))
    return {k: (min(v), max(v)) for k, v in res.items()}


print("sens strict", sens(strict))
print("sens all", sens(strict + relaxed))
json.dump(
    dict(
        strict=dict(table(strict)),
        relaxed=dict(table(relaxed)),
        n_strict=len(strict),
        n_relaxed=len(relaxed),
        status=dict(collections.Counter(x["status"] for x in R)),
        sens_strict=sens(strict),
        sens_all=sens(strict + relaxed),
        share_hyb_margin=float(np.mean(ms < 1e-3)),
    ),
    open("tier2_summary.json", "w"),
)
