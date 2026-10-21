import numpy as np, json, time
from src.optimization.admm_mimo import *

conds = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
strategies = ["fixed", "5rho_or", "balance", "regime", "random"]
NEV = 20
NTUNE = 5
GRID = (10.0, 30.0, 60.0)
out = {}
for gdb, dl in conds:

    def mk(seed):
        rng = np.random.default_rng(seed)
        H0 = channels(rng, 0.0)
        ref = solve_socp(H0, 0.0)
        if ref is None:
            return None
        gref = 2 * np.linalg.norm(ref["w"])
        H = H0.copy()
        H[0] *= 10 ** (-dl / 20)
        return Prob(H, gdb, gref)

    # tune rho0 for the fixed-penalty baseline
    best = None
    for rho in GRID:
        its = []
        for sd in range(NTUNE):
            P = mk(500000 + sd)
            its.append(run(P, "fixed", rho, maxit=3000)["it"] if P else 3000)
        med = np.median(its)
        print("tune", gdb, dl, rho, med, flush=True)
        if best is None or med < best[0]:
            best = (med, rho)
    rho0 = best[1]
    res = {st: [] for st in strategies}
    for sd in range(NEV):
        P = mk(600000 + sd)
        if P is None:
            continue
        for st in strategies:
            r = run(P, st, rho0, maxit=3000, seed=sd)
            res[st].append(
                dict(
                    it=r["it"],
                    time=r["time"],
                    conv=r["conv"],
                    labs=[l for l in r["labs"]][:200],
                )
            )
        print(
            "cond",
            gdb,
            dl,
            "seed",
            sd,
            {st: res[st][-1]["it"] for st in strategies},
            flush=True,
        )
    out[f"{gdb}_{dl}"] = dict(rho0=rho0, res=res)
    json.dump(out, open("res_admm.json", "w"), default=float)
