import numpy as np, json, time, collections
from mimo import *

TH = dict(eta=0.05, ksrd=1.4, ksat=5.5, gI=1.3)
EPS = 1e-3
gams = [0, 5, 10, 15, 20, 25]
dels = [0, 5, 10, 15, 20]
R = 20
out = {}
t0 = time.time()
for gi, g in enumerate(gams):
    for di, dl in enumerate(dels):
        cnt = collections.Counter()
        Fs = []
        lic = 0
        sc = 0
        offmin = []
        for r in range(R):
            rng = np.random.default_rng(
                100000 + r
            )  # same fading realisation across the grid
            H0 = channels(rng, 0.0)
            ref = solve_socp(H0, 0.0)
            gref = 2 * np.linalg.norm(ref["w"])
            H = H0.copy()
            H[0] *= 10 ** (-dl / 20)
            s = solve_socp(H, g)
            if s is None:
                cnt["infeasible/fail"] += 1
                continue
            L, th, I, q = feats(s["w"], s["lam"], s["Mk"], gref)
            lab = classify_features(L, th, I, TH, EPS)
            cnt[lab] += 1
            Fs.append((L, th, I, q))
            G = grads(s["w"], s["Mk"])
            J = np.vstack([G, s["E"]])
            lic += (
                np.linalg.matrix_rank(J, tol=1e-8 * np.linalg.norm(J, 2)) == J.shape[0]
            )
            sc += s["lam"].min() > 1e-6 * s["lam"].max()
            Nn = G / np.linalg.norm(G, axis=1)[:, None]
            N = Nn @ Nn.T
            offmin.append(N[~np.eye(K, dtype=bool)].min())
        out[f"{g}_{dl}"] = dict(
            counts=dict(cnt),
            med=np.median(np.array(Fs), axis=0).tolist() if Fs else None,
            licq=lic,
            sc=sc,
            n=len(Fs),
            offmin_med=float(np.median(offmin)) if offmin else None,
        )
        print(
            g,
            dl,
            dict(cnt),
            np.round(np.median(np.array(Fs), axis=0), 3).tolist() if Fs else None,
            "licq",
            lic,
            "sc",
            sc,
            round(time.time() - t0),
            flush=True,
        )
json.dump(out, open("res_mimo_map.json", "w"), default=float)
