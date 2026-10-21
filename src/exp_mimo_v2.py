import numpy as np, json, time, sys
from mimo_v2 import *

which = sys.argv[1]
if which == "minpower":
    levels = [0, 5, 10, 15, 20, 25]
    fn = feats_minpower
else:
    levels = [0, 5, 10, 15, 20]
    fn = feats_maxmin
dels = [0, 5, 10, 15, 20]
NT, NC = 12, 6
out = {"levels": levels, "dels": dels, "test": {}, "calib": {}}
t0 = time.time()
for split, seed0, R in (("calib", 200000, NC), ("test", 100000, NT)):
    for g in levels:
        for d in dels:
            rec = []
            for r in range(R):
                x = fn(seed0 + r, float(g), float(d))
                if x is not None:
                    rec.append(x)
            out[split][f"{g}_{d}"] = rec
        print(which, split, g, round(time.time() - t0), flush=True)
    json.dump(out, open(f"res_mimo_v2_{which}.json", "w"), default=float)
