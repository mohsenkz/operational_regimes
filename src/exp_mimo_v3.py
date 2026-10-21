import numpy as np, json, time
from mimo_v3 import *

levels = [5, 10, 15, 20, 25]
dels = [0, 5, 10, 15, 20]
NT, NC = 8, 4
import os

out = (
    json.load(open("res_mimo_v3_rate.json"))
    if os.path.exists("res_mimo_v3_rate.json")
    else {"levels": levels, "dels": dels, "test": {}, "calib": {}}
)
t0 = time.time()
for split, seed0, R in (("calib", 200000, NC), ("test", 100000, NT)):
    for g in levels:
        if all(f"{g}_{d}" in out[split] for d in dels):
            continue
        for d in dels:
            rec = []
            for r in range(R):
                try:
                    x = feats_rate(seed0 + r, float(g), float(d))
                except Exception as e:
                    x = None
                if x is not None:
                    rec.append(x)
            out[split][f"{g}_{d}"] = rec
        print(split, g, round(time.time() - t0), flush=True)
        json.dump(out, open("res_mimo_v3_rate.json", "w"), default=float)
