import sys
import numpy as np, pandas as pd

path = sys.argv[1]
d = pd.read_csv(path)
d["ct"] = d.crowd_disc * d.ticks
rng = np.random.default_rng(0)

def boot(g, n=2000):
    ct, tk, ep = g.ct.to_numpy(), g.ticks.to_numpy(), g.crowd_disc.to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    pooled = ct[idx].sum(1) / tk[idx].sum(1)
    return np.percentile(pooled, [2.5, 97.5]), np.percentile(ep[idx].mean(1), [2.5, 97.5])

rows = []
for name, g in d.groupby("policy"):
    (plo, phi), (elo, ehi) = boot(g)
    rows.append({"policy": name, "n_eps": len(g),
                 "crowd_pooled": g.ct.sum() / g.ticks.sum(),
                 "pooled_95ci": "%.3f-%.3f" % (plo, phi),
                 "crowd_ep_mean": g.crowd_disc.mean(),
                 "ep_mean_95ci": "%.3f-%.3f" % (elo, ehi),
                 "slow_ep_mean": g.slow.mean(),
                 "mean_len": g.ticks.mean(),
                 "success": (g.outcome == "success").mean()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
