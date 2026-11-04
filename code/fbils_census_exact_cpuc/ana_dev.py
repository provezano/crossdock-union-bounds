"""Dev summary: per arm, paired % vs EACDC (instance = mean over seeds), CPU- and evaluation-matched; CI95 bootstrap."""
import json, glob, sys, collections, numpy as np
R = collections.defaultdict(lambda: collections.defaultdict(list))
for fp in glob.glob(sys.argv[1] + "/*.json"):
    r = json.load(open(fp)); R[r["arm"]][r["key"]].append(r)
rng = np.random.default_rng(0)
def ci(x): b = [rng.choice(x, len(x)).mean() for _ in range(2000)]; return np.quantile(b, [.025, .975])
for arm, d in sorted(R.items()):
    for m in ("v_cpu", "v_ev"):
        x = np.array([np.mean([(r[m] - r["eacdc"]) / r["eacdc"] * 100 for r in rs]) for rs in d.values()])
        lo, hi = ci(x); w = (x < -1e-9).sum(); l = (x > 1e-9).sum()
        print(f"{arm:14s} {m:6s} n={len(x):3d} {x.mean():+.3f}% [{lo:+.3f}, {hi:+.3f}]  W/T/L {w}/{len(x)-w-l}/{l}")
