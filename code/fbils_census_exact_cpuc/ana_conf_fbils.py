"""Prereg ILS v2 analysis: E1 v2 vs EACDC (CPU), E3 v2 vs EACDC (evaluations), E5 v2 vs v1 (CPU), E6 v2 vs v1 (evaluations);
one-sided sign-flip (100k), Holm, bootstrap CI95 (10k), W/T/L, by n1 and group; IG as secondary. usage: ana_conf_fbils.py OUT [TXT]"""
import json, glob, sys, collections, numpy as np
R = collections.defaultdict(list)
for fp in glob.glob(sys.argv[1] + "/*.json"):
    r = json.load(open(fp)); R[r["key"]].append(r)
keys = sorted(R); rng = np.random.default_rng(0); out = []; P = lambda s: out.append(s)
def d(k, a, m, b, mb):
    rs = R[k]; return np.mean([(r[a][m] - (r[b][mb] if b != "eacdc" else r["eacdc"]["best"])) / (r[b][mb] if b != "eacdc" else r["eacdc"]["best"]) * 100 for r in rs])
def test(x):
    obs = x.mean(); s = rng.choice([-1, 1], size=(100000, len(x))); null = (s * x).mean(1)
    p = (np.sum(null <= obs) + 1) / (100000 + 1)
    b = [rng.choice(x, len(x)).mean() for _ in range(10000)]; return obs, p, np.quantile(b, [.025, .975])
E = {"E1 v2-EACDC cpu": ("v2", "v_cpu", "eacdc", None), "E3 v2-EACDC ev": ("v2", "v_ev", "eacdc", None),
     "E5 v2-v1 cpu": ("v2", "v_cpu", "v1", "v_cpu"), "E6 v2-v1 ev": ("v2", "v_ev", "v1", "v_ev")}
res = {}
P(f"instances {len(keys)}, runs {sum(len(v) for v in R.values())}, seeds/instance {collections.Counter(len(v) for v in R.values())}")
for name, (a, m, b, mb) in E.items():
    x = np.array([d(k, a, m, b, mb) for k in keys]); obs, p, ci = test(x); res[name] = (obs, p, ci, x)
ps = sorted(res.items(), key=lambda kv: kv[1][1]); adj = {}; run = 0
for i, (nm, v) in enumerate(ps):
    run = max(run, min(1, (len(ps) - i) * v[1])); adj[nm] = run
for nm, (obs, p, ci, x) in res.items():
    w = (x < -1e-9).sum(); l = (x > 1e-9).sum()
    ok = adj[nm] < 0.05 and ci[1] < 0
    P(f"{nm:18s} {obs:+.3f}% [{ci[0]:+.3f}, {ci[1]:+.3f}]  p={p:.1e} Holm={adj[nm]:.1e}  W/T/L {w}/{len(x)-w-l}/{l}  {'CONFIRMED' if ok else 'NOT CONFIRMED'}")
for nm in res:
    x = res[nm][3]; by = collections.defaultdict(list)
    for k, v in zip(keys, x): by[int(k.split('_')[1])].append(v)
    P(f"  {nm} by n1: " + " ".join(f"{n}:{np.mean(v):+.2f}" for n, v in sorted(by.items())))
    byg = collections.defaultdict(list)
    for k, v in zip(keys, x): byg[k.split('_')[0]].append(v)
    P(f"  {nm} by group: " + " ".join(f"{g}:{np.mean(v):+.2f}" for g, v in sorted(byg.items())))
v1c = np.array([d(k, "v1", "v_cpu", "eacdc", None) for k in keys]); v1e = np.array([d(k, "v1", "v_ev", "eacdc", None) for k in keys])
igc = np.array([d(k, "ig", "v_cpu", "eacdc", None) for k in keys]); ige = np.array([d(k, "ig", "v_ev", "eacdc", None) for k in keys])
P(f"secondary: v1-EACDC cpu {v1c.mean():+.3f}%  ev {v1e.mean():+.3f}%;  IG-EACDC cpu {igc.mean():+.3f}%  ev {ige.mean():+.3f}%;  v2-IG cpu {np.mean([d(k,'v2','v_cpu','ig','v_cpu') for k in keys]):+.3f}%")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 2: open(sys.argv[2], "w").write(txt + "\n")
