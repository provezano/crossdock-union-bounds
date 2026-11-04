import json, glob, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); G1 = f"{D}/../lower_bounds"
def g1lb(key):
    v = []
    for d in ("out_s105_b4", "out_inexact_b60", "out_cert"):
        f = f"{G1}/{d}/{key}.json"
        if os.path.exists(f):
            r = json.load(open(f)); v += [r[k] for k in r if (k.startswith("lb") or k in ("cp_lb",)) and isinstance(r[k], (int, float))]
    return max(v) if v else 0
def boot(x, B=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x); m = rng.choice(x, (B, len(x))).mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
def wtl(a, b): a, b = np.asarray(a), np.asarray(b); return int((a < b).sum()), int((a == b).sum()), int((a > b).sum())
res = {}
for st, meths in (("A", ["constr", "ils10", "lns10", "cp10"]), ("B", ["ils60"])):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/out_{st}/*.json"))]
    if not R: continue
    for r in R:
        r["constr"] = min(r["constr"].values()); key = f"{r['g']}_{r['n1']}_{r['seed']}"
        r["lb"] = max(g1lb(key), r.get("cp10_lb", 0))
        r["best"] = min([r["ub_hgr"]] + [r[m] for m in meths])
        assert r["lb"] <= r["best"] + 1e-6, key
    hg = np.array([r["ub_hgr"] for r in R]); lb = np.array([r["lb"] for r in R])
    out = dict(n=len(R), gap_hgr_vs_lb=boot((hg - lb) / lb))
    for m in meths:
        u = np.array([r[m] for r in R], float)
        out[m] = dict(rel_vs_hgr=boot((u - hg) / hg), gap_vs_lb=boot((u - lb) / lb), wtl_vs_hgr=wtl(u, hg))
    for n1 in sorted({r["n1"] for r in R}):
        S = [r for r in R if r["n1"] == n1]; m = meths[min(1, len(meths)-1)]
        out[f"n1={n1}:{m}"] = float(np.mean([(r[m] - r["ub_hgr"]) / r["ub_hgr"] for r in S]))
    for g in sorted({r["g"] for r in R}):
        S = [r for r in R if r["g"] == g]; m = meths[min(1, len(meths)-1)]
        out[f"{g}:{m}"] = float(np.mean([(r[m] - r["ub_hgr"]) / r["ub_hgr"] for r in S]))
    if st == "A":
        import collections; out["constr_best_counts"] = collections.Counter(r["constr_best"] for r in R).most_common()
        names = R[0]["constr"] if False else None
    res[st] = out
json.dump(res, open(f"{D}/analysis.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
