"""Paired faithfulness diagnostic: our EACDC-paper / EACDCml-ours vs the public GA.py (fast, identical output) on
the SAME reproduction instances; gap vs LB_PCH (/UB), stratified over cells."""
import json, glob, re, os, sys
import numpy as np
import ana_repro as A
T = A.paper_table(); GM = {'2maquinas': 'm2', '4maquinas': 'm4', '10maquinas': 'm10', '[2-4]maquinas': 'u24', '[2-10]': 'u210'}
G = {}
for f in sorted(glob.glob("gapy_repro_*.json")):
    for r in json.load(open(f)):
        m = re.search(r"Dados/(.+?)/(\d+)jobs/(\d+)\.dat", r["inst"]); G[f"{GM[m.group(1)]}_{m.group(2)}_{m.group(3)}"] = r
R = {}
for f in glob.glob("out_repro/*.json"):
    r = json.load(open(f))
    if r["key"] in G: R[r["key"]] = r
cells = sorted({(r["g"], r["n1"]) for r in R.values()}); rng = np.random.default_rng(0); B = 10000
def strat(fn):
    vals = {c: np.array([fn(k, r) for k, r in R.items() if (r["g"], r["n1"]) == c]) for c in cells}
    pt = np.mean([v.mean() for v in vals.values()]); bs = np.zeros(B)
    for v in vals.values(): bs += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    bs /= len(cells); return [round(float(x), 3) for x in (pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5))]
gp = lambda k, r: 100 * (G[k]["best"] - r["lb_pch"]) / G[k]["best"]
out = dict(n=len(R), cells=len(cells),
           gapy_gap=strat(gp), paper_eacdc=float(np.mean([T[c]["eacdc"] for c in cells])),
           paper_ml=float(np.mean([T[c]["ml"] for c in cells])),
           static_gap=strat(lambda k, r: 100 * r["static"]["gap"]), ml_gap=strat(lambda k, r: 100 * r["ml"]["gap"]),
           static_minus_gapy_pp=strat(lambda k, r: 100 * r["static"]["gap"] - gp(k, r)),
           ml_minus_gapy_pp=strat(lambda k, r: 100 * r["ml"]["gap"] - gp(k, r)),
           ml_minus_gapy_mksp_pct=strat(lambda k, r: 100 * (r["ml"]["best"] - G[k]["best"]) / G[k]["best"]),
           gapy_cpu_mean=float(np.mean([G[k]["cpu"] for k in R])))
out["by_n1"] = {n1: dict(gapy=round(float(np.mean([gp(k, r) for k, r in R.items() if r["n1"] == n1])), 2),
                         static=round(float(np.mean([100 * r["static"]["gap"] for r in R.values() if r["n1"] == n1])), 2),
                         ml=round(float(np.mean([100 * r["ml"]["gap"] for r in R.values() if r["n1"] == n1])), 2),
                         paper_st=round(float(np.mean([T[c]["eacdc"] for c in cells if c[1] == n1])), 2))
                for n1 in range(20, 90, 10)}
print(json.dumps(out, indent=1)); json.dump(out, open("gapy_paired_analysis.json", "w"), indent=1)
