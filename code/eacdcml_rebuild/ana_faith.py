"""Faithfulness summary of the frozen arms on repro210 + seed-to-seed spread (5 seeds: 169 and 1-4).
Deviations from the paper are expressed in units of the between-seed SD of the stratified group-mean gap."""
import json, glob, re
import numpy as np
import ana_repro as A
T = A.paper_table(); GROUPS = A.GROUPS; rng = np.random.default_rng(0); B = 10000
GM = {'2maquinas': 'm2', '4maquinas': 'm4', '10maquinas': 'm10', '[2-4]maquinas': 'u24', '[2-10]': 'u210'}
def load(d): return {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob(f"{d}/*.json")}
V2, CO = load("out_repro_v2"), load("out_repro_code")
cells = sorted({(r["g"], r["n1"]) for r in V2.values()}, key=lambda c: (GROUPS.index(c[0]), c[1]))
def strat(R, fn):
    vals = {c: np.array([fn(r) for r in R.values() if (r["g"], r["n1"]) == c]) for c in cells}
    pt = np.mean([v.mean() for v in vals.values()]); bs = np.zeros(B)
    for v in vals.values(): bs += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
    bs /= len(cells); return [round(float(x), 3) for x in (pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5))]
P_ML = np.mean([T[c]["ml"] for c in cells]); P_ST = np.mean([T[c]["eacdc"] for c in cells])
out = dict(paper_ml=P_ML, paper_static=P_ST)
out["v2_ml"] = strat(V2, lambda r: 100 * r["ml"]["gap"]); out["v2_static"] = strat(V2, lambda r: 100 * r["static"]["gap"])
out["v2_ml_minus_static_pp"] = strat(V2, lambda r: 100 * (r["ml"]["gap"] - r["static"]["gap"]))
out["code_ml"] = strat(CO, lambda r: 100 * r["mlc"]["gap"]); out["code_static"] = strat(CO, lambda r: 100 * r["gapy"]["gap"])
out["code_ml_minus_static_pp"] = strat(CO, lambda r: 100 * (r["mlc"]["gap"] - r["gapy"]["gap"]))
out["v2_ml_minus_code_ml_pp"] = strat({k: dict(V2[k], mlc=CO[k]["mlc"]) for k in V2}, lambda r: 100 * (r["ml"]["gap"] - r["mlc"]["gap"]))
w = lambda a, b: (sum(x < y for x, y in zip(a, b)), sum(x == y for x, y in zip(a, b)), sum(x > y for x, y in zip(a, b)))
out["WTL_v2_ml_vs_static"] = w([V2[k]["ml"]["best"] for k in V2], [V2[k]["static"]["best"] for k in V2])
out["WTL_code_ml_vs_static"] = w([CO[k]["mlc"]["best"] for k in CO], [CO[k]["gapy"]["best"] for k in CO])
out["cpu"] = dict(v2_ml=float(np.mean([r["ml"]["cpu"] for r in V2.values()])), v2_st=float(np.mean([r["static"]["cpu"] for r in V2.values()])),
                  code_ml=float(np.mean([r["mlc"]["cpu"] for r in CO.values()])), code_st=float(np.mean([r["gapy"]["cpu"] for r in CO.values()])))
# seed spread (machine-B): seeds 1-4 from out_seeds, seed 169 from gapy_repro_*.json and out_diag_code_es0
S = [json.load(open(f)) for f in glob.glob("out_seeds/*.json")]
G169 = {}
for f in glob.glob("gapy_repro_*.json"):
    for r in json.load(open(f)):
        m = re.search(r"Dados/(.+?)/(\d+)jobs/(\d+)\.dat", r["inst"]); G169[f"{GM[m.group(1)]}_{m.group(2)}_{m.group(3)}"] = r["best"]
D169 = load("out_diag_code_es0")
key2cell = {k: (r["g"], r["n1"]) for k, r in V2.items()}; lb = {k: r["lb_pch"] for k, r in V2.items()}
def grand(gaps):  # gaps: key -> gap%
    return np.mean([np.mean([gaps[k] for k in gaps if key2cell[k] == c]) for c in cells])
spread = {}
for arm in ("gapy", "v2"):
    per_seed = {}
    for s in sorted({x["seed"] for x in S}):
        per_seed[s] = grand({x["key"]: 100 * x[arm]["gap"] for x in S if x["seed"] == s})
    per_seed[169] = grand({k: 100 * (G169[k] - lb[k]) / G169[k] for k in V2}) if arm == "gapy" else grand({k: 100 * D169[k]["static"]["gap"] for k in V2})
    v = np.array(list(per_seed.values()))
    inst_sd = np.mean([np.std([100 * x[arm]["gap"] for x in S if x["key"] == k], ddof=1) for k in V2])
    spread[arm] = dict(per_seed={str(k): round(float(x), 3) for k, x in per_seed.items()}, sd_grand=float(v.std(ddof=1)),
                       range_grand=float(v.max() - v.min()), mean_per_instance_seed_sd=float(inst_sd))
out["seed_spread_repro210"] = spread
sd = spread["gapy"]["sd_grand"]; sdi = spread["gapy"]["mean_per_instance_seed_sd"]
# paper grand mean averages ~299 instances per cell: its seed SD ~ per-instance SD / sqrt(35*299) (independence approx)
out["paper_seed_sd_est"] = float(sdi / np.sqrt(35 * 299))
out["deviation_in_seed_sd"] = {k: round((out[k][0] - (P_ML if "ml" in k else P_ST)) / sd, 2) for k in ("v2_ml", "v2_static", "code_ml", "code_static")}
out["deviation_pp"] = {k: round(out[k][0] - (P_ML if "ml" in k else P_ST), 3) for k in ("v2_ml", "v2_static", "code_ml", "code_static")}
print(json.dumps(out, indent=1)); json.dump(out, open("faith_analysis.json", "w"), indent=1)
