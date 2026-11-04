"""Exploratory analysis of the 10x-calibration sensitivity arm (model_table2_10x.pt). Not a prereg family.
(a) faithfulness on repro210 (seed 169): ml10 gap vs paper 23.17 and paired vs ml (1x) / static, from out_repro_v2;
(b) confirmation set: ILS v1/v2 traces from out_match read at ml10's CPU / evals (same machine, machine-C, but separate
    process); 'beyond' = ml10 budget exceeds the actual end of the ILS run (ILS then read at its end: conservative for ILS)."""
import os, sys, json, glob, collections
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
from ana_lib import at_cpu, at_eval, wtl, boot, signflip_p
PAPER_ML = 23.174285714285716
res = {}
# (a)
A = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob(f"{D}/out_repro_v2/*.json")}
S = {json.load(open(f))["key"]: json.load(open(f))["ml10"] for f in glob.glob(f"{D}/out_sens10x_repro/*.json")}
cells = collections.defaultdict(list)
for k in S: cells[(A[k]["g"], A[k]["n1"])].append(k)
def strat(fn, B=10000, seed=0):
    rng = np.random.default_rng(seed); C = list(cells.values())
    m = np.mean([np.mean([fn(k) for k in c]) for c in C])
    bs = [np.mean([np.mean([fn(k) for k in rng.choice(c, len(c))]) for c in C]) for _ in range(B)]
    return [float(m), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
res["repro_ml10_gap"] = strat(lambda k: 100 * S[k]["gap"], 2000)
res["repro_ml10_minus_paper_pp"] = res["repro_ml10_gap"][0] - PAPER_ML
res["repro_ml10_minus_ml1x_pp"] = strat(lambda k: 100 * (S[k]["gap"] - A[k]["ml"]["gap"]), 2000)
res["repro_ml10_minus_static_pp"] = strat(lambda k: 100 * (S[k]["gap"] - A[k]["static"]["gap"]), 2000)
res["repro_WTL_ml10_vs_ml1x"] = wtl([S[k]["best"] for k in S], [A[k]["ml"]["best"] for k in S])
res["repro_cpu_ml10_ml1x"] = [float(np.mean([S[k]["cpu"] for k in S])), float(np.mean([A[k]["ml"]["cpu"] for k in S]))]
# (b)
M = {(r["key"], r["seed"]): r for r in (json.load(open(f)) for f in glob.glob(f"{D}/out_match/*.json"))}
T = {(r["key"], r["seed"]): r["ml10"] for r in (json.load(open(f)) for f in glob.glob(f"{D}/out_sens10x_match/*.json"))}
per = collections.defaultdict(lambda: collections.defaultdict(list)); beyond = collections.Counter(); n1of = {}
for (k, s), x in T.items():
    r = M[(k, s)]; n1of[k] = r["n1"]
    for tag, il in (("v1", r["ils"]), ("v2", r["ils2"])):
        tr = il["trace"]; cend = il["cpu"]; eend = il["nev"]   # actual end of the ILS run (trace lists improvements only)
        if x["cpu"] > cend: beyond[f"{tag}_cpu"] += 1
        if x["evals"] > eend: beyond[f"{tag}_evals"] += 1
        for kind, v in (("cpu", at_cpu(tr, x["cpu"])), ("evals", at_eval(tr, x["evals"]))):
            per[k][f"{tag}_{kind}"].append((v - x["best"]) / x["best"]); per[k][f"ils_{tag}_{kind}"].append(v); per[k][f"riv_{tag}_{kind}"].append(x["best"])
    per[k]["ml10_vs_ml"].append((x["best"] - r["rivals"]["ml"]["best"]) / r["rivals"]["ml"]["best"])
    per[k]["ml10_vs_mlc"].append((x["best"] - r["rivals"]["mlc"]["best"]) / r["rivals"]["mlc"]["best"])
    per[k]["cpu"].append(x["cpu"]); per[k]["ev"].append(x["evals"])
keys = sorted(per)
for e in ("v1_cpu", "v1_evals", "v2_cpu", "v2_evals", "ml10_vs_ml", "ml10_vs_mlc"):
    v = np.array([100 * np.mean(per[k][e]) for k in keys]); m, lo, hi = boot(v)
    d = dict(mean_pct=m, ci95=(lo, hi), p_less=signflip_p(v, "less"),
             by_n1={n: float(np.mean([v[i] for i, k in enumerate(keys) if n1of[k] == n])) for n in sorted(set(n1of.values()))})
    if not e.startswith("ml10"): d["WTL_inst"] = wtl([np.mean(per[k]["ils_" + e]) for k in keys], [np.mean(per[k]["riv_" + e]) for k in keys])
    res["match_" + e] = d
res["match_beyond_trace_end"] = dict(beyond); res["n_runs"] = len(T)
res["ml10_cpu_mean"] = float(np.mean([c for k in keys for c in per[k]["cpu"]])); res["ml10_evals_mean"] = float(np.mean([c for k in keys for c in per[k]["ev"]]))
res["hosts"] = collections.Counter(json.load(open(f))["env"]["host"] for f in glob.glob(f"{D}/out_sens10x_*/*.json")).most_common()
json.dump(res, open(f"{D}/sens10x_analysis.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
