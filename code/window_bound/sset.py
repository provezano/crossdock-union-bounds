"""Read-only loader of the solved cross-dock optima (step 1/step 5 arms + step 6 v3 arm). Never writes into exact_cpsat/fbils_census_exact_cpuc."""
import json, glob, os, collections
D = os.path.dirname(os.path.abspath(__file__))
EX5 = f"{D}/snap/out_exact"; EX6 = f"{D}/snap/out_exact_v3"   # snapshot (read-only copies of exact_cpsat/fbils_census_exact_cpuc outputs; see snap/SNAPSHOT_TIME.txt)
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
def path(g, n1, s): return f"{D}/../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{s}.dat"
def load():
    B = {}; A = collections.defaultdict(dict)
    for fp in glob.glob(f"{EX5}/*_base.json"):
        b = json.load(open(fp)); B[b["key"]] = b
    for fp in glob.glob(f"{EX5}/*_tl300.json") + glob.glob(f"{EX6}/*_tl300.json"):
        a = json.load(open(fp)); A[a["key"]][a["arm"]] = a
    out = {}
    for k, b in B.items():
        arms = A.get(k, {})
        opts = {a["opt"] for a in arms.values() if a.get("opt") is not None}
        assert len(opts) <= 1, (k, opts)
        opt = next(iter(opts)) if opts else None
        lo = max([b["lbu"]] + [a["lo"] for a in arms.values() if a.get("lo") is not None])
        hi = min([b["ub_ils10"]] + [a["hi"] for a in arms.values() if a.get("hi") is not None])
        sched = next((a["sched"] for a in arms.values() if a.get("opt") is not None and a.get("sched")), None)
        out[k] = dict(b, opt=opt, best_lo=lo, best_hi=hi, opt_sched=sched, arms=sorted(arms))
    return out
if __name__ == "__main__":
    import numpy as np
    O = load(); S = [v for v in O.values() if v["opt"] is not None]
    print(len(O), "instances;", len(S), "solved;", sum(v["opt_sched"] is not None for v in S), "with optimal schedule")
    for n1 in (20, 30):
        s = [v for v in S if v["n1"] == n1]
        print(n1, len(s), "LB*=OPT", sum(v["lbu"] == v["opt"] for v in s), "mean gap %.2f%%" % np.mean([(v["opt"]-v["lbu"])/v["opt"]*100 for v in s]),
              collections.Counter(v["g"] for v in s))
