"""Collect the step 1 300-instance exact results (all arms) -> per-instance best proven LB, best UB, OPT if any arm proved it."""
import json, glob, os, collections
D = os.path.dirname(os.path.abspath(__file__)); EX = f"{D}/../exact_cpsat/out_exact"
def load():
    B = {}; A = collections.defaultdict(dict)
    for fp in glob.glob(f"{EX}/*_base.json"):
        b = json.load(open(fp)); B[b["key"]] = b
    for fp in glob.glob(f"{EX}/*_tl300.json"):
        a = json.load(open(fp)); A[a["key"]][a["arm"]] = a
    out = {}
    for k, b in B.items():
        arms = A.get(k, {})
        opt = next((a["opt"] for a in arms.values() if a.get("opt") is not None), None)
        lo = max([b["lbu"]] + [a["lo"] for a in arms.values() if a.get("lo") is not None])
        hi = min([b["ub_ils10"]] + [a["hi"] for a in arms.values() if a.get("hi") is not None])
        prim = arms.get("v2lbu", {})
        out[k] = dict(b, opt=opt, best_lo=lo, best_hi=hi, prim_opt=prim.get("opt"), prim_cpu=prim.get("cpu"), arms=sorted(arms))
    return out
if __name__ == "__main__":
    o = load(); s = [k for k, v in o.items() if v["opt"] is not None]
    print(len(o), "instances;", len(s), "solved;", sum(v["prim_opt"] is not None for v in o.values()), "by primary arm")
