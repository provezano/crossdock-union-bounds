"""Preregistered analysis. Unit = instance (mean over seeds of the paired % diff
(ILS - rival)/rival). Primary family E1-E5 (Holm, one-sided sign-flip); secondary E6, E7.
usage: ana_match.py OUTDIR"""
import sys, os, json, glob, collections
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
from ana_lib import at_cpu, at_eval, wtl, boot, signflip_p, holm

EP = {  # name: (rival, budget kind, alternative)
    "E1": ("ml", "cpu", "less"), "E2": ("ml", "evals", "less"), "E3": ("st", "cpu", "less"),
    "E4": ("mlc", "cpu", "less"), "E5": ("mlc", "evals", "less"), "E6": ("ml1", "cpu", "less")}
EP2 = {  # family F2 (amendment 2026-11-02): ILS v2 (frozen, fbils_frozen)
    "E9": ("ml", "cpu"), "E10": ("ml", "evals"), "E11": ("mlc", "cpu"), "E12": ("mlc", "evals")}


def main():
    out = sys.argv[1]; R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/*.json"))]
    flags = collections.Counter(); per = collections.defaultdict(lambda: collections.defaultdict(list)); n1of = {}
    for r in R:
        tr = r["ils"]["trace"]; V = r["rivals"]; k = r["key"]; n1of[k] = r["n1"]
        for a, x in V.items():
            if r["first"] == "ils" and x["cpu"] > r["ils"]["horizon"]: flags[f"horizon_exceeded_{a}"] += 1
            if r["ils"]["nev"] < x["evals"]: flags[f"ils_fewer_evals_{a}"] += 1
        for e, (a, kind, _) in EP.items():
            v = at_cpu(tr, V[a]["cpu"]) if kind == "cpu" else at_eval(tr, V[a]["evals"])
            if v is None: flags["ils_no_solution"] += 1; v = np.inf
            per[k][e].append((v - V[a]["best"]) / V[a]["best"]); per[k]["ils_" + e].append(v); per[k]["riv_" + e].append(V[a]["best"])
        tr2 = r["ils2"]["trace"]
        for a, x in V.items():
            if r["ils2"]["nev"] < x["evals"]: flags[f"ils2_fewer_evals_{a}"] += 1
        for e, (a, kind) in EP2.items():
            v = at_cpu(tr2, V[a]["cpu"]) if kind == "cpu" else at_eval(tr2, V[a]["evals"])
            if v is None: flags["ils2_no_solution"] += 1; v = np.inf
            per[k][e].append((v - V[a]["best"]) / V[a]["best"]); per[k]["ils_" + e].append(v); per[k]["riv_" + e].append(V[a]["best"])
        per[k]["E7"].append((V["ml"]["best"] - V["st"]["best"]) / V["st"]["best"])
        for a in V: per[k]["cpu_" + a].append(V[a]["cpu"]); per[k]["ev_" + a].append(V[a]["evals"])
    keys = sorted(per)
    res = dict(n_runs=len(R), n_inst=len(keys), seeds=sorted({r["seed"] for r in R}), flags=dict(flags),
               interpreters=[list(x) for x in collections.Counter((r["env"]["python"], r["env"]["host"]) for r in R).most_common()])
    ps = {}
    for e in list(EP) + ["E7"] + list(EP2):
        x = np.array([100 * np.mean(per[k][e]) for k in keys]); m, lo, hi = boot(x)
        p = signflip_p(x, "less") if e != "E7" else min(1.0, 2 * min(signflip_p(x, "less"), signflip_p(x, "greater")))
        ps[e] = p
        d = dict(mean_pct=m, ci95=(lo, hi), p=p,
                 by_n1={n: float(np.mean([x[i] for i, k in enumerate(keys) if n1of[k] == n])) for n in sorted(set(n1of.values()))})
        if e != "E7":
            d["WTL_inst"] = wtl([np.mean(per[k]["ils_" + e]) for k in keys], [np.mean(per[k]["riv_" + e]) for k in keys])
        res[e] = d
    adj = holm([ps[e] for e in ("E1", "E2", "E3", "E4", "E5")])
    res["holm"] = dict(zip(("E1", "E2", "E3", "E4", "E5"), adj))
    res["confirmed"] = {e: bool(res["holm"][e] < 0.05 and res[e]["ci95"][1] < 0) for e in res["holm"]}
    adj2 = holm([ps[e] for e in EP2]); res["holm_F2"] = dict(zip(EP2, adj2))
    res["F2_pass_before_gate"] = {e: bool(res["holm_F2"][e] < 0.05 and res[e]["ci95"][1] < 0) for e in EP2}
    res["F2_gate"] = "claimable only if ILS v2 passes its own preregistered confirmation"
    res["cpu_evals"] = {a: dict(cpu_mean=float(np.mean([c for k in keys for c in per[k]["cpu_" + a]])),
                                cpu_max=float(np.max([c for k in keys for c in per[k]["cpu_" + a]])),
                                evals_mean=float(np.mean([c for k in keys for c in per[k]["ev_" + a]])))
                        for a in ("ml", "st", "mlc", "ml1")}
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(f"{out.rstrip('/')}_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
