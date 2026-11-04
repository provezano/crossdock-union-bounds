"""Analysis of g4_match runs. usage: ana_match.py OUTDIR [--g2check] [--lb LBDIR] [--prereg]
Per run: ILS value at the same run's full-EACDC CPU (CPU-matched) and at EACDC's evaluation counts
(evalPop calls only = conservative for ILS; evalPop + init schedules). Instance = mean over seeds."""
import sys, os, json, glob, collections, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from ana_lib import at_cpu, at_eval, wtl, boot, signflip_p, holm

out = sys.argv[1]; args = sys.argv[2:]
R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/*.json"))]
print(f"runs: {len(R)}; interpreters: {collections.Counter((r['env']['python'], r['env']['host']) for r in R)}")
per = collections.defaultdict(dict); flag = 0; noeval = 0; rows = []
for r in R:
    e, i = r["eacdc"], r["ils"]; B = e["cpu"]
    if "horizon" in i and B > i["horizon"]: flag += 1
    v_cpu = at_cpu(i["trace"], B)
    v_gp = at_eval(i["trace"], e["n_evalpop"]); v_all = at_eval(i["trace"], e["n_evalpop"] + e["n_init"])
    if i["nev"] < e["n_evalpop"] + e["n_init"]: noeval += 1
    per[r["key"]][r["seed"]] = dict(E=e["best"], cpu=v_cpu, gp=v_gp, all=v_all, B=B, first=r["first"],
                                    evE=e["n_evalpop"] + e["n_init"], evI_atB=max(t[1] for t in i["trace"] if t[0] <= B + 1e-9) if v_cpu else None)
    rows.append((r["key"], r["seed"], r["first"], B, e["n_init"], e["n_evalpop"]))
print(f"ILS-first runs where EACDC CPU > ILS horizon (ILS read at horizon, conservative): {flag}; runs where ILS made fewer evals than EACDC: {noeval}")
seeds = sorted({s for v in per.values() for s in v}); keys = [k for k, v in per.items() if len(v) == len(seeds)]
n1 = lambda k: int(k.split("_")[1])
def cmp(col, label, ks=None, quiet=False):
    ks = ks or keys
    A = np.array([np.mean([per[k][s][col] for s in seeds]) for k in ks]); E = np.array([np.mean([per[k][s]["E"] for s in seeds]) for k in ks])
    d = (A - E) / E * 100; m, lo, hi = boot(d); w, t, l = wtl(A, E)
    if not quiet: print(f"  {label:58s} n={len(ks):3d} {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]  W/T/L {w}/{t}/{l}")
    return d
print(f"== per-run matched ILS vs full EACDC (seeds {seeds}, {len(keys)} instances)")
d1 = cmp("cpu", "CPU-matched (ILS at the run's own EACDC CPU)")
d3 = cmp("gp", "eval-matched: EACDC evalPop calls (conservative)")
cmp("all", "eval-matched: evalPop + init schedules")
print("  by n1 (CPU-matched / eval-matched gp):")
for n in sorted({n1(k) for k in keys}):
    ks = [k for k in keys if n1(k) == n]
    a = cmp("cpu", "", ks, True); b = cmp("gp", "", ks, True)
    print(f"    n1={n}: {a.mean():+.2f} / {b.mean():+.2f} (n={len(ks)})")
# order effect: per run, ILS-first vs EACDC-first
dd = collections.defaultdict(list)
for k in keys:
    for s in seeds:
        x = per[k][s]; dd[x["first"]].append((x["cpu"] - x["E"]) / x["E"] * 100)
print("  run-level CPU-matched diff by arm order:", {f: (round(float(np.mean(v)), 3), len(v)) for f, v in dd.items()})
B = np.array([per[k][s]["B"] for k in keys for s in seeds]); ev = np.array([per[k][s]["evE"] for k in keys for s in seeds])
evi = np.array([per[k][s]["evI_atB"] or 0 for k in keys for s in seeds])
print(f"  EACDC CPU per run: mean {B.mean():.2f} s (median {np.median(B):.2f}, max {B.max():.2f}); evals EACDC mean {ev.mean():.0f}; ILS evals within the same CPU mean {evi.mean():.0f} (ratio {np.mean(evi / ev):.2f})")
if "--g2check" in args:
    G2 = f"{D}/../decoder/out_eacdc"; same = 0; worse = 0; n = 0; rat = []
    for k in keys:
        for s in seeds:
            f = f"{G2}/{k}_s{s}.json"
            if os.path.exists(f):
                o = json.load(open(f)); n += 1; same += abs(o["best"] - per[k][s]["E"]) < 1e-9; worse += per[k][s]["E"] > o["best"] + 1e-9; rat.append(per[k][s]["B"] / o["cpu"])
    rat = np.array(rat)
    print(f"== determinism vs G2 default EACDC runs: identical best {same}/{n}, wrapper value worse than GA-returned value in {worse} (wrapper = best ever evaluated, GA returns its final selBest, so wrapper <= GA); CPU(this window)/CPU(G2 window): median {np.median(rat):.2f}, p10 {np.percentile(rat,10):.2f}, p90 {np.percentile(rat,90):.2f}, max {rat.max():.2f}")
ps = {}
if "--lb" in args:
    LBD = args[args.index("--lb") + 1]
    L = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(f"{LBD}/*.json")}
    ks = sorted(L); ub = {}
    for k in ks:  # best UB seen by any arm/seed (validity check)
        if k in per: ub[k] = min(min(x["E"], x["cpu"] or np.inf, x["all"] or np.inf) for x in per[k].values())
    new = np.array([max(L[k]["lb_u1"], L[k]["lb_rev"]) for k in ks]); no = np.array([np.ceil(max(L[k]["lb_pch"], L[k]["lb_cs"]) - 1e-9) for k in ks])
    g = (new - no) / no * 100; m, lo, hi = boot(g)
    viol = sum(1 for k, v in zip(ks, new) if k in ub and v > ub[k] + 1e-9)
    print(f"== LB: max(LB_U*, rev) vs ceil(best non-own = max(PCH, Chen-Song)) n={len(ks)}: {m:+.2f}% [{lo:+.2f}, {hi:+.2f}]; quantiles 0/10/50/90/100: {np.round(np.percentile(g,[0,10,50,90,100]),2).tolist()}; LB > best UB: {viol}/{len(ub)}; LB < baseline: {int((g < -1e-9).sum())}")
    u1 = np.array([L[k]["lb_u1"] for k in ks]); print(f"   LB_U* alone: {np.mean((u1-no)/no*100):+.2f}%;  exact (fwd) {np.mean([L[k]['u1_exact'] for k in ks]):.2f}")
    for n in sorted({n1(k) for k in ks}): print(f"   n1={n}: {np.mean([x for k, x in zip(ks, g) if n1(k) == n]):+.2f}%")
    gap = [(ub[k] - v) / v * 100 for k, v in zip(ks, new) if k in ub]; print(f"   gap best-UB vs new LB: mean {np.mean(gap):.2f}%")
    ps["E2"] = signflip_p(g, "greater")
if "--prereg" in args:
    ps["E1"] = signflip_p(d1, "less"); ps["E3"] = signflip_p(d3, "less")
    names = sorted(ps); adj = holm([ps[x] for x in names])
    print("== prereg Holm:", {x: (f"p={ps[x]:.2e}", f"adj={a:.2e}") for x, a in zip(names, adj)})
