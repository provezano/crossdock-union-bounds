"""Preregistered analysis: E2 LB* vs ceil(best non-own), E4 LB* vs LB_ER
(all instances in STRUCT dir), E1/E3 matched ILS vs static EACDC (MATCH dir; CPU-matched / evalPop-matched; runs with
EACDC CPU > 600 s dropped per prereg). One-sided sign-flip 100k, Holm over {E1,E2,E3,E4}, instance bootstrap CI95 10k,
W/T/L tol 1e-9 rel; by family and n1; LB CPU by n1; exploratory RER and certified gap. usage: ana_struct.py STRUCT MATCH [TXT]"""
import sys, os, json, glob, math, collections, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
from ana_lib import at_cpu, at_eval
rng = np.random.default_rng(0); out = []; P = lambda s: out.append(s)
def test(x, side):  # side=+1: H1 mean>0; side=-1: H1 mean<0
    obs = x.mean(); s = rng.choice([-1, 1], size=(100000, len(x))); null = (s * x).mean(1)
    p = (np.sum(side * null >= side * obs) + 1) / 100001
    b = [rng.choice(x, len(x)).mean() for _ in range(10000)]; return obs, p, np.quantile(b, [.025, .975])
def wtl(x): return int((x > 1e-9).sum()), int((np.abs(x) <= 1e-9).sum()), int((x < -1e-9).sum())
S = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[1] + "/*.json"))]
P(f"structured instances analysed: {len(S)} (families {dict(collections.Counter(r['fam'] for r in S))})")
viol = [r["key"] for r in S if r["lbstar"] > r["ub_ils10"] + 1e-9]; P(f"LB* <= UB violations: {len(viol)}")
base = np.array([math.ceil(max(r["lb_pch"], r["lb_cs"]) - 1e-9) for r in S], float); ls = np.array([r["lbstar"] for r in S])
er = np.array([r["lb_er"] for r in S]); ub = np.array([r["ub_ils10"] for r in S])
x2 = (ls - base) / base * 100; x4 = (ls - er) / er * 100
res = {"E2 LB*-bestnonown": (x2, +1), "E4 LB*-LB_ER": (x4, +1)}
M = [json.load(open(f)) for f in sorted(glob.glob(sys.argv[2] + "/*.json"))] if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]) else []
if M:
    per = collections.defaultdict(list); drop = set()
    for r in M:
        e, i = r["eacdc"], r["ils"]
        if e["cpu"] > 600: drop.add(r["key"]); continue
        per[r["key"]].append(((at_cpu(i["trace"], e["cpu"]) - e["best"]) / e["best"] * 100, (at_eval(i["trace"], e["n_evalpop"]) - e["best"]) / e["best"] * 100))
    ks = sorted(k for k in per if k not in drop and len(per[k]) == 3)
    P(f"matched: runs {len(M)}, instances with 3 seeds {len(ks)}, dropped (EACDC CPU > 600 s) {sorted(drop)}")
    x1 = np.array([np.mean([a for a, b in per[k]]) for k in ks]); x3 = np.array([np.mean([b for a, b in per[k]]) for k in ks])
    res["E1 ILS-EACDC cpu"] = (x1, -1); res["E3 ILS-EACDC evalPop"] = (x3, -1)
T = {nm: test(x, sd) for nm, (x, sd) in res.items()}
ps = sorted(T, key=lambda n: T[n][1]); adj = {}; run = 0
for i, nm in enumerate(ps): run = max(run, min(1, (len(ps) - i) * T[nm][1])); adj[nm] = run
for nm, (x, sd) in res.items():
    obs, p, ci = T[nm]; ok = adj[nm] < 0.05 and (ci[0] > 0 if sd > 0 else ci[1] < 0) and (nm[:2] != "E2" or not viol)
    P(f"{nm:22s} n={len(x):3d} {obs:+.3f}% [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p:.1e} Holm={adj[nm]:.1e} >/=/< {wtl(x)} {'CONFIRMED' if ok else 'NOT CONFIRMED'}  (Holm over {len(res)})")
fam = lambda r: r["fam"] + (str(r.get("density", "")) if r["fam"] == "S" else "")
for nm, xx in (("E2", x2), ("E4", x4)):
    by = collections.defaultdict(list)
    for r, v in zip(S, xx): by[(r["cell"], r["n1"])].append(v)
    P(f"{nm} by cell,n1: " + " ".join(f"{c}/{n}:{np.mean(v):+.1f}" for (c, n), v in sorted(by.items())))
by = collections.defaultdict(list)
for r in S: by[r["n1"]].append(r)
P("LB CPU by n1 (mean s): cheap / LB_U fwd / LB_U rev / ER / RER(n1<=100)  | capped walk exact fwd,rev")
for n, rs in sorted(by.items()):
    m = lambda f: np.mean([r.get(f, np.nan) for r in rs])
    P(f"  n1={n:3d} n={len(rs):3d}: {m('cpu_cheap'):.3f} / {m('cpu_u1'):.2f} / {m('cpu_rev'):.2f} / {m('cpu_er'):.2f} / {m('cpu_rer'):.2f}   exact {np.mean([r['u1_exact'] for r in rs]):.2f},{np.mean([r['rev_exact'] for r in rs]):.2f}")
gap = (ub - ls) / ls * 100; byc = collections.defaultdict(list)
for r, g in zip(S, gap): byc[(r["fam"], r["n1"])].append(g)
P("exploratory certified gap (ILS10 - LB*)/LB* by family,n1: " + " ".join(f"{f}/{n}:{np.mean(v):.2f}%" for (f, n), v in sorted(byc.items())))
R = [r for r in S if "lb_rer" in r]
if R:
    a = np.array([(r["lbstar"] - r["lb_rer"]) / r["lb_rer"] * 100 for r in R]); P(f"exploratory LB* vs RER (n1<=100, n={len(R)}): {a.mean():+.2f}%  >/=/< {wtl(a)}; RER>ER on {sum(r['lb_rer'] > r['lb_er'] + 1e-9 for r in R)}; max(LB*,RER) vs LB*: {np.mean([(max(r['lbstar'], r['lb_rer']) - r['lbstar']) / r['lbstar'] * 100 for r in R]):+.3f}%")
bind = collections.Counter("fwd" if r["lb_u1"] > r["lb_rev"] + 1e-9 else ("rev" if r["lb_rev"] > r["lb_u1"] + 1e-9 else "tie") for r in S)
P(f"exploratory binding direction of LB*: {dict(bind)}")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 3: open(sys.argv[3], "w").write(txt + "\n")
