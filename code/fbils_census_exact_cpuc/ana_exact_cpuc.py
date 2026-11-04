"""Item 2: v3 (CP-SAT + reified union cuts) vs the step 1 arms on the same 300 instances / 300 s. Solved counts by cell,
paired times, final bounds on unsolved, LB* tightness and true gaps on the enlarged solved set. usage: ana_exact_cpuc.py [TXT]"""
import json, glob, sys, os, collections, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import solved_set
O = solved_set.load(); V = {}
for fp in glob.glob(f"{D}/out_exact_v3/*_tl300.json"):
    a = json.load(open(fp)); V[a["key"]] = a
out = []; P = out.append
ks = sorted(V); P(f"v3 results: {len(ks)}/300")
G = ["m2", "m4", "m10", "u24", "u210"]
for n1 in (20, 30):
    row = []
    for g in G:
        kk = [k for k in ks if O[k]["g"] == g and O[k]["n1"] == n1]
        row.append(f"{g} {sum(V[k]['opt'] is not None for k in kk)}/{sum(O[k]['prim_opt'] is not None for k in kk)}/{sum(O[k]['opt'] is not None for k in kk)} of {len(kk)}")
    kk = [k for k in ks if O[k]["n1"] == n1]
    P(f"n1={n1}: v3 / v2lbu / any-G1-arm solved: " + "; ".join(row) + f"  TOTAL {sum(V[k]['opt'] is not None for k in kk)}/{sum(O[k]['prim_opt'] is not None for k in kk)}/{sum(O[k]['opt'] is not None for k in kk)} of {len(kk)}")
for k in ks:
    if V[k]["opt"] is not None and O[k]["opt"] is not None: assert V[k]["opt"] == O[k]["opt"], k
    if V[k]["opt"] is not None: assert O[k]["best_lo"] <= V[k]["opt"] <= O[k]["best_hi"], k
    if O[k]["opt"] is not None: assert V[k]["lo"] <= O[k]["opt"], k
both = [k for k in ks if V[k]["opt"] is not None and O[k]["prim_opt"] is not None]
if both:
    r = np.array([V[k]["cpu"] / O[k]["prim_cpu"] for k in both])
    P(f"both solved (v3, v2lbu): {len(both)}; median time {np.median([V[k]['cpu'] for k in both]):.1f} vs {np.median([O[k]['prim_cpu'] for k in both]):.1f} s; geo-mean ratio {np.exp(np.log(r).mean()):.3f}; v3 faster on {np.mean(r<1)*100:.0f}%")
only3 = [k for k in ks if V[k]["opt"] is not None and O[k]["opt"] is None]; onlyG = [k for k in ks if V[k]["opt"] is None and O[k]["opt"] is not None]
P(f"solved only by v3: {len(only3)}; only by a G1 arm: {len(onlyG)} {onlyG}")
uns = [k for k in ks if V[k]["opt"] is None]
if uns:
    P(f"unsolved by v3 ({len(uns)}): residual (hi-lo)/hi v3 {np.mean([(V[k]['hi']-V[k]['lo'])/V[k]['hi']*100 for k in uns]):.2f}% vs G1 best {np.mean([(O[k]['best_hi']-O[k]['best_lo'])/O[k]['best_hi']*100 for k in uns]):.2f}%; "
      f"v3 final LB vs LB* +{np.mean([(V[k]['lo']-O[k]['lbu'])/O[k]['lbu']*100 for k in uns]):.2f}%; vs G1 best LB +{np.mean([(V[k]['lo']-O[k]['best_lo'])/O[k]['best_lo']*100 for k in uns]):.2f}%")
OPT = {k: V[k]["opt"] if V[k]["opt"] is not None else O[k]["opt"] for k in ks}; S = [k for k in ks if OPT[k] is not None]
P(f"enlarged solved set: {len(S)} (n1=20 {sum(O[k]['n1']==20 for k in S)}, n1=30 {sum(O[k]['n1']==30 for k in S)})")
for n1 in (20, 30):
    s = [k for k in S if O[k]["n1"] == n1]
    if not s: continue
    g = lambda f: np.mean([(f(k) - OPT[k]) / OPT[k] * 100 for k in s])
    P(f"  n1={n1}: LB*=OPT {sum(O[k]['lbu']==OPT[k] for k in s)}/{len(s)}; (OPT-LB*)/OPT {np.mean([(OPT[k]-O[k]['lbu'])/OPT[k]*100 for k in s]):.2f}%; "
      f"gaps to OPT: HgR {g(lambda k: O[k]['ub_hgr']):.2f}%  EACDC {g(lambda k: O[k]['ub_eacdc']):.2f}%  ILS4 {g(lambda k: O[k]['ub_ils4']):.2f}%  ILS10 {g(lambda k: O[k]['ub_ils10']):.2f}%")
    for gg in G:
        q = [k for k in s if O[k]["g"] == gg]
        if q: P(f"     {gg}: n={len(q)} (OPT-LB*)/OPT {np.mean([(OPT[k]-O[k]['lbu'])/OPT[k]*100 for k in q]):.2f}%")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
