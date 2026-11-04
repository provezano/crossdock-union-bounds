"""Exploratory: exact v3 (CP-SAT + union cuts, g6cp.exact_v3, k = 12, 300 s) on the structured matched instances with
n1 = 30, to measure TRUE headroom (UB_EACDC - OPT)/OPT. Start: lo = LB* (out_struct), incumbent = FB-ILS run of 200k
evaluations (seed 1). usage: exact_struct.py OUT TL SHARD NSH"""
import sys, os, json
G6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fbils_census_exact_cpuc"); sys.path.insert(0, G6); sys.path.insert(0, f"{G6}/../exact_cpsat")
import g6cp, srch, g5lib as L
g2 = L.g2; out, tl, sh, ns = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
keys = sorted({json.load(open(f))["key"] for f in os.scandir(f"{G6}/out_match_struct") for f in [f.path]} )
keys = [k for k in keys if k.split("_")[1] == "30"]
for q, k in enumerate(keys):
    if q % ns != sh: continue
    fo = f"{out}/{k}.json"
    if os.path.exists(fo): continue
    S = json.load(open(f"{G6}/out_struct/{k}.json")); I = L.cdc.parse(f"{G6}/data_struct/{k}.dat")
    C = g2.all_constructive(I); kb = min(C, key=lambda c: C[c][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I)
    b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, 0.0, 200000, cap=120, seed=101, nev0=len(C), pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    cm, s1, s2 = g2.schedule_from(I, side, bo); hi = L.cdc.check_schedule(I, s1, s2)
    x = g6cp.exact_v3(I, S["lbstar"], hi, (list(map(int, s1)), list(map(int, s2))), tl, k=12)
    json.dump(dict(key=k, lbstar=S["lbstar"], ub_fb=float(hi), opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"]), open(fo, "w"))
    print(k, x["opt"], x["lo"], x["hi"], round(x["cpu"], 1), flush=True)
print("done")
