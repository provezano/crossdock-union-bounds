"""Gap 6b: deterministic step-capped LB_U* (g5det.lb_union_det, det_tl = 0.5 deterministic s per knapsack solve,
max_steps = 50) on n1 >= 60: 10 instances per (group, n1) cell for n1 in {60, 70, 80} (150, rng 606), forward only,
best = max(lb_pch, lb_e2, lb_load, lb_hgr) as in step 1. Compared with the cached CPU-capped (4 s) lb_u1.
Run on two machines; identical outputs = machine independence.
usage: run_det.py OUT.json [K per cell]"""
import sys, os, json, glob, time, platform, socket
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np
import g5lib as L, g5det

out = sys.argv[1]; k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
R = json.load(open(f"{D}/../lower_bounds/allbounds.json")); rng = np.random.default_rng(606); rows = []
for n1 in (60, 70, 80):
    for g in L.GROUPS:
        cell = sorted(r['seed'] for r in R if r['g'] == g and r['n1'] == n1)
        for sd in rng.choice(cell, k, replace=False).tolist():
            r = L.g1rec(g, n1, sd); I = L.cdc.parse(L.path(g, n1, sd))
            lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
            x = g5det.lb_union_det(I, lo, det_tl=0.5, max_steps=50)
            rows.append(dict(key=f"{g}_{n1}_{sd}", lb_det=x['lb'], exact_det=x['exact'], steps=x['steps'], cpu=x['cpu'],
                             lb_cpu4=r['lb_u1'], exact_cpu4=r['u1_exact'], cpu4=r['u1_cpu']))
json.dump(dict(host=socket.gethostname(), py=platform.python_version(), rows=rows), open(out, "w"))
print("done", len(rows))
