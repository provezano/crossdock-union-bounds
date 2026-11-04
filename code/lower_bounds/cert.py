"""Task 2: CP-SAT (60 s, 1 worker) with cmax >= LB_new as a redundant constraint; certificates at n1 = 20-30.
usage: cert.py SHARD NSHARDS"""
import sys, os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, cdc, g1
from ortools.sat.python import cp_model
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
sh, ns = int(sys.argv[1]), int(sys.argv[2]); os.makedirs(f"{D}/out_cert", exist_ok=True)
L = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/../benchmark_data/out/*_tl60.json"))] + \
    [json.load(open(f)) for f in sorted(glob.glob(f"{D}/../benchmark_data/out_union/*.json"))]
L = [r for r in L if r['n1'] in (20, 30)]
seen = set()
for k, r in enumerate(L):
    key = (r['g'], r['n1'], r['seed'])
    if key in seen or k % ns != sh: continue
    seen.add(key); fo = f"{D}/out_cert/{r['g']}_{r['n1']}_{r['seed']}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
    lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
    u = g1.lb_union_exact(I, lo, tl=5, budget=120)
    lbn = int(u['lb'])
    ubl, C1, C2 = cdc.list_ub(I, np.argsort(-I['p1'], kind='stable'))
    # CP-SAT with the LB as redundant constraint (re-implemented here to add cmax >= lbn)
    t = time.process_time()
    res = cdc.cpsat(I, 60.0, hint=(C1, C2), strong=True, horizon=int(min(r['ub_hgr'], ubl)), cmax_lb=lbn)  # horizon = known UB (valid)
    # second solve with the cut: rebuild via objective-free probe "cmax <= lbn feasible?" is the certificate check
    best_ub = min(res['ub'] if res['ub'] == res['ub'] else 1e18, r['ub_hgr'], ubl)
    cert_new = best_ub <= max(lbn, np.ceil(res['lb'])) + 1e-9
    probe = None
    if not cert_new and best_ub - lbn <= 3:  # cheap destructive probe at LB level
        p = cdc.cpsat(I, 20.0, horizon=lbn, strong=True, objective=False)
        probe = p['status']
        if p['status'] in ('FEASIBLE', 'OPTIMAL'): best_ub = min(best_ub, cdc.check_schedule(I, *p['sched'])); cert_new = best_ub <= lbn
    g0_lb = max(r.get('lb_union', 0), r.get('cp_lb', 0), r.get('lb_destr', 0), lo)
    out = dict(r, lb_new=lbn, lb_new_exact=u['exact'], lb_new_raw=u['raw'], cp_ub2=res['ub'], cp_lb2=res['lb'], best_ub=best_ub,
               cert_new=bool(cert_new), probe=probe, g0_lb=float(np.ceil(g0_lb - 1e-9)),
               cert_g0=bool(min(r['ub_hgr'], r.get('cp_ub', 1e18) if r.get('cp_ub', 1e18) == r.get('cp_ub', 1e18) else 1e18) <= np.ceil(g0_lb - 1e-9) + 1e-9),
               cpu=time.process_time() - t)
    if res['sched'] is not None: assert cdc.check_schedule(I, *res['sched']) == res['ub']
    json.dump(out, open(fo, 'w'))
