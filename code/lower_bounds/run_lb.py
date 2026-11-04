"""Per-instance cached run of the exact-breakpoint union LB. usage: run_lb.py LIST OUTDIR TL BUDGET SHARD NSHARDS"""
import sys, os, json, time, resource
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import cdc, g1
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
lst, out, tl, bud, sh, ns = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
os.makedirs(out, exist_ok=True)
R = json.load(open(lst))
for k, r in enumerate(R):
    if k % ns != sh: continue
    fo = f"{out}/{r['g']}_{r['n1']}_{r['seed']}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
    lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
    res = g1.lb_union_exact(I, lo, tl=tl, budget=bud)
    rv = g1.lb_union_exact(g1.reverse(I), 0.0, tl=tl, budget=float(os.environ.get('REV_BUDGET', bud)))
    r = dict(r, lb_rev=rv['lb'], rev_raw=rv['raw'], rev_exact=rv['exact'], rev_cpu=rv['cpu'], lb_u1=res['lb'], u1_raw=res['raw'], u1_ndfs=res['n_dfs'], lo_cheap=lo, u1_steps=res['steps'], u1_nopt=res['n_opt'], u1_exact=res['exact'], u1_cpu=res['cpu'],
             rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    json.dump(r, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
