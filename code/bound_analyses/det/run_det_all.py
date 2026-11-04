"""#3a: deterministic LB_U*-det (g5det.lb_union_det, det_tl=0.5, max_steps=50; forward; best = max(lb_pch, lb_e2,
lb_load, lb_hgr) exactly as exact_cpsat run_det.py) on every benchmark instance, n1 >= 60 first. Read-only import of
exact_cpsat/lower_bounds/benchmark_data code. One JSON per instance (resumable). usage: run_det_all.py OUTDIR SHARD NSHARDS"""
import sys, os, json, socket, platform
D = os.path.dirname(os.path.abspath(__file__))
G5 = f"{D}/../../exact_cpsat"; sys.path.insert(0, G5)
import g5lib as L, g5det
out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
os.makedirs(out, exist_ok=True)
R = json.load(open(f"{D}/../../lower_bounds/allbounds.json"))
R.sort(key=lambda r: (-(r['n1'] >= 60), r['n1'], r['g'], r['seed']))
host = socket.gethostname(); py = platform.python_version()
for k, r0 in enumerate(R):
    if k % ns != sh: continue
    key = f"{r0['g']}_{r0['n1']}_{r0['seed']}"; fo = f"{out}/{key}.json"
    if os.path.exists(fo): continue
    r = L.g1rec(r0['g'], r0['n1'], r0['seed']); I = L.cdc.parse(L.path(r0['g'], r0['n1'], r0['seed']))
    lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
    x = g5det.lb_union_det(I, lo, det_tl=0.5, max_steps=50)
    rec = dict(key=key, g=r0['g'], n1=r0['n1'], lb_det=x['lb'], raw_det=x['raw'], exact_det=x['exact'], steps=x['steps'],
               cpu=x['cpu'], lb_cpu4=r['lb_u1'], exact_cpu4=r['u1_exact'], cpu4=r['u1_cpu'], lo=lo, host=host, py=py)
    json.dump(rec, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
