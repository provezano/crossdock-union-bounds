"""#3b: LB_U^LP (strongly polynomial, machine independent) on benchmark instances, forward AND reversed, n1>=60 first.
Per instance: lp (= ceil(max(L0, LB_U^LP))), lpint (integer-capacity variant), both directions, CPU, hull size.
L0 = max(lb_pch, lb_e2, lb_load, lb_hgr) as in the census. usage: run_lp.py OUTDIR SHARD NSHARDS [MIN_N1]"""
import sys, os, json
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import lblp, g1, cdc
out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); mn = int(sys.argv[4]) if len(sys.argv) > 4 else 0
os.makedirs(out, exist_ok=True)
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
B = f"{D}/../../lower_bounds"
R = json.load(open(f"{B}/allbounds.json"))
R = [r for r in R if r['n1'] >= mn]; R.sort(key=lambda r: (-(r['n1'] >= 60), r['n1'], r['g'], r['seed']))
for k, r0 in enumerate(R):
    if k % ns != sh: continue
    key = f"{r0['g']}_{r0['n1']}_{r0['seed']}"; fo = f"{out}/{key}.json"
    if os.path.exists(fo): continue
    r = json.load(open(f"{B}/out_full/{key}.json"))
    I = cdc.parse(f"{D}/../../benchmark_data/data/Dados/{F[r0['g']]}/{r0['n1']}jobs/{r0['seed']}.dat")
    lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
    f = lblp.run(I, lo); b = lblp.run(g1.reverse(I), lo)
    rec = dict(key=key, g=r0['g'], n1=r0['n1'], lo=lo, lb_u1=r['lb_u1'], lb_rev=r['lb_rev'], u1_exact=r['u1_exact'],
               rev_exact=r['rev_exact'], **{f"f_{a}": v for a, v in f.items()}, **{f"r_{a}": v for a, v in b.items()})
    json.dump(rec, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
