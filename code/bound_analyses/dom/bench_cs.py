"""#4 on the benchmark: is the Chen-Song term ever binding? For every instance with n1 in {20, 30} (where the census walk
is exact), recompute LB_U* WITHOUT the Chen-Song term: walk started from max(PCH, e2, load) only (g1.lb_union_exact,
budget 20 s, tl 2 s), forward and reversed; compare max(fwd, rev) with ceil(max(L_CS fwd, rev)) = ceil(lb_hgr).
usage: bench_cs.py OUT.jsonl"""
import sys, os, json
D = os.path.dirname(os.path.abspath(__file__))
for p in ("benchmark_data", "lower_bounds"): sys.path.insert(0, f"{D}/../../{p}")
import cdc, g1, math
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
R = [r for r in json.load(open(f"{D}/../../lower_bounds/allbounds.json")) if r['n1'] in (20, 30)]
done = set()
if os.path.exists(sys.argv[1]): done = {json.loads(l)['key'] for l in open(sys.argv[1])}
with open(sys.argv[1], "a") as f:
    for r in R:
        key = f"{r['g']}_{r['n1']}_{r['seed']}"
        if key in done: continue
        I = cdc.parse(f"{D}/../../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat"); J = g1.reverse(I)
        out = dict(key=key, g=r['g'], n1=r['n1'], lb_hgr=r['lb_hgr'])
        for nm, X in (("f", I), ("r", J)):
            lo = max(cdc.lb_pch(X), cdc.lb_stage2_energy(X), cdc.lb_load(X))
            x = g1.lb_union_exact(X, lo, tl=2.0, budget=20.0)
            out[nm] = x['lb']; out[nm + "_exact"] = x['exact']
        f.write(json.dumps(out) + "\n"); f.flush()
