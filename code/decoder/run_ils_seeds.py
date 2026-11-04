"""ILS (step 2) from the best union-greedy constructive, CPU budget 60 s with trace; seeds. Args: seed shard nsh budget"""
import sys, os, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import cdc, g2
seed, shard, nsh = map(int, sys.argv[1:4]); bud = float(sys.argv[4])
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
L = json.load(open(f"{D}/../lower_bounds/sample105.json"))
od = f"{D}/out_ils"; os.makedirs(od, exist_ok=True)
for k, r in enumerate(L):
    if k % nsh != shard: continue
    fo = f"{od}/{r['g']}_{r['n1']}_{r['seed']}_s{seed}_b{int(bud)}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
    C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0])
    b, bo, tr = g2.ils(I, C[kb][1], C[kb][2], bud, seed=100 + seed)
    cm, s1, s2 = g2.schedule_from(I, C[kb][1], bo); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6
    json.dump(dict(best=b, trace=tr, seed=seed, constr=C[kb][0]), open(fo, "w"))
