"""Per-evaluation CPU of the C searches measured INSIDE C (no ctypes per-call overhead): ILS v1-in-C (algo 0) and
ILS v2 (search_fbi, wfrac .25, pfbi 0, rounds 3: the confirm_fbils settings), eval budget N, cpu/nev. Same instances
as cx.py. Writes cx/cx2.json."""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__)); F = f"{D}/../.."
for p in ("benchmark_data", "decoder", "fbils_census_exact_cpuc"): sys.path.insert(0, f"{F}/{p}")
import numpy as np, cdc, g2, srch
FD = {'m4': '4maquinas', 'm10': '10maquinas'}; rows = []
for g in FD:
    for n1 in range(20, 81, 10):
        for sd in (17, 18, 19):
            I = cdc.parse(f"{F}/benchmark_data/data/Dados/{FD[g]}/{n1}jobs/{sd}.dat"); o = list(range(n1)); N = 200000
            t0 = time.process_time(); b, bo, tr, ne = srch.run(I, o, "ils", 0.0, N, cap=600, seed=1); t1 = time.process_time()
            b2, bo2, tr2, ne2 = srch.run_fbi(I, g2.reverse(I), o, 0.0, N, cap=600, seed=1, pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25); t2 = time.process_time()
            r = dict(g=g, n1=n1, n2=I["n2"], E=int(I["prec"].sum()), c_ils=(t1 - t0) / ne, c_v2=(t2 - t1) / ne2, nev=ne, nev2=ne2)
            rows.append(r); print(r, flush=True)
json.dump(rows, open(f"{D}/cx2.json", "w"))
