"""LBs on fresh instances (prereg E2). usage: lb_new.py OUTDIR SHARD NSH"""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__))
for p in ("benchmark_data", "lower_bounds", "eacdc_public/code"): sys.path.insert(0, f"{D}/../{p}")
from ortools.sat.python import cp_model  # noqa
import cdc, g1, fonseca_port
out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
for q, r in enumerate(json.load(open(f"{D}/list_new.json"))):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/{r['path']}")
    LBs, LBr, _, _ = fonseca_port.lbs_lbr(I["n1"], I["n2"], I["m1"], I["m2"], I["p1"].tolist(), I["p2"].tolist(), I["prec"].astype(int).tolist())
    kl = dict(lb_pch=cdc.lb_pch(I), lb_cs=max(LBs, LBr), lb_e2=cdc.lb_stage2_energy(I), lb_load=cdc.lb_load(I))
    lo = max(kl.values()); t = time.process_time()
    u = g1.lb_union_exact(I, lo, tl=1.0, budget=4.0); rv = g1.lb_union_exact(g1.reverse(I), 0.0, tl=1.0, budget=4.0)
    json.dump(dict(r, **kl, lb_u1=u["lb"], u1_exact=u["exact"], lb_rev=rv["lb"], rev_exact=rv["exact"], cpu=time.process_time() - t), open(fo, "w"))
print("done")
