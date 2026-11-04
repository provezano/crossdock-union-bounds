"""structured-instance experiment (preregistered; see the paper): per structured instance, ILS 10 s seed 1 (UB),
cheap LBs (PCH, Chen-Song as in HgR's code, e2, load), LB_U*-det forward and reversed (50 steps, 0.5 det s per
knapsack), energetic reasoning ER (er.py logic, memory-chunked), and RER (ER + shaving; exploratory, n1 <= 100,
30 CPU s cap). usage: struct_run.py LIST OUT SHARD NSH"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__))
for p in (D, f"{D}/../exact_cpsat", f"{D}/../eacdc_public/code"): sys.path.insert(0, p)
import g5lib as L, g5det, er, er_shave, fonseca_port  # noqa
import numpy as np  # noqa
er._stage = er_shave.stage_chunked
cdc, g1 = L.cdc, L.g1

def main():
    lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
    for q, r in enumerate(json.load(open(lst))):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}.json"
        if os.path.exists(fo): continue
        r["path"] = r["path"] if os.path.exists(r["path"]) else os.path.join(os.path.dirname(os.path.abspath(lst)), os.path.basename(r["path"])); I = cdc.parse(r["path"]); rec = dict(r, py=platform.python_version(), host=platform.node())
        ub, sched, cpu, tr, constr = L.warm_ub(I, 10.0, seed=1)
        rec.update(ub_ils10=ub, ub_ils4=min([v for c, v in tr if c <= 4.0 + 1e-9] or [None]), constr=constr)
        t = time.process_time()
        LBs, LBr, _, _ = fonseca_port.lbs_lbr(I["n1"], I["n2"], I["m1"], I["m2"], I["p1"].tolist(), I["p2"].tolist(), I["prec"].astype(int).tolist())
        kl = dict(lb_pch=cdc.lb_pch(I), lb_cs=max(LBs, LBr), lb_e2=cdc.lb_stage2_energy(I), lb_load=cdc.lb_load(I))
        rec.update(kl, cpu_cheap=time.process_time() - t); lo = max(kl.values())
        t = time.process_time(); u = g5det.lb_union_det(I, lo, det_tl=0.5, max_steps=50); rec.update(lb_u1=u["lb"], u1_raw=u["raw"], u1_exact=u["exact"], cpu_u1=time.process_time() - t)
        t = time.process_time(); v = g5det.lb_union_det(g1.reverse(I), 0.0, det_tl=0.5, max_steps=50); rec.update(lb_rev=v["lb"], rev_raw=v["raw"], rev_exact=v["exact"], cpu_rev=time.process_time() - t)
        rec["lbstar"] = max(rec["lb_u1"], rec["lb_rev"])
        t = time.process_time(); e, ne, _ = er.lb_er(I, lo, ub); rec.update(lb_er=float(e), cpu_er=time.process_time() - t)
        if I["n1"] <= 100:
            t = time.process_time(); e2, _, _ = er_shave.lb_rer(I, max(lo, e), ub, budget=30.0); rec.update(lb_rer=float(e2), cpu_rer=time.process_time() - t)
        assert rec["lbstar"] <= ub + 1e-9, (r["key"], rec["lbstar"], ub)
        json.dump(rec, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo); print(r["key"], rec["lbstar"], ub, flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
