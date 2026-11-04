"""Time-matched arms on sample105, one process per (arm, shard). usage: run_arms.py ARM SHARD NSH
ARM: eacdcR10 (EACDC anytime w/ restarts, 10 s; truncation of its trace gives EACDC@4s)
     ma10     (memetic: EACDC 40% + ILS intensification, 10 s)
     ilslbp4  (ILS 4 s from the best constructive INCLUDING the LB-prefix orders; constructive time counted)"""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g3, cdc, g2
arm, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L = json.load(open(f"{D}/../lower_bounds/sample105.json"))
od = f"{D}/out_{arm}"; os.makedirs(od, exist_ok=True)
jobs = [(r, s) for s in range(1, 6) for r in L]
for k, (r, s) in enumerate(jobs):
    if k % ns != sh: continue
    fo = f"{od}/{r['g']}_{r['n1']}_{r['seed']}_s{s}.json"
    if os.path.exists(fo): continue
    if arm == "eacdcR10":
        o = g3.eacdc(r['g'], r['n1'], r['seed'], s, budget=10.0); o.pop("perm")
    elif arm == "ma10":
        o = g3.memetic(r['g'], r['n1'], r['seed'], s, 10.0)
    elif arm == "ilslbp4":
        I = cdc.parse(g3.path(r['g'], r['n1'], r['seed'])); t0 = time.process_time()
        kb, C = g3.best_constructive(I, with_lbp=True, lbp_budget=1.0); tc = time.process_time() - t0
        b, bo, tr = g3.ils_p(I, C[kb][1], C[kb][2], max(0.0, 4.0 - tc), seed=100 + s, t_offset=tc)
        cm, s1, s2 = g2.schedule_from(I, C[kb][1], bo); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6
        o = dict(best=b, trace=tr, start=kb, constr=C[kb][0], constr_cpu=tc,
                 lbp=min(C["lbp_f"][0], C["lbp_r"][0]), best_old=min(v[0] for kk, v in C.items() if not kk.startswith("lbp")))
    o["seed"] = s
    json.dump(o, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
print("done")
