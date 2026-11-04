"""Toolkit (d): oracle swap per component for WIB, on the solved instances that have a stored optimal schedule S*.
Rows: V1 / V2 with inferred windows; with S* heads (outbound: max pred completion in S*), S* tails, both; and
per-test ablations (flow only, no ER; stage-2 test only; stage-1 test only). Oracle rows assert LB <= OPT.
usage: wib_oracle.py OUT SHARD NSH"""
import sys; sys.dont_write_bytecode = True
import os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset, wib


def one_stage(C, T, h, q, which):
    W = wib.init_win(C.I, T, h, q)
    try: wib.propagate(C, W)
    except wib.Infeasible: return True
    if which == 2: return not (wib.flow_ok(W.r2, W.d2, C.p2, C.m2) and wib.er_ok(W.r2, W.d2, C.p2, C.m2))
    return not (wib.flow_ok(W.r1, W.d1, C.p1, C.m1) and wib.er_ok(W.r1, W.d1, C.p1, C.m1))


def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    O = sset.load(); keys = sorted(k for k, v in O.items() if v["opt"] is not None and v["opt_sched"] is not None)
    for qd, k in enumerate(keys):
        if qd % ns != sh: continue
        fo = f"{out}/{k}.json"
        if os.path.exists(fo): continue
        v = O[k]; I = lbx.parse(sset.path(v["g"], v["n1"], v["seed"])); C = wib.Ctx(I); opt = int(v["opt"]); lo, hi = int(v["lbu"]), int(v["best_hi"])
        h, q = lbx.heads_tails_bp(I); P = I["prec"]
        s1, s2 = np.array(v["opt_sched"][0]), np.array(v["opt_sched"][1]); C1 = s1 + I["p1"]
        rs = np.where(P, C1[None, :], 0).max(axis=1).astype(np.int64)
        qs = np.array([opt - s2[P[:, i]].min() if P[:, i].any() else 0 for i in range(I["n1"])], dtype=np.int64)
        hO, qO = np.maximum(h, rs), np.maximum(q, qs)
        r = dict(key=k, g=v["g"], n1=v["n1"], opt=opt, lbstar=lo)
        rows = {
            "v1": lambda T: wib.refute_v1(C, T, h, q, budget=30)[0],
            "v1_flowonly": lambda T: wib.refute_v1(C, T, h, q, budget=30, er=False)[0],
            "v1_dich": lambda T: wib.refute_v1(C, T, h, q, budget=30, mode="dich")[0],
            "noshave_s2": lambda T: one_stage(C, T, h, q, 2),
            "noshave_s1": lambda T: one_stage(C, T, h, q, 1),
            "O_v1_heads": lambda T: wib.refute_v1(C, T, hO, q, budget=30)[0],
            "O_v1_tails": lambda T: wib.refute_v1(C, T, h, qO, budget=30)[0],
            "O_v1_both": lambda T: wib.refute_v1(C, T, hO, qO, budget=30)[0],
            "O_s2_heads_noshave": lambda T: one_stage(C, T, hO, q, 2),
            "O_s1_tails_noshave": lambda T: one_stage(C, T, h, qO, 1),
        }
        for nm, f in rows.items():
            lb, n, cpu = lbx.destructive(f, lo, hi, budget=90); r[nm] = dict(lb=int(lb), cpu=round(cpu, 2))
            assert lb <= opt, ("VALIDITY/ORACLE", k, nm, lb, opt)
        # head deficit at T = OPT: inferred V1 windows (valid for every optimal schedule) vs S*
        W = wib.refute_v1(C, opt, h, q, budget=30)[1]
        r["head_deficit_rel"] = float(np.maximum(rs - W.r2, 0).sum() / max(1, rs.sum()))
        r["tail_deficit_rel"] = float(np.maximum(W.d1 - (opt - qs), 0).sum() / max(1, qs.sum()))
        json.dump(r, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo); print(k, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
