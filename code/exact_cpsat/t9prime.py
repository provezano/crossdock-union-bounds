"""step 3a: check T9' -- every list schedule of the project decoder satisfies UB <= (3 - 1/max(m1, m2)) * LB_U*,
using the PROJECT's own decoder (g2.Dec: stage-1 list scheduling in order pi, stage-2 non-idle ERD/LPT) and LB code
(g1.lb_union_exact with best = max(lb_pch, lb_e2, lb_load), i.e. LB_U* of Thm 4; forward instance only, which is
the stricter check).
Part A: 200k random tiny instances (random order pi), two generators:
  'bench': every inbound has >= 1 successor and every outbound >= 1 predecessor (the benchmark's condition);
  'free' : outbound >= 1 predecessor only (inbound jobs may have no successor) -- checks whether the
           implemented LB_U* needs the explicit max p1 term.
Part B: all 10,500 benchmark instances, 3 orders (PCH, LPT, random), LB = cached forward lb_u1.
usage: t9prime.py OUT.json NTINY"""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g5lib as L
import numpy as np
cdc, g1, g2 = L.cdc, L.g1, L.g2


def lbstar(I):
    lo = max(cdc.lb_pch(I), cdc.lb_stage2_energy(I), cdc.lb_load(I))
    return g1.lb_union_exact(I, lo, tl=1.0, budget=5.0)


def tiny(rng, gen):
    n1, n2 = int(rng.integers(1, 8)), int(rng.integers(1, 8)); m1, m2 = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    K = int(rng.choice([1, 3, 10, 100]))
    p1 = rng.integers(1, K + 1, n1); p2 = rng.integers(1, K + 1, n2)
    P = rng.random((n2, n1)) < rng.random()
    for j in range(n2):
        if not P[j].any(): P[j, rng.integers(n1)] = True
    if gen == "bench":
        for i in range(n1):
            if not P[:, i].any(): P[rng.integers(n2), i] = True
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=p1.astype(np.int64), p2=p2.astype(np.int64), prec=P)


def main():
    out, ntiny = sys.argv[1], int(sys.argv[2])
    res = {}
    for gen in ("bench", "free"):
        rng = np.random.default_rng(20261102 if gen == "bench" else 20261103)
        worst = (0.0, None); worst_p = (0.0, None); nviol = nviol_p = 0; nexact = 0; t0 = time.process_time()
        for it in range(ntiny // 2):
            I = tiny(rng, gen); pi = rng.permutation(I["n1"]).tolist()
            ub = g2.Dec(I)(pi); lb = lbstar(I); nexact += lb["exact"]
            M = max(I["m1"], I["m2"]); bnd = 3 - 1 / M
            rel = ub / lb["lb"] / bnd
            lbp = max(lb["lb"], float(I["p1"].max()))
            relp = ub / lbp / bnd
            if rel > 1 + 1e-12: nviol += 1
            if relp > 1 + 1e-12: nviol_p += 1
            if rel > worst[0]: worst = (rel, dict(I={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in I.items()}, pi=pi, ub=ub, lb=lb["lb"]))
            if relp > worst_p[0]: worst_p = (relp, ub / lbp)
        res[gen] = dict(n=ntiny // 2, max_rel=worst[0], n_viol=nviol, max_rel_with_p=worst_p[0], n_viol_with_p=nviol_p,
                        frac_lb_exact=nexact / (ntiny // 2), worst=worst[1], cpu=time.process_time() - t0)
        print(gen, {k: v for k, v in res[gen].items() if k != "worst"}, flush=True)
    # Part B: benchmark
    rng = np.random.default_rng(7); rows = []; t0 = time.process_time()
    import glob
    for fj in sorted(glob.glob(f"{D}/../lower_bounds/out_full/*.json")):
        if True:
            if True:
                r0 = json.load(open(fj)); g, n1, sd = r0["g"], r0["n1"], r0["seed"]; fp = L.path(g, n1, sd)
                I = cdc.parse(fp); r = L.g1rec(g, n1, sd); Dd = g2.Dec(I); o = g2.orders_basic(I)
                ubs = [Dd(o["pch"]), Dd(o["lpt"]), Dd(rng.permutation(I["n1"]).tolist())]
                M = max(I["m1"], I["m2"]); lb = r["lb_u1"]
                rows.append((max(ubs) / lb, max(ubs) / lb / (3 - 1 / M), r["ub_hgr"] / lb, bool(I["prec"].any(axis=0).all())))
    A = np.array([x[:3] for x in rows])
    res["bench10500"] = dict(n=len(rows), max_ratio_LS=float(A[:, 0].max()), max_rel_LS=float(A[:, 1].max()),
                             n_viol=int((A[:, 1] > 1 + 1e-12).sum()), max_ratio_hgr=float(A[:, 2].max()),
                             all_inbound_have_succ=bool(all(x[3] for x in rows)), cpu=time.process_time() - t0)
    print("bench10500", res["bench10500"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
