"""step 2a: Fonseca, Nogueira & Ravetti (2019, EJOR) 2-dock F2|CD|Cmax set (500 public instances, authors' Drive,
copied to data_fonseca2dock/). Per instance: Chen & Lee LB1, Fonseca LB2 (to verify instance identity against the
paper's Table 4 means), LB_U* forward and reversed (m1 = m2 = 1), ILS UB (4 s), and for n <= 10 the exact optimum
(hybrid CP-SAT probing, 60 s). The paper reports only per-(group, n) means of its Lagrangian LB (LB_HgR, Table 4)
and per-(n, m) gaps (Table 3), so the LB comparison is on means.
usage: g2dock.py OUTDIR SHARD NSH"""
import sys, os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g5lib as L
import numpy as np
cdc, g1, g2 = L.cdc, L.g1, L.g2


def parse2(fp):
    t = open(fp).read().split("\n"); t = [x.split() for x in t if x.strip()]
    n, m = int(t[0][0]), int(t[1][0]); p1 = np.array(t[2][:n], np.int64); p2 = np.array(t[3][:m], np.int64)
    P = np.zeros((m, n), bool)
    for j in range(m):
        row = [int(x) for x in t[4 + j]]; k = row[0]; assert len(row) == k + 1
        P[j, row[1:]] = True
    return dict(n1=n, n2=m, m1=1, m2=1, p1=p1, p2=p2, prec=P)


def lb12(I):
    P = I["prec"]; p1, p2 = I["p1"], I["p2"]
    lb1 = p2.sum() + min(p1[P[j]].sum() for j in range(I["n2"]))
    S = P.T
    lb2 = p1.sum() + min(p2[S[i]].sum() for i in range(I["n1"]))  # inbound without successor -> 0 term
    return float(lb1), float(lb2)


def lbstar(I, budget):
    lo = max(cdc.lb_pch(I), cdc.lb_stage2_energy(I), cdc.lb_load(I))
    return g1.lb_union_exact(I, lo, tl=1.0, budget=budget)


def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(f"{D}/data_fonseca2dock/Grupo*/*/*.txt"))
    for q, fp in enumerate(files):
        if q % ns != sh: continue
        key = os.path.basename(os.path.dirname(os.path.dirname(fp))) + "_" + os.path.basename(fp)[:-4]
        fo = f"{out}/{key}.json"
        if os.path.exists(fo): continue
        I = parse2(fp); lb1, lb2 = lb12(I); t = time.process_time()
        u = lbstar(I, 10.0); rv = lbstar(g1.reverse(I), 10.0); cpu_lb = time.process_time() - t
        ub, sched, cpu_ub, _, _ = L.warm_ub(I, 4.0)
        rec = dict(key=key, n=I["n1"], m=I["n2"], lb1=lb1, lb2=lb2, lb_u1=u["lb"], u1_exact=u["exact"], lb_rev=rv["lb"],
                   rev_exact=rv["exact"], cpu_lb=cpu_lb, ub_ils4=ub, every_in_has_succ=bool(I["prec"].any(axis=0).all()))
        lb = max(u["lb"], rv["lb"], lb1, lb2)
        if I["n1"] <= 10 and lb < ub:
            x = L.exact_hybrid(I, lb, ub, sched, 60.0); rec.update(opt=x["opt"], ex_lo=x["lo"], ex_hi=x["hi"], ex_cpu=x["cpu"])
        elif lb >= ub:
            rec.update(opt=ub, ex_lo=lb, ex_hi=ub, ex_cpu=0.0)
        json.dump(rec, open(fo, "w"))
    print("done")


if __name__ == "__main__":
    main()
