"""#4 computational search. Generators: 'bench' (Cota rules: pair prob 1/2, lengths U[10,100], every truck connected,
n2 in [0.8 n1, 1.2 n1] rounded), 'free' (random density, lengths 1..K, doors 1..4).
Per instance: exact LB_U* (old cap, paper definition) and LB_U*+ (door-start cap) forward and reversed, L_CS fwd/rev,
OPT by CP-SAT (30 s). Records violations: L_CS > LB_U*, ceil(L_CS) > ceil(LB_U*), BLB = max(L_CS fwd, rev) >
LB* = max(LB_U* fwd, rev) (and ceiled), LB_U*+ < L_CS (must never happen), LB_U*+ > OPT (must never happen).
usage: search.py OUT.jsonl N SEED"""
import sys, json, math, numpy as np, bf
from fractions import Fraction as Fr
out, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(seed)
def gen(kind):
    if kind == "bench":
        n1 = int(rng.integers(2, 7)); n2 = max(1, min(7, int(round(rng.uniform(0.8, 1.2) * n1))))
        m1, m2 = (int(x) for x in rng.integers(1, 5, 2)); p1 = rng.integers(10, 101, n1); p2 = rng.integers(10, 101, n2)
        P = rng.random((n2, n1)) < 0.5
    else:
        n1, n2 = int(rng.integers(1, 7)), int(rng.integers(1, 8)); m1, m2 = (int(x) for x in rng.integers(1, 5, 2))
        K = int(rng.choice([1, 3, 10, 100])); p1 = rng.integers(1, K + 1, n1); p2 = rng.integers(1, K + 1, n2)
        P = rng.random((n2, n1)) < rng.random()
    for j in range(n2):
        if not P[j].any(): P[j, rng.integers(n1)] = True
    for i in range(n1):
        if not P[:, i].any(): P[rng.integers(n2), i] = True
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=p1.astype(np.int64), p2=p2.astype(np.int64), prec=P)
c = lambda x: math.ceil(x)
with open(out, "w") as f:
    for k in range(N):
        kind = "bench" if k % 2 == 0 else "free"
        I = gen(kind); R = bf.reverse(I)
        o_f, o_r = bf.lb_us(I), bf.lb_us(R); n_f, n_r = bf.lb_us(I, "new"), bf.lb_us(R, "new")
        cs_f, cs_r = bf.l_cs(I), bf.l_cs(R)
        opt = bf.opt_cpsat(I)
        rec = dict(k=k, kind=kind, n1=I["n1"], n2=I["n2"], m1=I["m1"], m2=I["m2"], old_f=float(o_f), old_r=float(o_r),
                   new_f=float(n_f), new_r=float(n_r), cs_f=float(cs_f), cs_r=float(cs_r), opt=opt,
                   cs_gt_old=bool(cs_f > o_f), cs_gt_old_ceil=bool(c(cs_f) > c(o_f)),
                   blb_gt_lbstar=bool(max(cs_f, cs_r) > max(o_f, o_r)), blb_gt_lbstar_ceil=bool(c(max(cs_f, cs_r)) > c(max(o_f, o_r))),
                   new_lt_cs=bool(n_f < cs_f), new_gt_maxold=bool(n_f > max(o_f, cs_f)),
                   new_gt_opt=bool(opt is not None and n_f > opt + 1e-9), old_gt_opt=bool(opt is not None and o_f > opt + 1e-9))
        if rec["cs_gt_old"]: rec["inst"] = dict(p1=I["p1"].tolist(), p2=I["p2"].tolist(), prec=I["prec"].astype(int).tolist())
        f.write(json.dumps(rec) + "\n"); f.flush()
