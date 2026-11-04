"""Ratio checks on tiny instances: PS-LPT (phase-separated LPT), non-idle LS with LPT order, union-greedy LS,
min(LS, PS). OPT by CP-SAT. Adversarial hill-climb over instances to probe worst ratios."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cdc, g2

def lpt_ms(p, m):
    import heapq
    h = [0] * m
    for x in sorted(p, reverse=True): heapq.heappush(h, heapq.heappop(h) + x)
    return max(h)

def ps_lpt(I):
    return lpt_ms(I["p1"].tolist(), I["m1"]) + (lpt_ms(I["p2"].tolist(), I["m2"]) if I["n2"] else 0)

def ls_lpt(I):
    return g2.Dec(I)(g2.orders_basic(I)["lpt"])

def ls_ug(I):
    return g2.Dec(I)(g2.union_greedy(I, 1.0))

def opt(I):
    r = cdc.cpsat(I, 20.0); assert r["status"] == "OPTIMAL"; return r["ub"]

def bound(I):
    return 8 / 3 - 1 / (3 * I["m1"]) - 1 / (3 * I["m2"])

def rand_inst(rng, nmax=6, mmax=3, pmax=9):
    n1, n2 = rng.integers(1, nmax + 1, 2); m1, m2 = rng.integers(1, mmax + 1, 2)
    P = rng.random((n2, n1)) < rng.uniform(0.1, 0.9)
    return dict(n1=int(n1), n2=int(n2), m1=int(m1), m2=int(m2), p1=rng.integers(1, pmax + 1, n1),
                p2=rng.integers(1, pmax + 1, n2), prec=P)

def mutate(I, rng, pmax=12):
    J = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in I.items()}
    u = rng.integers(3)
    if u == 0: J["p1"][rng.integers(J["n1"])] = rng.integers(1, pmax + 1)
    elif u == 1: J["p2"][rng.integers(J["n2"])] = rng.integers(1, pmax + 1)
    else: a, b = rng.integers(J["n2"]), rng.integers(J["n1"]); J["prec"][a, b] = ~J["prec"][a, b]
    return J

def graham_family(m, K=10):
    g = [x for k in range(2 * m - 1, m, -1) for x in (k, k)] + [m, m, m]
    p1 = np.array([x * K for x in g] + [1]); p2 = np.array([x * K for x in g])
    n1, n2 = len(p1), len(p2); P = np.zeros((n2, n1), bool); P[:, -1] = True
    return dict(n1=n1, n2=n2, m1=m, m2=m, p1=p1, p2=p2, prec=P)

if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    out = {}
    # 1) random: verify PS-LPT <= bound*OPT; record worst ratios
    worst = dict(ps=0, ls=0, ug=0, mn=0, mn3=0)
    for _ in range(300):
        I = rand_inst(rng); o = opt(I); a, b, c = ps_lpt(I), ls_lpt(I), ls_ug(I)
        assert a <= bound(I) * o + 1e-9, (I, a, o)
        worst["ps"] = max(worst["ps"], a / o); worst["ls"] = max(worst["ls"], b / o); worst["ug"] = max(worst["ug"], c / o)
        worst["mn"] = max(worst["mn"], min(a, b) / o); worst["mn3"] = max(worst["mn3"], min(a, b, c) / o)
    out["random300"] = worst
    # 2) adversarial hill-climb maximizing each ratio
    for name, f in (("ls", lambda I: ls_lpt(I)), ("mn3", lambda I: min(ps_lpt(I), ls_lpt(I), ls_ug(I))), ("ps", ps_lpt)):
        best = 0
        for rs in range(6):
            I = rand_inst(rng); cur = f(I) / opt(I)
            for _ in range(150):
                J = mutate(I, rng); v = f(J) / opt(J)
                if v >= cur: I, cur = J, v
            if cur > best: best, bi = cur, I
        out["adv_" + name] = dict(ratio=best, inst={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in bi.items()})
    # 3) tight family for PS-LPT
    fam = []
    for m in (2, 3, 4):
        I = graham_family(m); o = opt(I); fam.append(dict(m=m, opt=o, ps=ps_lpt(I), ls=ls_lpt(I), ug=ls_ug(I), ratio_ps=ps_lpt(I) / o, bound=bound(I)))
    out["graham_family"] = fam
    print(json.dumps(out, indent=1))
