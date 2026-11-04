"""#4 Dominance LB_U* vs Chen-Song (2009). Exact brute-force evaluation of the paper's DEFINITIONS on tiny instances
(no walk, no pruning, no external terms), in exact rational arithmetic:
  LB_U*  = max(a, p, q, sup{t + Phi(t) : Phi(t) > 0}), Phi(t) = min_{A in F(t)} max((P2 - min(p2(A), cap(t)))/m2,
           max_{j notin A} p2_j), F(t) = {A : p1(Pred A) <= m1 t, max_{Pred A} p1 <= t};
  cap(t) = m2 (t - rmin)^+                (paper, 'old')
  cap(t) = sum_{k<=m2} (t - h_(k))^+      (door-start cap, 'new': proposed fix)
  L_CS   = (sum of the m2 smallest heads + P2)/m2   (Chen & Song 2009; all heads if n2 < m2).
Sup computation: on each interval between consecutive feasibility thresholds tau_A = max(p1(Pred A)/m1, max p1)
the family F is fixed and t + Phi(t) is non-decreasing (cap has slope <= m2), so the sup is the left limit at the
next threshold if Phi > 0 there, else the zero z of Phi in the interval (found exactly: Phi is a min of max of
piecewise-linear pieces; we bisect on rationals to 1e-12 and then snap - only used for reporting)."""
from fractions import Fraction as Fr
import itertools
import numpy as np


def heads(I):
    P, p1, m1 = I["prec"], I["p1"], I["m1"]
    h = []
    for j in range(I["n2"]):
        pr = [i for i in range(I["n1"]) if P[j][i]]
        h.append(max(Fr(sum(int(p1[i]) for i in pr), m1), Fr(max(int(p1[i]) for i in pr))) if pr else Fr(0))
    return h


def l_cs(I):
    h = sorted(heads(I)); k = min(I["m2"], I["n2"])
    return (sum(h[:k]) + int(sum(I["p2"]))) / Fr(I["m2"])


def reverse(I):
    return dict(n1=I["n2"], n2=I["n1"], m1=I["m2"], m2=I["m1"], p1=I["p2"].copy(), p2=I["p1"].copy(),
                prec=I["prec"].T.copy())


def _sets(I):
    n2, P, p1, p2, m1 = I["n2"], I["prec"], I["p1"], I["p2"], I["m1"]
    S = []
    for mask in range(1 << n2):
        A = [j for j in range(n2) if mask >> j & 1]
        pr = set(i for j in A for i in range(I["n1"]) if P[j][i])
        tau = max(Fr(sum(int(p1[i]) for i in pr), m1), Fr(max([int(p1[i]) for i in pr], default=0)))
        rest = max([int(p2[j]) for j in range(n2) if not mask >> j & 1], default=0)
        S.append((tau, int(sum(int(p2[j]) for j in A)), rest))
    return S


def lb_us(I, cap="old"):
    m2 = I["m2"]; P2 = int(sum(I["p2"])); h = sorted(heads(I)); rmin = h[0] if h else Fr(0)
    a = Fr(int(sum(I["p1"])), I["m1"]); p = Fr(int(max(I["p1"]))); q = Fr(int(max(I["p2"])))
    if cap == "old":
        capf = lambda t: m2 * max(t - rmin, Fr(0))
    else:
        hk = h[:min(m2, len(h))]
        capf = lambda t: sum(max(t - x, Fr(0)) for x in hk)
    S = _sets(I)
    taus = sorted(set(s[0] for s in S))

    def phi(t, fam):
        return min(max((P2 - min(v, capf(t))) / Fr(m2), Fr(r)) for (_, v, r) in fam)
    best = max(a, p, q)
    for k, tk in enumerate(taus):
        fam = [s for s in S if s[0] <= tk]
        nxt = taus[k + 1] if k + 1 < len(taus) else None
        if nxt is not None:
            v = phi(nxt, fam)  # continuous in t within the interval -> left limit = value at nxt with this family
            if v > 0:
                best = max(best, nxt + v); continue
        # find sup over [tk, nxt) with phi > 0: phi non-increasing; zero z -> sup = z (limit)
        if phi(tk, fam) <= 0:
            continue
        lo, hi = tk, (nxt if nxt is not None else tk + Fr(P2 + 1))
        for _ in range(80):
            mid = (lo + hi) / 2
            if phi(mid, fam) > 0: lo = mid
            else: hi = mid
        best = max(best, lo + phi(lo, fam))
    return best


def opt_cpsat(I, tl=30.0):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../benchmark_data"))
    import cdc
    r = cdc.cpsat(I, tl)
    return r["ub"] if r["status"] == "OPTIMAL" else None
