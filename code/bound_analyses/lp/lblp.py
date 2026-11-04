"""#3b: LB_U^LP of Proposition lp, strongly polynomial and machine independent.

Step 1 (hull): the vertices of K_hat = upper concave envelope of {(p1(Pred A), p2(A))} are optimal closures of
  max p2(A) - lam * p1(Pred A); we find all of them by Eisner-Severance breakpoint search over lam (exact rationals),
  one max-flow (OR-Tools SimpleMaxFlow, push-relabel, integer capacities) per probe. Number of probes <= 2V - 1,
  V <= n1 + n2 + 1 (closures are nested in lam), so the time is O((n1 + n2) * maxflow(n1 + n2 + 2, n1 + n2 + |E|)).
  This replaces the single parametric max flow of Gallo-Grigoriadis-Tarjan by O(n) max flows: still strongly
  polynomial, simpler. Reuses the hull routine of lower_bounds g1.closure_hull (imported read-only), re-checked here.
Step 2 (evaluate): f_hat(t) = t + (P2 - min(K_hat(m1 t), m2 (t - rmin)^+))/m2. As in Thm breakpoints, f = max(g1, g2);
  g2 <= rmin + b; g1 is piecewise linear in t with breakpoints at the hull vertices, so
      LB_U^LP = max(rmin + b, max_{vertices (c_k, v_k), v_k < P2} [c_k/m1 + (P2 - v_k)/m2], c_full/m1).
  (the last term is the left limit at c_full where K_hat -> P2). O(V) arithmetic, exact rationals.
Optional integer variant (lp_int): K(c) <= floor(K_hat(c)) at every integer c (pseudo-polynomial in P1; reported
  separately, never called strongly polynomial).
The reported machine-independent value is ceil(max(L0, LB_U^LP)), L0 = max(L_CS (as reported), PCH, e2, load)."""
import sys, os, time
from fractions import Fraction as Fr
D = os.path.dirname(os.path.abspath(__file__))
for p in ("benchmark_data", "lower_bounds"):
    q = f"{D}/../../{p}"
    if q not in sys.path: sys.path.insert(0, q)
import numpy as np
import cdc, g1


def hull(I):
    return g1.closure_hull(I)


def lb_lp(I, V=None):
    """Exact LB_U^LP (Fraction) and the hull."""
    p1, p2, P = I["p1"], I["p2"], I["prec"]; m1, m2 = I["m1"], I["m2"]
    P2 = int(p2.sum()); h = cdc.heads(I)
    # exact rational heads
    hs = []
    for j in range(I["n2"]):
        pr = np.flatnonzero(P[j])
        hs.append(max(Fr(int(p1[pr].sum()), m1), Fr(int(p1[pr].max()))) if pr.size else Fr(0))
    rmin = min(hs)
    if V is None: V = hull(I)
    cfull = int(p1[P.any(axis=0)].sum())
    best = rmin + Fr(P2, m2)
    for c, v in V:
        if 0 < c <= cfull and v < P2:
            best = max(best, Fr(c, m1) + Fr(P2 - v, m2))
    if cfull > 0:
        best = max(best, Fr(cfull, m1))
    return best, V


def lb_lp_int(I, V):
    """Integer-capacity variant: term at integer c uses floor(K_hat(c)); same walk formula as the census."""
    p1, p2, P = I["p1"], I["p2"], I["prec"]; m1, m2 = I["m1"], I["m2"]
    P2 = float(p2.sum()); rmin = float(cdc.heads(I).min())
    full = int(p1[P.any(axis=0)].sum())
    if full == 0: return rmin + P2 / m2
    Va = np.array(V, float); c = np.arange(full); thi = (c + 1) / m1
    U = np.minimum(np.floor(np.interp(c, Va[:, 0], Va[:, 1]) + 1e-9), P2)
    W = np.minimum(U, m2 * np.maximum(thi - rmin, 0))
    f = np.where(U < P2, thi + (P2 - W) / m2, -np.inf)
    return float(max(rmin + P2 / m2, f.max()))


def run(I, L0):
    t0 = time.process_time(); lp, V = lb_lp(I); t1 = time.process_time()
    li = lb_lp_int(I, V); t2 = time.process_time()
    return dict(lp_raw=float(lp), lp=float(np.ceil(max(L0, float(lp)) - 1e-9)), lpint_raw=li,
                lpint=float(np.ceil(max(L0, li) - 1e-9)), nV=len(V), cpu_lp=t1 - t0, cpu_int=t2 - t1)
