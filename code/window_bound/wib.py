"""Window-inference bound (WIB), v2 of the LB bet (2026-11-02). Destructive: for a trial makespan T, infer windows
conditional on Cmax <= T and refute T when some window becomes empty or a relaxation is infeasible.

Windows (integer):  inbound i in stage 1 runs inside [r1_i, d1_i];  outbound j in stage 2 runs inside [r2_j, d2_j].
Initial: r1 = 0, d1 = T - q (bin-packing tails), r2 = h (bin-packing heads), d2 = T.

Components (proofs in the paper supplement, Section S5):
  P   propagate(): cross-stage head/tail rules (Lemma W1) to a fixpoint.
  F   flow_ok():   exact feasibility of P|pmtn, r_j, d_j| (Horn 1974 network) per stage (relaxation).
  E   er_ok():     energetic reasoning for P|r_j, d_j| per stage (non-preemptive, Baptiste-Le Pape-Nuijten).
  L   JLP:         LP relaxation of the split-cut joint test J(T, t) (Theorem J) with window forcing and caps; used
                   (a) as a refutation test and (b) for LP probing (shaving), Lemma W3.
  S   shave():     shaving of the four window sides with hypothesis tests built from P+F+E (Lemma W2), to a fixpoint.
Every refutation is a proof by itself (no monotonicity is needed for validity). LP refutations use an
outward-relaxed right-hand side (+EPS on every inequality), so a floating-point tolerance cannot fake infeasibility.
"""
import sys
sys.dont_write_bytecode = True
import os, time, math
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
if D not in sys.path: sys.path.insert(0, D)
import lbx  # noqa: E402
from ortools.graph.python import max_flow  # noqa: E402

BIG = 1 << 40
PROF = {}   # id(I) -> (U2, U1) when door caps are on (v6)


class Infeasible(Exception):
    pass


# ----------------------------------------------------------------------------- single-stage tests

def flow_ok(r, d, p, m):
    """Exact P|pmtn, r_j, d_j| feasibility (Horn network), vectorised arc construction."""
    n = len(p)
    if n == 0: return True
    if np.any(r + p > d): return False
    if m >= n: return True
    pts = np.unique(np.concatenate([r, d])); L = len(pts) - 1
    a, b = pts[:-1], pts[1:]
    ok = (r[:, None] <= a[None, :]) & (d[:, None] >= b[None, :])
    jj, kk = np.nonzero(ok)
    f = max_flow.SimpleMaxFlow(); S, T = n + L, n + L + 1
    tails = np.concatenate([np.full(n, S), jj, n + np.arange(L)]).astype(np.int32)
    heads = np.concatenate([np.arange(n), n + kk, np.full(L, T)]).astype(np.int32)
    caps = np.concatenate([p, (b - a)[kk], m * (b - a)]).astype(np.int64)
    f.add_arcs_with_capacity(tails, heads, caps)
    f.solve(S, T)
    return f.optimal_flow() == int(p.sum())


def er_ok(r, d, p, m):
    """Energetic reasoning (non-preemptive minimal intersection) on m identical machines: necessary condition."""
    n = len(p)
    if n == 0: return True
    if np.any(r + p > d): return False
    if m >= n: return True
    A = np.unique(np.concatenate([r, d - p])); B = np.unique(np.concatenate([d, r + p]))
    rp, dp = r + p, d - p
    tot = int(p.sum())
    for a in A.tolist():                         # one start point at a time: (B x n) memory only
        Bb = B[B > a]
        if Bb.size == 0: continue
        L = (Bb - a)[:, None]
        mi = np.minimum(np.minimum(L, p[None, :]), np.minimum(rp[None, :] - a, Bb[:, None] - dp[None, :]))
        W = np.maximum(mi, 0).sum(axis=1)
        if np.any(W > m * (Bb - a)): return False
    return True


# ----------------------------------------------------------------------------- windows and propagation

class Win:
    __slots__ = ("r1", "d1", "r2", "d2")

    def __init__(self, r1, d1, r2, d2):
        self.r1, self.d1, self.r2, self.d2 = r1, d1, r2, d2

    def copy(self):
        return Win(self.r1.copy(), self.d1.copy(), self.r2.copy(), self.d2.copy())

    def key(self):
        return (self.r1.tobytes(), self.d1.tobytes(), self.r2.tobytes(), self.d2.tobytes())


def init_win(I, T, h, q):
    return Win(np.zeros(I["n1"], np.int64), (T - q).astype(np.int64), h.astype(np.int64).copy(), np.full(I["n2"], T, np.int64))


class Ctx:
    """Per-instance constants (precedence masks as float matrices for fast propagation)."""

    def __init__(self, I):
        self.I = I; self.P = I["prec"]; self.Pf = I["prec"].astype(np.float64); self.p1 = I["p1"].astype(np.int64); self.p2 = I["p2"].astype(np.int64)
        self.m1, self.m2 = int(I["m1"]), int(I["m2"]); self.n1, self.n2 = int(I["n1"]), int(I["n2"])
        self.has_pred = self.P.any(axis=1); self.has_succ = self.P.any(axis=0)
        self.prof = None   # (U2, U1) union lower bounds -> door caps in every test (v6)


def _caps(C, T, W):
    """Door caps from the current windows (heads r2, tails T - d1) if C.prof is set, else (None, None)."""
    if C.prof is None: return None, None
    return lbx.door_caps(W.r2, T - W.d1, T, C.m1, C.m2, C.prof)


def propagate(C, W, max_iter=50):
    """Lemma W1 rules to a fixpoint; raises Infeasible if a window empties."""
    P, Pf, p1, p2, m1, m2 = C.P, C.Pf, C.p1, C.p2, C.m1, C.m2
    for _ in range(max_iter):
        ch = False
        # (W1a) heads: r2_j >= a + ceil(sum_{i in N(j)} clip(r1_i + p1_i - a, 0, p1_i) / m1) for a in {r1 values}
        A = np.unique(W.r1)
        R = np.clip(W.r1[None, :] + p1[None, :] - A[:, None], 0, p1[None, :]).astype(np.float64)   # (A, n1)
        Wk = Pf @ R.T                                                                            # (n2, A)
        cand = A[None, :] + np.ceil(Wk / m1 - 1e-9).astype(np.int64)
        cand = np.where(Wk > 0, cand, 0).max(axis=1)
        cand = np.maximum(cand, np.where(P, (W.r1 + p1)[None, :], 0).max(axis=1))
        nr2 = np.maximum(W.r2, cand)
        # (W1b) tails: d1_i <= b - ceil(sum_{j in S(i)} clip(b - (d2_j - p2_j), 0, p2_j) / m2) for b in {d2 values}
        Bv = np.unique(W.d2)
        Rb = np.clip(Bv[:, None] - (W.d2 - p2)[None, :], 0, p2[None, :]).astype(np.float64)     # (B, n2)
        Wb = Pf.T @ Rb.T                                                                         # (n1, B)
        candb = Bv[None, :] - np.ceil(Wb / m2 - 1e-9).astype(np.int64)
        candb = np.where(Wb > 0, candb, BIG).min(axis=1)
        candb = np.minimum(candb, np.where(P.T, (W.d2 - p2)[None, :], BIG).min(axis=1))
        nd1 = np.minimum(W.d1, candb)
        if not np.array_equal(nr2, W.r2): W.r2 = nr2; ch = True
        if not np.array_equal(nd1, W.d1): W.d1 = nd1; ch = True
        if np.any(W.r1 + p1 > W.d1) or np.any(W.r2 + p2 > W.d2): raise Infeasible
        if not ch: return W
    return W


def stages_ok(C, W, er=True):
    if not flow_ok(W.r1, W.d1, C.p1, C.m1): return False
    if not flow_ok(W.r2, W.d2, C.p2, C.m2): return False
    if er:
        if not er_ok(W.r1, W.d1, C.p1, C.m1): return False
        if not er_ok(W.r2, W.d2, C.p2, C.m2): return False
    return True


def consistent(C, W, er=True):
    """P + F (+E): False = refuted (proof that no schedule with Cmax <= T fits the windows)."""
    try:
        propagate(C, W)
    except Infeasible:
        return False
    return stages_ok(C, W, er)


# ----------------------------------------------------------------------------- shaving with window hypotheses (Lemma W2)

def _hyp(C, W, side, k, t, er):
    """Test one hypothesis on a copy of W; True = REFUTED.
    side 'r2': hyp s_j <= t-1 (d2_j := t-1+p2_j)   -> refuted => r2_j >= t
    side 'd2': hyp s_j >= t   (r2_j := t)          -> refuted => d2_j <= t-1+p2_j
    side 'r1': hyp C_i <= t-1 (d1_i := t-1)        -> refuted => r1_i >= t - p1_i
    side 'd1': hyp C_i >= t   (r1_i := t-p1_i)     -> refuted => d1_i <= t-1"""
    V = W.copy()
    if side == "r2": V.d2[k] = min(V.d2[k], t - 1 + C.p2[k])
    elif side == "d2": V.r2[k] = max(V.r2[k], t)
    elif side == "r1": V.d1[k] = min(V.d1[k], t - 1)
    else: V.r1[k] = max(V.r1[k], t - C.p1[k])
    return not consistent(C, V, er)


def shave_side(C, W, side, k, er, mode="bisect"):
    """Shave one side of one window. Returns True if the whole T is refuted (window empty)."""
    if side in ("r2", "d2"):
        lo, hi = int(W.r2[k]), int(W.d2[k] - C.p2[k])      # s_j in [lo, hi]
    else:
        lo, hi = int(W.r1[k] + C.p1[k]), int(W.d1[k])      # C_i in [lo, hi]
    if lo > hi: return True
    if lo == hi: return False
    if side in ("r2", "r1"):
        # largest t in (lo, hi] with "x <= t-1" refuted -> x >= t   (x = s_j or C_i)
        a, b, best = lo + 1, hi + 1, None
        if mode == "dich": b = min(b, (lo + hi + 1) // 2 + 1); a = b   # single midpoint test (HKN 2012 style)
        while a <= b:
            c = (a + b) // 2
            if _hyp(C, W, side, k, c, er): best = c; a = c + 1
            else: b = c - 1
        if best is not None:
            if best > hi: return True
            if side == "r2": W.r2[k] = max(W.r2[k], best)
            else: W.r1[k] = max(W.r1[k], best - C.p1[k])
    else:
        # smallest t in [lo+1, hi] with "x >= t" refuted -> x <= t-1
        a, b, best = lo, hi, None
        if mode == "dich": a = (lo + hi + 1) // 2; b = a
        while a <= b:
            c = (a + b) // 2
            if _hyp(C, W, side, k, c, er): best = c; b = c - 1
            else: a = c + 1
        if best is not None:
            if best <= lo: return True
            if side == "d2": W.d2[k] = min(W.d2[k], best - 1 + C.p2[k])
            else: W.d1[k] = min(W.d1[k], best - 1)
    return False


def shave(C, W, er=True, rounds=6, mode="bisect", deadline=None, sides=("r2", "d1", "d2", "r1"), extra=None):
    """Shaving to a fixpoint (or `rounds`). Returns (W, refuted). `extra(W)` is an additional per-round
    window-tightening procedure (e.g. LP probing); it returns True if it refutes T."""
    try:
        propagate(C, W)
    except Infeasible:
        return W, True
    if not stages_ok(C, W, er): return W, True
    for _ in range(rounds):
        k0 = W.key()
        for side in sides:
            n = C.n2 if side in ("r2", "d2") else C.n1
            for k in range(n):
                if deadline is not None and time.process_time() > deadline: return W, False
                if shave_side(C, W, side, k, er, mode): return W, True
            try:
                propagate(C, W)
            except Infeasible:
                return W, True
        if extra is not None and extra(W): return W, True
        try:
            propagate(C, W)
        except Infeasible:
            return W, True
        if not stages_ok(C, W, er): return W, True
        if W.key() == k0: break
    return W, False


# ----------------------------------------------------------------------------- LP of the joint split-cut test (Lemma W3)

EPS = 1e-3   # outward relaxation of every LP inequality (in time units); integer data => still a valid relaxation


class JLP:
    """LP relaxation of J(T, t) with window forcing, built once per (T, t, windows) with GLOP; probes change bounds only."""

    def __init__(self, C, T, t, W, dff=True, n_lam=4, avoid=True):
        from ortools.linear_solver import pywraplp
        self.C, self.T, self.t = C, T, t; p1, p2, m1, m2 = C.p1, C.p2, C.m1, C.m2
        s = pywraplp.Solver.CreateSolver("GLOP"); self.s = s; inf = s.infinity()
        P1, P2 = int(p1.sum()), int(p2.sum()); t1 = t - 1; L = T - t
        self.trivial_inf = False
        # forcing (Theorem J (6) with windows): i in Y needs r1_i + p1_i <= t-1; d1_i <= t-1 => i in Y;
        # j in A needs r2_j <= t-1; d2_j - p2_j <= t-1 => j in A
        ylo = (W.d1 <= t1).astype(float); yhi = ((W.r1 + p1) <= t1).astype(float)
        xlo = ((W.d2 - p2) <= t1).astype(float); xhi = (W.r2 <= t1).astype(float)
        if np.any(ylo > yhi) or np.any(xlo > xhi): self.trivial_inf = True; return
        self.y = [s.NumVar(float(ylo[i]), float(yhi[i]), "") for i in range(C.n1)]
        self.x = [s.NumVar(float(xlo[j]), float(xhi[j]), "") for j in range(C.n2)]
        self.base = (ylo, yhi, xlo, xhi)
        y, x = self.y, self.x
        for j in range(C.n2):                                      # closure
            if xhi[j] == 0: continue
            for i in np.flatnonzero(C.P[j]):
                s.Add(x[j] - y[i] <= EPS)
        # S1 packing (bins of size t-1) + DFF rows
        s.Add(sum(float(p1[i]) * y[i] for i in range(C.n1) if yhi[i] > 0) <= m1 * t1 + EPS)
        if dff and t1 > 0:
            idx = np.flatnonzero(yhi > 0)
            if idx.size:
                for f, r in lbx.dff_rows(p1[idx], t1, n_lam):
                    s.Add(sum(float(f[a]) * y[idx[a]] for a in range(idx.size) if f[a]) <= m1 * r + EPS)
        # S2 packing of J2 \ A (bins of size T-t) + DFF rows
        s.Add(sum(float(p2[j]) * (1 - x[j]) for j in range(C.n2)) <= m2 * L + EPS)
        if dff and L > 0:
            idx = np.flatnonzero(xlo < 1)
            okp = p2[idx] <= L
            idx = idx[okp]
            if idx.size:
                for f, r in lbx.dff_rows(p2[idx], L, n_lam):
                    s.Add(sum(float(f[a]) * (1 - x[idx[a]]) for a in range(idx.size) if f[a]) <= m2 * r + EPS)
        # S2 work: P2 - done(t) <= m2 (T - t); done_j <= clip(t - r2_j, 0, p2_j); done(t) <= m2 (t - rmin)
        need2 = P2 - m2 * L
        rmin = int(W.r2.min())
        Hc, Ec = _caps(C, T, W)
        if need2 > (int(np.maximum(0, t - Hc).sum()) if Hc is not None else m2 * max(0, t - rmin)): self.trivial_inf = True; return
        c2 = np.clip(t - W.r2, 0, p2).astype(float)
        if need2 > 0: s.Add(sum(c2[j] * x[j] for j in range(C.n2) if c2[j] > 0) >= need2 - EPS)
        if avoid: _avoid_rows(s, C, y, x, c2 * (xhi > 0), W, t)
        # S1 work at tau in {t-1, t}: P1 - after(tau) <= m1 tau; after_i <= clip(d1_i - tau, 0, p1_i)
        dmax = int(W.d1.max())
        for tau in (t - 1, t):
            need1 = P1 - m1 * tau
            if need1 <= 0: continue
            if need1 > (int(np.maximum(0, Ec - tau).sum()) if Ec is not None else m1 * max(0, dmax - tau)): self.trivial_inf = True; return
            c1 = np.clip(W.d1 - tau, 0, p1).astype(float)
            s.Add(sum(c1[i] * (1 - y[i]) for i in range(C.n1) if c1[i] > 0) >= need1 - EPS)
        self.ok = None

    def feasible(self):
        if self.trivial_inf: return False
        st = self.s.Solve()
        if st == self.s.INFEASIBLE: return False
        return True   # OPTIMAL / FEASIBLE / anything else (timeouts, numerical trouble) = not refuted

    def probe(self, var, val):
        """True = REFUTED (the LP with var fixed to val is infeasible). Restores the bounds."""
        lo, hi = var.lb(), var.ub()
        if val < lo or val > hi: return False   # already excluded by forcing
        var.SetBounds(val, val); r = not self.feasible()
        self.last = None if r else self.values()      # witness read BEFORE the bounds are restored
        var.SetBounds(lo, hi)
        return r

    def values(self):
        return np.array([v.solution_value() for v in self.y]), np.array([v.solution_value() for v in self.x])


def _threshold(keys, ps, forced_ps, m, cap, total):
    """Largest value v such that the set {k : keys_k < v} together with the forced items can fit in m bins of size cap
    (DFF necessary condition). Items sorted by key; returns `total` if everything fits. Lemma W6."""
    if cap < 0: return -1.0
    o = np.argsort(keys, kind="stable"); ks = keys[o]; pp = ps[o]
    forced = list(forced_ps)
    if forced and not lbx.dff_fits(forced, m, cap): return -1.0
    # binary search on the prefix length k (fits is monotone in the prefix: a superset that fits implies subsets fit)
    lo, hi = 0, len(ks)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if lbx.dff_fits(forced + pp[:mid].tolist(), m, cap): lo = mid
        else: hi = mid - 1
    # prefix of length lo fits, lo+1 does not -> any v > ks[lo] forces the first lo+1 items -> v <= ks[lo]
    return float(total) if lo == len(ks) else float(ks[lo])


def _avoid_rows(s, C, y, x, c2, W, t):
    """Lemma W5 (avoidance inequalities), valid for 0/1 points of the J system at cut t:
    (a) w = sum_j c2_j x_j; a_i = sum_{j: i notin N(j)} c2_j.  w > a_i  =>  A meets S(i)  =>  y_i = 1, so
        (Wt - a_i) y_i >= w - a_i   (Wt = sum_j c2_j).
    (b) z = sum_i c1_i (1 - y_i) with c1_i = clip(d1_i - t, 0, p1_i); e_j = sum_{i notin N(j)} c1_i.  z > e_j  =>  some
        pred of j is outside Y  =>  x_j = 0, so  (Z - e_j) x_j <= Z - z  (Z = sum_i c1_i)."""
    P = C.P; Wt = float(c2.sum()); t1 = t - 1; L = W.d2.max() - t
    ylo = np.array([v.lb() for v in y]); yhi = np.array([v.ub() for v in y])
    xlo = np.array([v.lb() for v in x]); xhi = np.array([v.ub() for v in x])
    if Wt > 0:
        a = (~P).T.astype(float) @ c2           # (n1,)
        w = sum(float(c2[j]) * x[j] for j in range(C.n2) if c2[j] > 0)
        # (c) union threshold: Y contains {i : a_i < w} and the forced members, and fits m1 bins of size t-1
        free = (ylo < 1)
        om = _threshold(a[free], C.p1[free], C.p1[ylo >= 1].tolist(), C.m1, t1, Wt)
        if om < 0: s.Add(w <= -1.0)          # forced members alone do not fit: infeasible
        elif om < Wt: s.Add(w <= om + EPS)
        for i in np.flatnonzero(a < Wt - 1e-9):
            if y[i].lb() >= 1 or y[i].ub() <= 0: continue
            s.Add(float(Wt - a[i]) * y[i] - w >= -float(a[i]) - EPS)
    c1 = np.clip(W.d1 - t, 0, C.p1).astype(float); Z = float(c1.sum())
    if Z > 0:
        e = (~P).astype(float) @ c1              # (n2,)
        z = sum(float(c1[i]) * (1 - y[i]) for i in range(C.n1) if c1[i] > 0)
        # (d) symmetric threshold: J2 \ A contains {j : e_j < z} and the forced-out jobs, and fits m2 bins of size T-t
        free = (xhi > 0)
        ze = _threshold(e[free], C.p2[free], C.p2[xhi <= 0].tolist(), C.m2, int(L), Z)
        if ze < 0: s.Add(z <= -1.0)
        elif ze < Z: s.Add(z <= ze + EPS)
        for j in np.flatnonzero(e < Z - 1e-9):
            if x[j].lb() >= 1 or x[j].ub() <= 0: continue
            s.Add(float(Z - e[j]) * x[j] + z <= Z + EPS)


def lp_cuts(C, T, W, k):
    a, b = max(1, int(W.r2.min()) + 1), T - max(1, int(T - W.d1.max()))
    if b < a: return []
    return sorted(set(np.linspace(a, b, min(k, b - a + 1)).round().astype(int).tolist()))


def lp_refute(C, T, W, cuts, dff=True):
    for t in cuts:
        if not JLP(C, T, t, W, dff).feasible(): return True
    return False


def lp_probe(C, T, W, cuts, dff=True, deadline=None, stats=None, avoid=True):
    """LP probing at each cut t (Lemma W3): x_j = 1 infeasible => s_j >= t; x_j = 0 infeasible => s_j <= t-1;
    y_i = 1 infeasible => C_i >= t; y_i = 0 infeasible => C_i <= t-1. Witness skipping: a value taken by some
    feasible LP solution needs no probe. Returns True if T is refuted."""
    for t in cuts:
        if deadline is not None and time.process_time() > deadline: return False
        M = JLP(C, T, t, W, dff, avoid=avoid)
        if not M.feasible(): return True
        yv, xv = M.values()
        seen_y1, seen_y0 = yv > 1 - 1e-6, yv < 1e-6
        seen_x1, seen_x0 = xv > 1 - 1e-6, xv < 1e-6
        ylo, yhi, xlo, xhi = M.base
        for j in range(C.n2):
            if xhi[j] > 0 and xlo[j] < 1 and not seen_x1[j]:
                if M.probe(M.x[j], 1.0):
                    W.r2[j] = max(W.r2[j], t)
                    if stats is not None: stats["x1"] = stats.get("x1", 0) + 1
                else:
                    a, b = M.last; seen_x1 |= b > 1 - 1e-6; seen_x0 |= b < 1e-6; seen_y1 |= a > 1 - 1e-6; seen_y0 |= a < 1e-6
            if xhi[j] > 0 and xlo[j] < 1 and not seen_x0[j]:
                if M.probe(M.x[j], 0.0):
                    W.d2[j] = min(W.d2[j], t - 1 + C.p2[j])
                    if stats is not None: stats["x0"] = stats.get("x0", 0) + 1
                else:
                    a, b = M.last; seen_x1 |= b > 1 - 1e-6; seen_x0 |= b < 1e-6; seen_y1 |= a > 1 - 1e-6; seen_y0 |= a < 1e-6
        for i in range(C.n1):
            if yhi[i] > 0 and ylo[i] < 1 and not seen_y1[i]:
                if M.probe(M.y[i], 1.0):
                    W.r1[i] = max(W.r1[i], t - C.p1[i])
                    if stats is not None: stats["y1"] = stats.get("y1", 0) + 1
                else:
                    a, b = M.last; seen_x1 |= b > 1 - 1e-6; seen_x0 |= b < 1e-6; seen_y1 |= a > 1 - 1e-6; seen_y0 |= a < 1e-6
            if yhi[i] > 0 and ylo[i] < 1 and not seen_y0[i]:
                if M.probe(M.y[i], 0.0):
                    W.d1[i] = min(W.d1[i], t - 1)
                    if stats is not None: stats["y0"] = stats.get("y0", 0) + 1
                else:
                    a, b = M.last; seen_x1 |= b > 1 - 1e-6; seen_x0 |= b < 1e-6; seen_y1 |= a > 1 - 1e-6; seen_y0 |= a < 1e-6
        if np.any(W.r1 + C.p1 > W.d1) or np.any(W.r2 + C.p2 > W.d2): return True
    return False


# ----------------------------------------------------------------------------- CP-SAT nested test with windows (optional)

def nested_with_windows(I, T, W, cuts, det=2.0, cfg=None):
    """Proposition N with heads h = r2, tails q = T - d1 and extra forcing from d2 (j in A if d2_j - p2_j <= t-1) and
    r1 (i notin Y if r1_i + p1_i > t-1). Timeouts = not refuted."""
    p1, p2 = I["p1"], I["p2"]
    def ex(m, cs, Y, X):
        for k, t in enumerate(cs):
            for j in np.flatnonzero(W.d2 - p2 <= t - 1): m.Add(X[k][j] == 1)
            for i in np.flatnonzero(W.r1 + p1 > t - 1): m.Add(Y[k][i] == 0)
    m = lbx.build_cuts(I, T, cuts, cfg or lbx.Cfg(prof=PROF.get(id(I))), W.r2, (T - W.d1), nested=True, extra=ex)
    return lbx.solve_feas(m, det=det) == "INF"


# ----------------------------------------------------------------------------- the WIB refutation procedures (versions)

def refute_v1(C, T, h, q, budget=60.0, mode="bisect", er=True):
    """V1: windows only (P + F + E shaving to a fixpoint)."""
    W = init_win(C.I, T, h, q)
    W, ref = shave(C, W, er=er, mode=mode, deadline=time.process_time() + budget)
    return ref, W


def refute_v2(C, T, h, q, budget=60.0, k_cuts=16, mode="bisect", er=True, dff=True, stats=None, avoid=False):
    """V2: V1 + JLP probing at k cuts inside every shaving round, and JLP refutation at the cuts."""
    W = init_win(C.I, T, h, q); dl = time.process_time() + budget
    def extra(W_):
        cuts = lp_cuts(C, T, W_, k_cuts)
        if lp_probe(C, T, W_, cuts, dff, deadline=dl, stats=stats, avoid=avoid): return True
        return False
    W, ref = shave(C, W, er=er, mode=mode, deadline=dl, extra=extra)
    return ref, W


# ----------------------------------------------------------------------------- LP of the nested test (Proposition N + windows), Lemma W4

def nuk_lp_feasible(C, T, W, cuts, dff=True, n_lam=4, tl=30.0):
    """LP relaxation of the nested union-knapsack test at cuts t_1 < ... < t_K with window forcing and bucket-refined
    work caps. False = REFUTED. Built with GLOP; RHS relaxed outward by EPS."""
    from ortools.linear_solver import pywraplp
    p1, p2, m1, m2, n1, n2 = C.p1, C.p2, C.m1, C.m2, C.n1, C.n2
    cuts = sorted(set(int(t) for t in cuts if 0 < t < T)); K = len(cuts)
    if K == 0: return True
    s = pywraplp.Solver.CreateSolver("GLOP"); s.SetTimeLimit(int(tl * 1000))
    P1, P2 = int(p1.sum()), int(p2.sum()); rmin = int(W.r2.min()); dmax = int(W.d1.max())
    Hc, Ec = _caps(C, T, W)
    Y, X = [], []
    for k, t in enumerate(cuts):
        t1 = t - 1
        ylo = (W.d1 <= t1).astype(float); yhi = ((W.r1 + p1) <= t1).astype(float)
        xlo = ((W.d2 - p2) <= t1).astype(float); xhi = (W.r2 <= t1).astype(float)
        if np.any(ylo > yhi) or np.any(xlo > xhi): return False
        Y.append([s.NumVar(float(ylo[i]), float(yhi[i]), "") for i in range(n1)])
        X.append([s.NumVar(float(xlo[j]), float(xhi[j]), "") for j in range(n2)])
    zero = None
    for k, t in enumerate(cuts):
        y, x = Y[k], X[k]; t1 = t - 1; L = T - t
        if k > 0:
            for i in range(n1): s.Add(Y[k - 1][i] - y[i] <= EPS)
            for j in range(n2): s.Add(X[k - 1][j] - x[j] <= EPS)
        for j in range(n2):
            if x[j].ub() == 0: continue
            for i in np.flatnonzero(C.P[j]): s.Add(x[j] - y[i] <= EPS)
        s.Add(sum(float(p1[i]) * y[i] for i in range(n1) if y[i].ub() > 0) <= m1 * t1 + EPS)
        if dff and t1 > 0:
            idx = np.array([i for i in range(n1) if y[i].ub() > 0], dtype=int)
            if idx.size:
                for f, r in lbx.dff_rows(p1[idx], t1, n_lam):
                    s.Add(sum(float(f[a]) * y[idx[a]] for a in range(idx.size) if f[a]) <= m1 * r + EPS)
        s.Add(sum(float(p2[j]) * (1 - x[j]) for j in range(n2)) <= m2 * L + EPS)
        if dff and L > 0:
            idx = np.array([j for j in range(n2) if x[j].lb() < 1 and p2[j] <= L], dtype=int)
            if idx.size:
                for f, r in lbx.dff_rows(p2[idx], L, n_lam):
                    s.Add(sum(float(f[a]) * (1 - x[idx[a]]) for a in range(idx.size) if f[a]) <= m2 * r + EPS)
        # S2 work with bucket caps: a job started in bucket l (t_{l-1} <= s_j <= t_l - 1) has done <= t - max(r2_j, t_{l-1})
        need2 = P2 - m2 * L
        if need2 > (int(np.maximum(0, t - Hc).sum()) if Hc is not None else m2 * max(0, t - rmin)): return False
        if need2 > 0:
            terms = []
            for j in range(n2):
                for l in range(k + 1):
                    lo = max(int(W.r2[j]), cuts[l - 1] if l > 0 else 0)
                    cap = float(max(0, min(int(p2[j]), t - lo)))
                    if cap == 0: continue
                    terms.append(cap * X[l][j]);
                    if l > 0: terms.append(-cap * X[l - 1][j])
            s.Add(sum(terms) >= need2 - EPS)
        # S1 work at tau in {t-1, t} with bucket caps: i completing in bucket l > k has after(tau) <= min(t_l - 1, d1_i) - tau
        for tau in (t - 1, t):
            need1 = P1 - m1 * tau
            if need1 <= 0: continue
            if need1 > (int(np.maximum(0, Ec - tau).sum()) if Ec is not None else m1 * max(0, dmax - tau)): return False
            terms = []
            for i in range(n1):
                for l in range(k + 1, K + 1):
                    hi_ = min(cuts[l] - 1 if l < K else T, int(W.d1[i]))
                    cap = float(max(0, min(int(p1[i]), hi_ - tau)))
                    if cap == 0: continue
                    if l < K: terms += [cap * Y[l][i], -cap * Y[l - 1][i]]
                    else: terms += [cap, -cap * Y[K - 1][i]]
            s.Add(sum(terms) >= need1 - EPS)
    st = s.Solve()
    return st != s.INFEASIBLE


def refute_v4(C, T, h, q, budget=60.0, k_cuts=16, nuk_sets=((12, "u"), (24, "u")), stats=None):
    """V4: V2 windows, then NUK-LP refutation at nested cut sets (all polynomial)."""
    t0 = time.process_time()
    ref, W = refute_v2(C, T, h, q, budget=budget * 0.6, k_cuts=k_cuts, stats=stats)
    if ref: return True, W
    for K, _ in nuk_sets:
        if time.process_time() - t0 > budget: break
        if not nuk_lp_feasible(C, T, W, lp_cuts(C, T, W, K), tl=max(1.0, budget - (time.process_time() - t0))):
            return True, W
    return False, W


# ----------------------------------------------------------------------------- staggered dock capacity (Lemma W7)

def min_union(P, w, k, det=1.0):
    """Valid LOWER bound on U(k) = min over k-subsets A of the columns-side jobs of w(union of preds of A).
    P: (n_out, n_in) bool. CP-SAT best objective bound after `det` deterministic seconds (exact if solved)."""
    from ortools.sat.python import cp_model
    n_out, n_in = P.shape
    if k <= 0: return 0
    if k > n_out: return int(w.sum()) + 1
    if k == 1: return int(min(w[P[j]].sum() for j in range(n_out)))
    m = cp_model.CpModel(); x = [m.NewBoolVar("") for _ in range(n_out)]; y = [m.NewBoolVar("") for _ in range(n_in)]
    m.Add(sum(x) == k)
    for j in range(n_out):
        for i in np.flatnonzero(P[j]): m.AddImplication(x[j], y[i])
    m.Minimize(sum(int(w[i]) * y[i] for i in range(n_in)))
    sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.random_seed = 0; sv.parameters.max_deterministic_time = det
    st = sv.Solve(m)
    if st == cp_model.INFEASIBLE: return int(w.sum()) + 1
    return int(math.ceil(sv.BestObjectiveBound() - 1e-6))


def union_profiles(C, det=1.0):
    """U2[k-1] >= min p1-weight of the pred union of k outbound jobs (k = 1..m2);
    U1[k-1] >= min p2-weight of the succ union of k inbound jobs (k = 1..m1). Monotone repaired (U(k) is nondecreasing)."""
    U2 = [min_union(C.P, C.p1, k, det) for k in range(1, min(C.m2, C.n2) + 1)]
    U1 = [min_union(C.P.T, C.p2, k, det) for k in range(1, min(C.m1, C.n1) + 1)]
    return np.maximum.accumulate(np.array(U2, np.int64)), np.maximum.accumulate(np.array(U1, np.int64))


def start_profile(C, W, U2):
    """H[k-1] <= k-th smallest outbound start (any schedule in W): max(k-th smallest r2, ceil(U2(k)/m1))."""
    k = len(U2); hs = np.sort(W.r2)[:k]
    H = np.maximum(hs, -(-U2 // C.m1)); return np.maximum.accumulate(H)


def end_profile(C, W, U1):
    """E[k-1] >= k-th largest inbound completion: min(k-th largest d1, T - ceil(U1(k)/m2)) with T = max d2."""
    k = len(U1); T = int(W.d2.max()); ds = np.sort(W.d1)[::-1][:k]
    E = np.minimum(ds, T - (-(-U1 // C.m2))); return np.minimum.accumulate(E)


def flow_ok_prof(r, d, p, m, H=None, E=None):
    """P|pmtn, r_j, d_j| with a time-varying machine count: on [a, b], at most min(m, #{k: H_k <= a}) jobs (start
    profile) and at most min(m, #{k: E_k >= b}) jobs (end profile) run in parallel. Exact for that relaxation."""
    n = len(p)
    if n == 0: return True
    if np.any(r + p > d): return False
    extra = [r, d] + ([H] if H is not None else []) + ([E] if E is not None else [])
    pts = np.unique(np.concatenate(extra)); pts = pts[(pts >= r.min()) & (pts <= d.max())]
    if pts.size < 2: return False
    L = len(pts) - 1; a, b = pts[:-1], pts[1:]
    cap = np.full(L, m, np.int64)
    if H is not None: cap = np.minimum(cap, (H[None, :] <= a[:, None]).sum(axis=1))
    if E is not None: cap = np.minimum(cap, (E[None, :] >= b[:, None]).sum(axis=1))
    if int((cap * (b - a)).sum()) < int(p.sum()): return False
    ok = (r[:, None] <= a[None, :]) & (d[:, None] >= b[None, :])
    jj, kk = np.nonzero(ok)
    f = max_flow.SimpleMaxFlow(); S, Tn = n + L, n + L + 1
    tails = np.concatenate([np.full(n, S), jj, n + np.arange(L)]).astype(np.int32)
    heads = np.concatenate([np.arange(n), n + kk, np.full(L, Tn)]).astype(np.int32)
    caps = np.concatenate([p, (b - a)[kk], cap * (b - a)]).astype(np.int64)
    f.add_arcs_with_capacity(tails, heads, caps)
    f.solve(S, Tn)
    return f.optimal_flow() == int(p.sum())


def profile_lb(C, h, q, U2, U1, lo, hi):
    """Static staggered-capacity bound: smallest T for which both per-stage profile flows are feasible
    (stage 2: r = h, d = T, start profile; stage 1: r = 0, d = T - q, end profile)."""
    def ref(T):
        W = init_win(C.I, T, h, q)
        if not flow_ok_prof(W.r2, W.d2, C.p2, C.m2, H=start_profile(C, W, U2)): return True
        if not flow_ok_prof(W.r1, W.d1, C.p1, C.m1, E=end_profile(C, W, U1)): return True
        return False
    return lbx.destructive(ref, lo, hi, budget=60)[0]


def min_union_pairs(P, w):
    """Exact U(2) by enumeration (vectorised): min over pairs j < j' of w(N(j) | N(j'))."""
    Pf = P.astype(np.float64); a = Pf @ w.astype(np.float64)
    inter = (Pf * w[None, :]) @ Pf.T
    U = a[:, None] + a[None, :] - inter
    np.fill_diagonal(U, np.inf); return int(round(U.min()))


def union_profiles2(C, det=1.0):
    U2, U1 = union_profiles(C, det)
    if U2.size >= 2: U2[1] = max(U2[1], min_union_pairs(C.P, C.p1))
    if U1.size >= 2: U1[1] = max(U1[1], min_union_pairs(C.P.T, C.p2))
    return np.maximum.accumulate(U2), np.maximum.accumulate(U1)


def lb_stagger_fwd(I, h, U2, grid=48, tl=1.0, Kcache=None):
    """Lemma W8: T >= t + (P2 - D(t)) / m2 for every t with D(t) < P2, where
    D(t) = min_{tau <= t} [ Kub(m1 tau) + sum_k ( t - max(H_k, tau) )^+ ],  H_k = max(k-th smallest h, ceil(U2(k)/m1)).
    Kub(c) = CP-SAT upper bound on the union knapsack K at the next grid capacity >= c (K is nondecreasing)."""
    import g1
    p1, p2, m1, m2 = I["p1"], I["p2"], int(I["m1"]), int(I["m2"]); P2 = int(p2.sum())
    k = len(U2); H = np.maximum.accumulate(np.maximum(np.sort(h)[:k], -(-U2 // m1))).astype(float)
    H = np.concatenate([H, np.full(max(0, m2 - k), H[-1] if k else 0.0)])[:m2]
    full = int(p1[I["prec"].any(axis=0)].sum())
    cs = np.unique(np.linspace(0, full, grid).round().astype(int))
    uk = g1.UnionKnap(I)
    Kc = Kcache if Kcache is not None else {}
    for c in cs:
        if int(c) not in Kc: Kc[int(c)] = P2 if c >= full else uk.solve(int(c), tl)[0]
    Kv = np.array([Kc[int(c)] for c in cs], float)
    def Kub(cap):  # cap array -> upper bound
        idx = np.searchsorted(cs, cap, side="left")
        return np.where(idx < cs.size, Kv[np.minimum(idx, cs.size - 1)], float(P2))
    tmax = full / m1 + 1
    ts = np.unique(np.concatenate([np.linspace(H[0], tmax, 400), cs / m1, H]))
    best = 0.0
    taus = np.unique(np.concatenate([[0.0], cs / m1, H]))
    for t in ts:
        tt = taus[taus <= t]
        cap = np.maximum(0.0, t - np.maximum(H[None, :], tt[:, None])).sum(axis=1)
        Dv = (Kub(m1 * tt) + cap).min()
        Dv = min(Dv, Kub(np.array([m1 * t]))[0])
        if Dv < P2: best = max(best, t + (P2 - Dv) / m2)
    return int(math.ceil(best - 1e-9)), Kc
