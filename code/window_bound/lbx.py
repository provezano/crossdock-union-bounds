"""LB/UB research arm (2026-11-02): new lower bounds for P|M_j,prec|Cmax (two-stage cross-dock, Cota+ 2016).

All bounds here are DESTRUCTIVE: for a trial makespan T they try to prove "no schedule has Cmax <= T".
A refuted T proves OPT > T by itself, so  LB = (largest refuted T) + 1  is valid whatever the search order.

Notation (integer data): inbound i (n1, p1, m1 docks), outbound j (n2, p2, m2 docks), N(j) = preds of j,
S(i) = succs of i.  h_j = valid head of j (earliest start), q_i = valid tail of i (C_i <= Cmax - q_i).

Single-cut joint test J(T, t) (Theorem J in the notes; "split cut"). For any integer schedule with Cmax <= T and
integer t, define Y = {i : C_i <= t - 1} and A = {j : s_j <= t - 1}. Then
  (closure)  j in A  =>  N(j) subset of Y
  (S1 pack)  Y fits in m1 bins of size t - 1       -> every DFF f: sum_{i in Y} f(p1_i) <= m1 f(t - 1)
  (S1 work)  for tau in {t-1, t}: P1 - sum_{i notin Y} min(p1_i, T - q_i - tau) <= m1 tau, after part <= m1 (T - tau - qmin)
  (S2 work)  P2 - min(sum_{j in A} min(p2_j, t - h_j), m2 (t - rmin)) <= m2 (T - t)
  (S2 pack)  J2 \\ A fits in m2 bins of size T - t  -> every DFF f: sum_{j notin A} f(p2_j) <= m2 f(T - t)
  (forcing)  p1_i > t-1 => i notin Y;  T - q_i <= t-1 => i in Y;  h_j >= t => j notin A;  T - p2_j < t => j in A.
If no (Y, A) satisfies them, Cmax > T.  The forward LB_U* is the special case Y = N(A) with only (S1 load) and
(S2 work); the reversed LB_U* is the special case with only (S1 work) and (S2 load). The joint test couples both.

Nested test N(T, t_1..t_K): the single-cut systems at K cuts with Y_1 <= ... <= Y_K, A_1 <= ... <= A_K and
bucket-refined partial work (a job that starts in bucket (t_{l-1}, t_l] has done at most t_k - max(h_j, t_{l-1})
by t_k; symmetric for inbound work after t_k).
"""
import sys
sys.dont_write_bytecode = True
import os, time, math
D = os.path.dirname(os.path.abspath(__file__))
for p in ("lower_bounds", "benchmark_data"):
    q = f"{D}/../{p}"
    if q not in sys.path: sys.path.append(q)
from ortools.sat.python import cp_model  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------- instance helpers

def parse(path):
    tok = np.array(open(path).read().split(), dtype=np.int64)
    n1, n2, m1, m2 = (int(x) for x in tok[:4])
    p1 = tok[4:4 + n1].copy(); p2 = tok[4 + n1:4 + n1 + n2].copy()
    prec = tok[4 + n1 + n2:4 + n1 + n2 + n1 * n2].reshape(n2, n1).astype(bool)
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=p1, p2=p2, prec=prec)


def reverse(I):
    return dict(n1=I["n2"], n2=I["n1"], m1=I["m2"], m2=I["m1"], p1=I["p2"].copy(), p2=I["p1"].copy(), prec=I["prec"].T.copy())


def pcmax_lb(ps, m):
    """Valid integer LB on the P||Cmax makespan of jobs ps on m machines (0 if empty): max of load, pmax,
    p_(m)+p_(m+1), p_(2m-1)+p_(2m)+p_(2m+1), and DFF destructive tests (each failing C proves Cmax > C)."""
    ps = sorted((int(x) for x in ps), reverse=True)
    if not ps: return 0
    lb = max(-(-sum(ps) // m), ps[0])
    if len(ps) > m: lb = max(lb, ps[m - 1] + ps[m])
    if len(ps) > 2 * m: lb = max(lb, ps[2 * m - 2] + ps[2 * m - 1] + ps[2 * m])
    if len(ps) <= m: return lb
    C = lb
    for _ in range(4 * len(ps) + 64):          # scan upwards; each refuted C is a proof on its own
        if dff_fits(ps, m, C): return C
        C += 1
    return C


def heads_tails_bp(I):
    """h_j = P||Cmax LB of N(j) on m1 docks (>= ceil(max(p1(N)/m1, max p1))); q_i = same for S(i) on m2 docks."""
    P = I["prec"]
    h = np.array([pcmax_lb(I["p1"][P[j]], I["m1"]) for j in range(I["n2"])], dtype=np.int64)
    q = np.array([pcmax_lb(I["p2"][P[:, i]], I["m2"]) for i in range(I["n1"])], dtype=np.int64)
    return h, q


def heads_tails_basic(I):
    P = I["prec"]; S = P.T
    h = np.ceil(np.maximum(P @ I["p1"] / I["m1"], np.where(P, I["p1"][None, :], 0).max(axis=1)) - 1e-9).astype(np.int64)
    q = np.ceil(np.maximum(S @ I["p2"] / I["m2"], np.where(S, I["p2"][None, :], 0).max(axis=1)) - 1e-9).astype(np.int64)
    return h, q

# ----------------------------------------------------------------------------- dual feasible functions (integer)

def dff_rows(xs, C, n_lam=6, ks=(1, 2, 3)):
    """Integer DFF rows for bin capacity C: list of (coef per item, rhs per bin). For any set of items that fits in
    one bin, sum coef <= rhs (Fekete & Schepers 2001; Martello & Toth 1990 L2 family; Carlier, Clautiaux & Moukrim
    2007 f0).  Items with x > C must be excluded by the caller."""
    xs = np.asarray(xs, dtype=np.int64); rows = []
    if C <= 0: return rows
    small = np.unique(xs[(xs >= 1) & (2 * xs <= C)])
    if small.size > n_lam: small = small[np.unique(np.linspace(0, small.size - 1, n_lam).round().astype(int))]
    for lam in small.tolist():                      # MT: big items count C, mid items count x, small 0
        f = np.where(xs > C - lam, C, np.where(xs >= lam, xs, 0)); rows.append((f, C))
        fl = C // lam                               # CCM f0^lam
        g = np.where(2 * xs > C, 2 * (fl - (C - xs) // lam), np.where(2 * xs == C, fl, 2 * (xs // lam)))
        rows.append((g, 2 * fl))
    for k in ks:                                    # Fekete-Schepers u^(k), scaled by k*C
        f = np.where(((k + 1) * xs) % C == 0, k * xs, ((k + 1) * xs // C) * C); rows.append((f, k * C))
    out, seen = [], set()
    for f, r in rows:
        key = (tuple(f.tolist()), int(r))
        if key in seen or not np.any(f): continue
        seen.add(key); out.append((f.astype(np.int64), int(r)))
    return out


def dff_fits(ps, m, C):
    """Necessary condition (all DFF rows + load + pmax) for ps to fit in m bins of size C."""
    ps = np.asarray(ps, dtype=np.int64)
    if ps.size == 0: return True
    if ps.max() > C or ps.sum() > m * C: return False
    for f, r in dff_rows(ps, C):
        if f.sum() > m * r: return False
    return True

# ----------------------------------------------------------------------------- preemptive windows (flow) bound

def _pmtn_feasible(r, d, p, m):
    """Exact feasibility of P|pmtn, r_j, d_j| (Horn 1974 network) with integer data."""
    from ortools.graph.python import max_flow
    n = len(p)
    if n == 0: return True
    if np.any(r + p > d): return False
    if m >= n: return True
    pts = np.unique(np.concatenate([r, d])); L = len(pts) - 1
    f = max_flow.SimpleMaxFlow(); S, T = n + L, n + L + 1
    for j in range(n): f.add_arc_with_capacity(S, j, int(p[j]))
    for k in range(L):
        a, b = pts[k], pts[k + 1]; f.add_arc_with_capacity(n + k, T, int(m * (b - a)))
        for j in range(n):
            if r[j] <= a and d[j] >= b: f.add_arc_with_capacity(j, n + k, int(b - a))
    f.solve(S, T)
    return f.optimal_flow() == int(p.sum())


def refute_pmtn(I, T, h=None, q=None, max_iter=30):
    """Preemptive per-stage relaxation with release/deadline windows and cross-stage propagation:
    inbound [0, T - q_i], outbound [h_j, T]; r2_j >= max_{N(j)} (r1_i + p1_i), d1_i <= min_{S(i)} (d2_j - p2_j).
    Windows only tighten through propagation (no adjustments), so this is a fixpoint after one pass; True = refuted."""
    if h is None: h, q = heads_tails_bp(I)
    P = I["prec"]; S = P.T; p1, p2 = I["p1"], I["p2"]
    r1 = np.zeros(I["n1"], np.int64); d1 = (T - q).astype(np.int64)
    r2 = h.astype(np.int64).copy(); d2 = np.full(I["n2"], T, np.int64)
    for _ in range(max_iter):
        o = (r2.copy(), d1.copy())
        r2 = np.maximum(r2, np.where(P, (r1 + p1)[None, :], 0).max(axis=1))
        d1 = np.minimum(d1, np.where(S, (d2 - p2)[None, :], 1 << 40).min(axis=1))
        if np.array_equal(o[0], r2) and np.array_equal(o[1], d1): break
    if not _pmtn_feasible(r1, d1, p1, I["m1"]): return True
    if not _pmtn_feasible(r2, d2, p2, I["m2"]): return True
    return False

# ----------------------------------------------------------------------------- joint / nested union-knapsack tests

class Cfg:
    """Which ingredients the test uses (for the oracle-swap decomposition)."""
    def __init__(self, s1dff=True, s2dff=True, s1work=True, s2work=True, partial=True, exact_s1=False, exact_s2=False,
                 joint=True, n_lam=6, prof=None):
        self.s1dff, self.s2dff, self.s1work, self.s2work, self.partial = s1dff, s2dff, s1work, s2work, partial
        self.exact_s1, self.exact_s2, self.joint, self.n_lam = exact_s1, exact_s2, joint, n_lam
        self.prof = prof   # None = constant caps (original); (U2, U1) = door-start / door-end caps (Lemma W7, WIB v6)


def door_caps(h, q, T, m1, m2, prof):
    """Door-start cap profile H (k-th smallest outbound start >= H_k, k <= min(m2, n2)) and door-end profile E
    (k-th largest inbound completion <= E_k, k <= min(m1, n1)), from heads h, tails q and union lower bounds
    prof = (U2, U1) (either may be None). Lemma W7 and the door-start lemma (paper supplement, Section S5)."""
    U2, U1 = prof
    k2 = min(m2, len(h)); H = np.sort(np.asarray(h, np.int64))[:k2]
    if U2 is not None: H = np.maximum(H, -(-np.asarray(U2[:k2], np.int64) // m1))
    H = np.maximum.accumulate(H)
    k1 = min(m1, len(q)); E = np.sort(T - np.asarray(q, np.int64))[::-1][:k1]
    if U1 is not None: E = np.minimum(E, T - (-(-np.asarray(U1[:k1], np.int64) // m2)))
    E = np.minimum.accumulate(E)
    return H, E


def _add_pack_exact(m, sel, ps, mcap, C, positive=True):
    """sel[k] (bool) chosen items must fit in mcap bins of size C: assignment model (exact bin packing)."""
    n = len(ps)
    if n == 0 or mcap >= n:
        for k in range(n):
            if ps[k] > C: m.Add(sel[k] == (0 if positive else 1))
        return
    z = [[m.NewBoolVar("") for _ in range(mcap)] for _ in range(n)]
    for k in range(n):
        lit = sel[k] if positive else sel[k].Not()
        m.Add(sum(z[k]) == 1).OnlyEnforceIf(lit)
        m.Add(sum(z[k]) == 0).OnlyEnforceIf(lit.Not())
    for b in range(mcap):
        m.Add(sum(int(ps[k]) * z[k][b] for k in range(n)) <= int(C))
    for k in range(min(n, mcap)):                    # symmetry: item k uses one of the first k+1 bins
        for b in range(k + 1, mcap): m.Add(z[k][b] == 0)


def build_cuts(I, T, cuts, cfg, h, q, nested=True, extra=None):
    """CP-SAT feasibility model of the joint test at the given integer cuts (one cut = single-cut test J(T, t))."""
    p1, p2, P = I["p1"].astype(np.int64), I["p2"].astype(np.int64), I["prec"]; n1, n2, m1, m2 = I["n1"], I["n2"], I["m1"], I["m2"]
    P1, P2 = int(p1.sum()), int(p2.sum())
    rmin = int(h.min()) if n2 else 0
    qmin = int(q.min()) if n1 else 0
    if getattr(cfg, "prof", None) is not None:
        Hc, Ec = door_caps(h, q, T, m1, m2, cfg.prof)
        dcap = lambda t: int(np.maximum(0, t - Hc).sum())          # stage-2 work done by t
        ecap = lambda tau: int(np.maximum(0, Ec - tau).sum())       # stage-1 work after tau
    else:
        dcap = lambda t: m2 * max(0, t - rmin)
        ecap = lambda tau: m1 * max(0, T - tau - qmin)
    m = cp_model.CpModel(); E = [(j, i) for j in range(n2) for i in np.flatnonzero(P[j])]
    cuts = sorted(set(int(t) for t in cuts if 0 < t < T)); K = len(cuts)
    Y = [[m.NewBoolVar("") for _ in range(n1)] for _ in range(K)]
    X = [[m.NewBoolVar("") for _ in range(n2)] for _ in range(K)]
    for k, t in enumerate(cuts):
        y, x = Y[k], X[k]
        t1 = t - 1                                   # split cut: Y = {C_i <= t - 1}, A = {s_j <= t - 1}
        for i in range(n1):
            if p1[i] > t1: m.Add(y[i] == 0)
            if T - q[i] <= t1: m.Add(y[i] == 1)
        for j in range(n2):
            if h[j] >= t: m.Add(x[j] == 0)
            if T - p2[j] < t: m.Add(x[j] == 1)
        for j, i in E: m.AddImplication(x[j], y[i])
        # S1 packing of Y into m1 bins of size t - 1
        m.Add(sum(int(p1[i]) * y[i] for i in range(n1)) <= m1 * t1)
        if t1 > 0:
            if cfg.exact_s1: _add_pack_exact(m, y, p1, m1, t1, True)
            elif cfg.s1dff:
                ok = p1 <= t1
                for f, r in dff_rows(p1[ok], t1, cfg.n_lam):
                    idx = np.flatnonzero(ok); m.Add(sum(int(f[a]) * y[idx[a]] for a in range(idx.size) if f[a]) <= m1 * r)
        # S2 packing of J2 \ A into m2 bins of size T - t
        L = T - t
        m.Add(sum(int(p2[j]) * (1 - x[j]) for j in range(n2)) <= m2 * L)
        if cfg.exact_s2: _add_pack_exact(m, x, p2, m2, L, False)
        elif cfg.s2dff:
            ok = p2 <= L
            for f, r in dff_rows(p2[ok], L, cfg.n_lam):
                idx = np.flatnonzero(ok); m.Add(sum(int(f[a]) * (1 - x[idx[a]]) for a in range(idx.size) if f[a]) <= m2 * r)
        # S2 work: P2 - done(t) <= m2 (T - t), done(t) <= m2 (t - rmin), done_j <= min(p2_j, t - h_j) (bucket-refined)
        if cfg.s2work:
            dsum = []
            for j in range(n2):
                if h[j] >= t: continue
                if cfg.partial and nested and k > 0:
                    # started in bucket l (cuts[l-1] <= s < cuts[l]) -> done by t <= t - max(h_j, cuts[l-1])
                    terms = []
                    for l in range(k + 1):
                        lo = max(int(h[j]), cuts[l - 1] if l > 0 else 0)
                        cap = max(0, min(int(p2[j]), t - lo))
                        if cap == 0: continue
                        if l == 0: terms.append((cap, X[0][j], None))
                        else: terms.append((cap, X[l][j], X[l - 1][j]))
                    dj = m.NewIntVar(0, int(p2[j]), "")
                    m.Add(dj <= (sum(c * (a if b is None else (a - b)) for c, a, b in terms) if terms else 0))
                    dsum.append(dj)
                else:
                    cap = int(min(p2[j], t - h[j])) if cfg.partial else int(min(p2[j], max(0, t - rmin)))
                    dsum.append(cap * x[j])
            dn = m.NewIntVar(0, P2, "")
            m.Add(dn <= (sum(dsum) if dsum else 0)); m.Add(dn <= dcap(t))
            m.Add(P2 - dn <= m2 * (T - t))
        # S1 work at tau in {t-1, t} (i notin Y has C_i >= t > tau): P1 - after(tau) <= m1 tau,
        # after(tau) <= m1 (T - tau - qmin), after_i(tau) <= min(p1_i, C_i - tau) <= min(p1_i, T - q_i - tau) (bucket-refined)
        if cfg.s1work and cfg.joint:
            for tau in (t - 1, t):
                asum = []
                for i in range(n1):
                    capi = int(min(p1[i], max(0, T - q[i] - tau)))
                    if capi == 0: continue
                    if cfg.partial and nested and k < K - 1:
                        # completes in bucket l > k (cuts[l-1] <= C_i <= cuts[l] - 1) -> after(tau) <= min(cuts[l]-1, T-q_i) - tau
                        terms = []
                        for l in range(k + 1, K + 1):
                            hi_ = min(cuts[l] - 1 if l < K else T, int(T - q[i]))
                            cap = max(0, min(int(p1[i]), hi_ - tau))
                            if cap == 0: continue
                            if l < K: terms.append((cap, Y[l][i], Y[l - 1][i]))     # y_l - y_{l-1}
                            else: terms.append((cap, None, Y[K - 1][i]))           # 1 - y_K
                        ai = m.NewIntVar(0, int(p1[i]), "")
                        m.Add(ai <= (sum(c * ((a - b) if a is not None else (1 - b)) for c, a, b in terms) if terms else 0))
                        asum.append(ai)
                    else:
                        asum.append(capi * (1 - y[i]))
                af = m.NewIntVar(0, P1, "")
                m.Add(af <= (sum(asum) if asum else 0)); m.Add(af <= ecap(tau))
                m.Add(P1 - af <= m1 * tau)
    if nested:
        for k in range(K - 1):
            for i in range(n1): m.AddImplication(Y[k][i], Y[k + 1][i])
            for j in range(n2): m.AddImplication(X[k][j], X[k + 1][j])
    if extra is not None: extra(m, cuts, Y, X)
    return m


LINEARIZATION = None   # None = CP-SAT default; 2 = full linear relaxation at the root


def solve_feas(m, tl=5.0, det=None):
    sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.random_seed = 0
    if LINEARIZATION is not None: sv.parameters.linearization_level = LINEARIZATION
    if det is not None: sv.parameters.max_deterministic_time = det
    else: sv.parameters.max_time_in_seconds = tl
    st = sv.Solve(m)
    return {cp_model.INFEASIBLE: "INF", cp_model.FEASIBLE: "FEAS", cp_model.OPTIMAL: "FEAS"}.get(st, "UNK")


def default_cuts(I, T, h, q, n=24):
    rmin = int(h.min()); qmin = int(q.min())
    a, b = max(1, rmin + 1), T - max(1, qmin)
    if b < a: return []
    return sorted(set(np.linspace(a, b, n).round().astype(int).tolist()))


def refute_single(I, T, cfg, h, q, cuts=None, det=0.5, stats=None):
    """True if some single cut refutes T (each cut tested alone)."""
    cuts = default_cuts(I, T, h, q) if cuts is None else cuts
    for t in cuts:
        st = solve_feas(build_cuts(I, T, [t], cfg, h, q, nested=False), det=det)
        if stats is not None: stats[st] = stats.get(st, 0) + 1
        if st == "INF":
            if stats is not None: stats["tcut"] = t
            return True
    return False


def refute_nested(I, T, cfg, h, q, cuts=None, det=2.0, stats=None):
    cuts = default_cuts(I, T, h, q, 8) if cuts is None else cuts
    st = solve_feas(build_cuts(I, T, cuts, cfg, h, q, nested=True), det=det)
    if stats is not None: stats[st] = stats.get(st, 0) + 1
    return st == "INF"


def destructive(refute, lo, hi, budget=60.0):
    """lo: valid LB (integer), hi: valid UB. Gallop upwards from lo, then bisect. Returns (LB, n_tests, cpu).
    Each refuted T proves OPT > T on its own, so LB = max refuted + 1 is valid without any monotonicity."""
    t0 = time.process_time(); lo = int(math.ceil(lo - 1e-9)); hi = int(hi); best_ref = lo - 1; n = 0; step = 1
    T = lo; fail = hi           # hi is feasible: never tested
    while T < fail and time.process_time() - t0 < budget:
        n += 1
        if refute(T): best_ref = T; T += step; step *= 2
        else: fail = T; break
    a, b = best_ref, fail       # bisect in (a, b)
    while b - a > 1 and time.process_time() - t0 < budget:
        c = (a + b) // 2; n += 1
        if refute(c): a = c; best_ref = max(best_ref, c)
        else: b = c
    return best_ref + 1, n, time.process_time() - t0

# ----------------------------------------------------------------------------- single-stage exact relaxations

def stage_np_lb(p, r, m, tl=20.0, lo=0, hi=None):
    """Non-preemptive P|r_j|Cmax with integer r (exact by CP-SAT within tl; returns the proven lower bound)."""
    n = len(p)
    if n == 0: return 0
    H = int(r.max() + p.sum()) if hi is None else int(hi)
    md = cp_model.CpModel(); s = [md.NewIntVar(int(r[j]), H, "") for j in range(n)]
    iv = [md.NewFixedSizeIntervalVar(s[j], int(p[j]), "") for j in range(n)]
    if m < n: md.AddCumulative(iv, [1] * n, m)
    c = md.NewIntVar(int(lo), H, ""); [md.Add(c >= s[j] + int(p[j])) for j in range(n)]
    md.Minimize(c)
    sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.max_time_in_seconds = tl
    sv.parameters.use_overload_checker_in_cumulative = True; sv.parameters.use_timetable_edge_finding_in_cumulative = True
    st = sv.Solve(md)
    if st == cp_model.INFEASIBLE: return H + 1, True            # no schedule within the horizon H
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return int(lo), False
    return int(math.ceil(sv.BestObjectiveBound() - 1e-9)), st == cp_model.OPTIMAL


def stage_pmtn_lb(p, r, m):
    """P|pmtn, r_j|Cmax exact: smallest T with Horn-network feasibility (bisection on integers)."""
    p = np.asarray(p, np.int64); r = np.asarray(r, np.int64)
    if len(p) == 0: return 0
    lo = int(max((r + p).max(), -(-(p.sum()) // m))); hi = int(r.max() + p.sum())
    while lo < hi:
        c = (lo + hi) // 2
        if _pmtn_feasible(r, np.full(len(p), c, np.int64), p, m): hi = c
        else: lo = c + 1
    return lo

# ----------------------------------------------------------------------------- cut selection (LP hull of the knapsack)

def hull_terms(I, h):
    """Forward LB_U term with the LP (concave-envelope) knapsack K^ (parametric min-cut, g1.closure_hull):
    returns a function t -> lower estimate of the forward union term at integer cut t."""
    import g1
    V = np.array(g1.closure_hull(I), float); P2 = float(I["p2"].sum()); rmin = float(h.min()); m1, m2 = I["m1"], I["m2"]
    def f(t):
        t = np.asarray(t, float); K = np.interp(m1 * t, V[:, 0], V[:, 1])
        return np.where(K < P2, t + (P2 - np.minimum(K, m2 * np.maximum(t - rmin, 0))) / m2, -np.inf)
    return f


class CutPicker:
    def __init__(self, I, h, q):
        self.ff = hull_terms(I, h); R = reverse(I); self.fr = hull_terms(R, q)
        self.rmin, self.qmin = int(h.min()), int(q.min())

    def pick(self, T, k, mode="mix"):
        a, b = max(1, self.rmin + 1), T - max(1, self.qmin)
        if b < a: return []
        ts = np.arange(a, b + 1)
        if mode == "uniform" or ts.size <= k: return sorted(set(np.linspace(a, b, min(k, ts.size)).round().astype(int).tolist()))
        sc = np.maximum(self.ff(ts), self.fr(T - ts))
        ku = k // 2 if mode == "mix" else 0
        out = set(np.linspace(a, b, ku).round().astype(int).tolist()) if ku else set()
        gap = max(1, (b - a) // (4 * k))
        for idx in np.argsort(-sc):
            t = int(ts[idx])
            if all(abs(t - u) > gap for u in out): out.add(t)
            if len(out) >= k: break
        return sorted(out)


# ----------------------------------------------------------------------------- window shaving with the union test (B10)

def _forced(I, T, t, cfg, h, q, kind, idx, det):
    """Single-cut J(T, t) with one extra literal: kind 'x1' = outbound idx starts <= t-1; 'y0' = inbound idx has C >= t.
    INF proves (for schedules with Cmax <= T): 'x1' -> s_idx >= t; 'y0' -> C_idx <= t - 1."""
    def ex(m, cuts, Y, X):
        if not cuts: return
        if kind == "x1": m.Add(X[0][idx] == 1)
        else: m.Add(Y[0][idx] == 0)
    return solve_feas(build_cuts(I, T, [t], cfg, h, q, nested=False, extra=ex), det=det) == "INF"


def shave_windows(I, T, h, q, cfg=None, det=0.2, rounds=3, deadline=None):
    """Conditional on Cmax <= T: raise outbound heads h_j and inbound tails q_i by binary search with the forced
    J test (each INF answer is a proof by itself, so validity needs no monotonicity). Returns (h', q', refuted)."""
    cfg = cfg or Cfg(); h = h.astype(np.int64).copy(); q = q.astype(np.int64).copy(); p1, p2 = I["p1"], I["p2"]
    for _ in range(rounds):
        ch = False
        for j in range(I["n2"]):
            if deadline and time.process_time() > deadline: return h, q, False
            lo, hi = int(h[j]), int(T - p2[j])          # s_j in [h_j, T - p2_j]
            if lo > hi: return h, q, True
            a, b = lo, hi + 1                            # find largest t in (lo, hi+1] with "s_j <= t-1" refuted
            best = lo
            while b - a > 0:
                c = (a + b + 1) // 2
                if c <= lo: break
                if _forced(I, T, c, cfg, h, q, "x1", j, det): best = c; a = c
                else: b = c - 1
            if best > h[j]: h[j] = best; ch = True
            if h[j] > T - p2[j]: return h, q, True
        for i in range(I["n1"]):
            if deadline and time.process_time() > deadline: return h, q, False
            # C_i in [p1_i, T - q_i]; find smallest t with "C_i >= t" refuted -> C_i <= t - 1 -> q_i >= T - t + 1
            lo, hi = int(p1[i]) + 1, int(T - q[i])
            if lo - 1 > hi: return h, q, True
            a, b = lo, hi; best = None
            while a <= b:
                c = (a + b) // 2
                if _forced(I, T, c, cfg, h, q, "y0", i, det): best = c; b = c - 1
                else: a = c + 1
            if best is not None and T - best + 1 > q[i]: q[i] = T - best + 1; ch = True
            if T - q[i] < p1[i]: return h, q, True
        if refute_pmtn(I, T, h, q): return h, q, True
        if not ch: break
    return h, q, False


def refute_shave(I, T, h, q, det=0.2, rounds=3, budget=30.0):
    t0 = time.process_time()
    h2, q2, ref = shave_windows(I, T, h, q, det=det, rounds=rounds, deadline=t0 + budget)
    if ref: return True
    if refute_pmtn(I, T, h2, q2): return True
    lb2 = stage_np_lb(I["p2"], h2, I["m2"], tl=5.0, lo=0, hi=T + 1)[0]
    if lb2 > T: return True
    lb1 = stage_np_lb(I["p1"], q2, I["m1"], tl=5.0, lo=0, hi=T + 1)[0]
    return lb1 > T


def refute_combo(I, T, h, q, cp, det_shave=0.1, det_nest=3.0, sets=((8, "score"), (8, "mix"), (16, "uniform"), (12, "score")), budget=60.0):
    """B10 + B4: shave windows (conditional on Cmax <= T), check pmtn/non-pmtn single-stage relaxations on the shaved
    windows, then run the nested portfolio with the shaved heads/tails."""
    t0 = time.process_time()
    h2, q2, ref = shave_windows(I, T, h, q, det=det_shave, deadline=t0 + budget)
    if ref or refute_pmtn(I, T, h2, q2): return True
    if stage_np_lb(I["p2"], h2, I["m2"], tl=3.0, hi=T + 1)[0] > T: return True
    if stage_np_lb(I["p1"], q2, I["m1"], tl=3.0, hi=T + 1)[0] > T: return True
    for K, mode in sets:
        if refute_nested(I, T, Cfg(), h2, q2, cuts=cp.pick(T, K, mode), det=det_nest): return True
    return False


def dock_extra(I, T, h, q, s1=True, s2=True):
    """Dock-consistent nesting (version 3 of B4): one dock assignment per job shared by ALL cuts.
    Stage 1: jobs of dock d completed by t_k - 1 load at most t_k - 1; all jobs of dock d load at most T - min q.
    Stage 2: jobs of dock d NOT started by t_k - 1 load at most T - t_k; all jobs of dock d load at most T - min h.
    Valid: in any schedule, the jobs of one dock completed by time x are processed inside [0, x] on that dock, and the
    jobs starting at or after t on one dock are processed inside [t, T]."""
    p1, p2, m1, m2, n1, n2 = I["p1"], I["p2"], I["m1"], I["m2"], I["n1"], I["n2"]
    def ex(m, cuts, Y, X):
        for (on, n, mm, p, sel, cap_all) in ((s1, n1, m1, p1, Y, T - int(q.min())), (s2, n2, m2, p2, X, T - int(h.min()))):
            if not on or mm >= n: continue
            z = [[m.NewBoolVar("") for _ in range(mm)] for _ in range(n)]
            for a in range(n):
                m.AddExactlyOne(z[a])
                for d in range(a + 1, mm): m.Add(z[a][d] == 0)           # symmetry breaking (valid for identical docks)
            for d in range(mm): m.Add(sum(int(p[a]) * z[a][d] for a in range(n)) <= cap_all)
            for k, t in enumerate(cuts):
                for d in range(mm):
                    w = []
                    for a in range(n):
                        v = m.NewBoolVar("")
                        if sel is Y:   # v = z and y  (completed by t-1 on dock d)
                            m.AddImplication(v, z[a][d]); m.AddImplication(v, sel[k][a]); m.AddBoolOr([z[a][d].Not(), sel[k][a].Not(), v])
                        else:          # v = z and not x  (starts at or after t on dock d)
                            m.AddImplication(v, z[a][d]); m.AddImplication(v, sel[k][a].Not()); m.AddBoolOr([z[a][d].Not(), sel[k][a], v])
                        w.append(int(p[a]) * v)
                    m.Add(sum(w) <= (t - 1 if sel is Y else T - t))
    return ex


def refute_nested_docks(I, T, h, q, cuts, det=5.0, s1=True, s2=True, cfg=None):
    m = build_cuts(I, T, cuts, cfg or Cfg(), h, q, nested=True, extra=dock_extra(I, T, h, q, s1, s2))
    return solve_feas(m, det=det) == "INF"
