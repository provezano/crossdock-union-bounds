"""Lower-bound study (2026-11-01): exact-breakpoint union-of-predecessors LB for P|M_j,prec|Cmax (cross-docking).

K(c) = max { p2(A) : p1(Pred(A)) <= c }  (set-union / precedence-constrained knapsack, Pred(A) = union of preds).
For any t with W(t) := min(K(floor(m1 t)), m2 (t - rmin)) < P2:  Cmax >= t + (P2 - W(t)) / m2.
(If Cmax < t, every outbound job has started before t, so K(floor(m1 t)) = P2 and m2 (t - rmin) >= P2;
 hence W(t) < P2 forces Cmax >= t: no "t <= valid LB" restriction is needed.)
K is a nondecreasing step function of the integer capacity c; on a step [c0, c] with value v the sup of the bound
is reached at t -> (c+1)/m1, so it suffices to visit the Pareto steps of K (epsilon-constraint walk, exact CP-SAT
per step). Any solver upper bound on K is still valid, and stopping the walk early only loses tightness.
"""
import time
import numpy as np
from cdc import heads


class UnionKnap:
    """CP-SAT model of K(c) (rebuilt per capacity: proto mutation segfaults in this ortools build)."""

    def __init__(self, I):
        from ortools.sat.python import cp_model
        self.cp, self.I = cp_model, I
        self.edges = [(j, i) for j in range(I["n2"]) for i in np.flatnonzero(I["prec"][j])]

    def solve(self, c, tl, hint=None):
        """Returns (upper bound on K(c), incumbent value, incumbent union cost, optimal?, x)."""
        cp, I = self.cp, self.I
        p1, p2 = I["p1"], I["p2"]
        m = cp.CpModel()
        y = [m.NewBoolVar("") for _ in range(I["n1"])]
        x = [m.NewBoolVar("") for _ in range(I["n2"])]
        for j, i in self.edges:
            m.AddImplication(x[j], y[i])
        m.Add(sum(int(p1[i]) * y[i] for i in range(I["n1"])) <= int(c))
        m.Maximize(sum(int(p2[j]) * x[j] for j in range(I["n2"])))
        if hint is not None:
            for j in range(I["n2"]):
                m.AddHint(x[j], int(hint[j]))
        sv = cp.CpSolver()
        sv.parameters.max_time_in_seconds = tl
        sv.parameters.num_workers = 1
        st = sv.Solve(m)
        if st not in (cp.OPTIMAL, cp.FEASIBLE):
            return float(p2.sum()), 0.0, 0, False, None
        xs = np.array([sv.Value(v) for v in x], bool)
        cost = int(p1[I["prec"][xs].any(axis=0)].sum())
        return float(np.floor(sv.BestObjectiveBound() + 1e-6)), float(round(sv.ObjectiveValue())), cost, st == cp.OPTIMAL, xs


def greedy_chains(I, n_starts=8, seed=0):
    """Feasible closures (cost, value) from greedy inbound-removal chains: lower envelope of K for pruning."""
    p1, p2, P = I["p1"].astype(float), I["p2"].astype(float), I["prec"]
    rng = np.random.default_rng(seed)
    pts = []
    for s in range(n_starts):
        alive = np.ones(I["n2"], bool)
        while True:
            pu = P[alive].any(axis=0)
            pts.append((int(p1[pu].sum()), int(p2[alive].sum())))
            if not pu.any(): break
            loss = (P[alive] * p2[alive, None]).sum(axis=0)  # value lost by excluding inbound i
            score = np.where(pu, loss / p1, np.inf)
            if s: score = score * rng.uniform(0.7, 1.3, score.size)
            alive &= ~P[:, int(score.argmin())]
    return pts


def lb_union_exact(I, best=0.0, tl=5.0, budget=60.0, return_trace=False, dfs_nodes=30000):
    """Exact-breakpoint union LB (ceil'ed, integer data). best: any valid LB (only used for pruning)."""
    p1, p2, P = I["p1"], I["p2"], I["prec"]
    m1, m2 = I["m1"], I["m2"]
    P2 = float(p2.sum())
    rmin = float(heads(I).min())
    lb = max(best, rmin + P2 / m2)  # cap bound alone (valid: all stage-2 work runs after rmin)
    full = int(p1[P.any(axis=0)].sum())  # capacity from which K = P2 (if every outbound has preds)
    c = full - 1
    uk = UnionKnap(I); ud = UnionDFS(I); n_dfs = 0
    t0 = time.process_time()
    cs = np.arange(max(full, 1))
    thv = (cs + 1) / m1
    Kl = np.zeros(cs.size)
    for cv, vv in greedy_chains(I):
        Kl[cs >= cv] = np.maximum(Kl[cs >= cv], vv)
    raw = rmin + P2 / m2  # pure union-bound value (max of evaluated terms), independent of `best`
    for cg in np.unique(np.linspace(max(m1 * rmin, 0), full - 1, 24).astype(int)):  # warm start: cheap DFS probes
        r = ud.solve(int(cg), dfs_nodes // 10) if dfs_nodes else None
        if r is None: continue
        Kl[cs >= r[1]] = np.maximum(Kl[cs >= r[1]], r[0])
        th = (cg + 1) / m1
        if r[0] < P2 and th > rmin:
            term = th + (P2 - min(r[0], m2 * (th - rmin))) / m2
            raw = max(raw, term); lb = max(lb, term)
    fopt = lambda: np.where(Kl < P2, thv + (P2 - np.minimum(Kl, m2 * np.maximum(thv - rmin, 0))) / m2, -np.inf)
    steps = n_opt = 0
    trace = []
    exact = True
    while c >= 0:
        thi = (c + 1) / m1
        if thi + P2 / m2 <= lb + 1e-9 or thi <= rmin:  # no lower step can beat lb (K >= 0); cap binds below rmin
            break
        fo = fopt()[:c + 1]  # optimistic term (K >= Kl) for every c' <= c
        ok = np.flatnonzero(fo > lb + 1e-9)
        if ok.size == 0:
            break
        c = int(ok.max()); thi = (c + 1) / m1
        if time.process_time() - t0 > budget:
            exact = False
            break
        r = ud.solve(c, dfs_nodes) if dfs_nodes else None
        if r is not None:
            vub = vinc = float(r[0]); cost = r[1]; opt = True; n_dfs += 1
        else:
            vub, vinc, cost, opt, xs = uk.solve(c, tl)
        if vinc > 0:
            Kl[cs >= cost] = np.maximum(Kl[cs >= cost], vinc)
        steps += 1
        n_opt += opt
        exact &= opt
        if vub < P2:
            W = min(vub, m2 * (thi - rmin))
            term = thi + (P2 - W) / m2
            raw = max(raw, term); lb = max(lb, term)
        if return_trace:
            trace.append((c, cost, vub, vinc, opt))
        c = min(cost, c) - 1 if opt or vinc > 0 else c - 1
    out = dict(lb=float(np.ceil(lb - 1e-9)), raw=float(raw), steps=steps, n_dfs=n_dfs, n_opt=n_opt, exact=exact, cpu=time.process_time() - t0)
    if return_trace:
        out["trace"] = trace
    return out


def k_brute(I, c):
    """Brute-force K(c) over outbound subsets (tests only)."""
    n2 = I["n2"]
    best = 0
    for mask in range(1 << n2):
        A = np.array([(mask >> j) & 1 for j in range(n2)], bool)
        if I["p1"][I["prec"][A].any(axis=0)].sum() <= c:
            best = max(best, int(I["p2"][A].sum()))
    return best


def lb_union_brute_dense(I, n_grid=4000):
    """Reference LB on a fine t-grid with brute-force K (tests only)."""
    P2 = I["p2"].sum(); rmin = heads(I).min(); best = 0.0
    T = I["p1"].sum() + P2
    for t in np.linspace(0, T, n_grid):
        W = min(k_brute(I, int(np.floor(I["m1"] * t))), I["m2"] * max(0.0, t - rmin))
        if W < P2:
            best = max(best, t + (P2 - W) / I["m2"])
    return best


def closure_hull(I):
    """Upper concave hull of the Pareto set {(p1(Pred(A)), p2(A))} via parametric max-weight closure (min cut;
    Eisner-Severance on lambda). Returns sorted vertices (cost, value); every vertex is a real closure (feasible)
    and the piecewise-linear interpolation U(c) >= K(c) is the Lagrangian/LP bound of the knapsack."""
    from ortools.graph.python import max_flow
    from fractions import Fraction
    p1, p2, P = I["p1"], I["p2"], I["prec"]
    n1, n2 = I["n1"], I["n2"]
    S, T = n1 + n2, n1 + n2 + 1
    BIG = int(p2.sum() + 1) * int(p1.sum() + 1) * 4
    js, is_ = np.nonzero(P)

    def best(lam):  # max p2(A) - lam p1(Pred A), lam = a/b; ties -> any; returns (cost, value)
        a, b = lam.numerator, lam.denominator
        f = max_flow.SimpleMaxFlow()
        for j in range(n2): f.add_arc_with_capacity(S, n1 + j, int(p2[j]) * b)
        for i in range(n1): f.add_arc_with_capacity(i, T, int(p1[i]) * a)
        for j, i in zip(js, is_): f.add_arc_with_capacity(n1 + int(j), int(i), BIG * b)
        f.solve(S, T)
        src = set(f.get_source_side_min_cut())
        xs = np.array([(n1 + j) in src for j in range(n2)], bool)
        return int(p1[P[xs].any(axis=0)].sum()), int(p2[xs].sum())

    lo = (0, int(p2[~P.any(axis=1)].sum()))  # lam -> inf: only pred-free outbound
    hi = (int(p1[P.any(axis=0)].sum()), int(p2.sum()))  # lam = 0: everything
    V = {lo, hi}
    stack = [(lo, hi)]
    while stack:
        A, B = stack.pop()
        if B[0] == A[0]: continue
        lam = Fraction(B[1] - A[1], B[0] - A[0])
        C = best(lam)
        if (C[1] - A[1]) * (B[0] - A[0]) > (C[0] - A[0]) * (B[1] - A[1]):  # strictly above chord
            V.add(C); stack += [(A, C), (C, B)]
    return sorted(V)


def lb_union_fast(I, best=0.0, tl=2.0, budget=30.0):
    """Hull bound everywhere + exact CP-SAT only on capacities where it can still raise the LB.
    Returns dict(lb_hull, lb, steps, exact, cpu)."""
    t0 = time.process_time()
    p1, p2, P = I["p1"], I["p2"], I["prec"]; m1, m2 = I["m1"], I["m2"]
    P2 = float(p2.sum()); rmin = float(heads(I).min())
    V = np.array(closure_hull(I), float)
    full = int(p1[P.any(axis=0)].sum())
    c = np.arange(full)  # K(c) < P2 possible only for c < full
    thi = (c + 1) / m1
    U = np.interp(c, V[:, 0], V[:, 1])
    U = np.minimum(np.floor(U + 1e-9), P2)
    f = lambda W: np.where(W < P2, thi[:len(W)] + (P2 - np.minimum(W, m2 * np.maximum(thi[:len(W)] - rmin, 0))) / m2, -np.inf)
    lb_hull = max(best, rmin + P2 / m2, float(f(U).max()) if full else 0.0)
    # lower envelope of K from hull vertices (feasible closures) -> upper bound on the attainable term
    Kl = np.zeros(full)
    for cv, vv in V:
        Kl[c >= cv] = np.maximum(Kl[c >= cv], vv)
    lb = lb_hull; steps = 0; exact = True; cc = full - 1
    uk = UnionKnap(I)
    while cc >= 0:
        fup = np.maximum.accumulate(f(Kl[:cc + 1]))  # max over c' <= cc
        if fup[-1] <= lb + 1e-9: break
        if time.process_time() - t0 > budget: exact = False; break
        # jump to highest c' <= cc whose optimistic term beats lb
        cc = int(np.flatnonzero(f(Kl[:cc + 1]) > lb + 1e-9).max())
        vub, vinc, cost, opt, xs = uk.solve(cc, tl)
        steps += 1; exact &= opt
        vub = min(vub, U[cc])
        if vub < P2:
            lb = max(lb, thi[cc] + (P2 - min(vub, m2 * max(thi[cc] - rmin, 0))) / m2)
        if vinc > 0: Kl[c >= cost] = np.maximum(Kl[c >= cost], vinc)
        cc = (min(cost, cc) if vinc > 0 else cc) - 1
    return dict(lb_hull=float(np.ceil(lb_hull - 1e-9)), lb=float(np.ceil(lb - 1e-9)), steps=steps, exact=exact,
                nV=len(V), cpu=time.process_time() - t0)


class UnionDFS:
    """Exact K(c) by depth-first branch and bound over outbound sets (bitmask unions). Fast when few outbound fit
    (deep region of the walk, dense precedence). Returns None when the node limit is hit."""

    def __init__(self, I):
        p1, p2, P = I["p1"], I["p2"], I["prec"]
        self.o = np.argsort(-p2, kind="stable")
        self.v = [int(p2[j]) for j in self.o]
        self.mask = [sum(1 << int(i) for i in np.flatnonzero(P[j])) for j in self.o]
        self.p1 = [int(x) for x in p1]
        self.cache = {}

    def cost(self, m):
        c = self.cache.get(m)
        if c is None:
            c = 0; mm = m
            while mm:
                b = mm & -mm; c += self.p1[b.bit_length() - 1]; mm ^= b
            self.cache[m] = c
        return c

    def solve(self, c, node_limit=200000):
        n = len(self.v)
        cand0 = [j for j in range(n) if self.cost(self.mask[j]) <= c]
        best = [0, 0]  # value, union mask
        nodes = [0]

        def rec(cur_m, cur_v, cand):
            nodes[0] += 1
            if nodes[0] > node_limit: raise TimeoutError
            if cur_v > best[0]: best[0], best[1] = cur_v, cur_m
            if cur_v + sum(self.v[j] for j in cand) <= best[0]: return
            for k, j in enumerate(cand):
                nm = cur_m | self.mask[j]
                rest = [q for q in cand[k + 1:] if self.cost(nm | self.mask[q]) <= c]
                rec(nm, cur_v + self.v[j], rest)
                if cur_v + sum(self.v[q] for q in cand[k + 1:]) <= best[0]: return
        try:
            rec(0, 0, cand0)
        except TimeoutError:
            return None
        # free outbound (no preds) all fit trivially and are included by the search (mask 0)
        return best[0], self.cost(best[1])


def reverse(I):
    """Reversed instance (Chen & Song 2009; Chen & Lee 2009): run time backwards, swap stages, transpose precedence.
    Same optimal makespan, so any LB on it is an LB on I."""
    return dict(n1=I["n2"], n2=I["n1"], m1=I["m2"], m2=I["m1"], p1=I["p2"].copy(), p2=I["p1"].copy(),
                prec=I["prec"].T.copy())
