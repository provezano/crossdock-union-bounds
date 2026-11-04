"""Gap 6b: deterministic step-capped LB_U* (generated from g1.lb_union_exact by g5; see lb_union_det)."""
import sys, os, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../lower_bounds"); sys.path.insert(0, f"{D}/../benchmark_data")
from ortools.sat.python import cp_model  # noqa
import numpy as np
from cdc import heads
from g1 import UnionKnap, UnionDFS, greedy_chains


class UnionKnapDet(UnionKnap):
    def solve(self, c, det_tl, hint=None):
        cp, I = self.cp, self.I
        p1, p2 = I["p1"], I["p2"]
        m = cp.CpModel()
        y = [m.NewBoolVar("") for _ in range(I["n1"])]
        x = [m.NewBoolVar("") for _ in range(I["n2"])]
        for j, i in self.edges:
            m.AddImplication(x[j], y[i])
        m.Add(sum(int(p1[i]) * y[i] for i in range(I["n1"])) <= int(c))
        m.Maximize(sum(int(p2[j]) * x[j] for j in range(I["n2"])))
        sv = cp.CpSolver()
        sv.parameters.max_deterministic_time = det_tl
        sv.parameters.num_workers = 1
        sv.parameters.random_seed = 0
        st = sv.Solve(m)
        if st not in (cp.OPTIMAL, cp.FEASIBLE):
            return float(p2.sum()), 0.0, 0, False, None
        xs = np.array([sv.Value(v) for v in x], bool)
        cost = int(p1[I["prec"][xs].any(axis=0)].sum())
        return float(np.floor(sv.BestObjectiveBound() + 1e-6)), float(round(sv.ObjectiveValue())), cost, st == cp.OPTIMAL, xs


def lb_union_det(I, best=0.0, det_tl=2.0, max_steps=200, return_trace=False, dfs_nodes=30000):
    """Deterministic variant of g1.lb_union_exact (deterministic bound variant): the CPU budget is replaced by a cap on walk steps
    (max_steps) and every CP-SAT knapsack solve uses max_deterministic_time = det_tl instead of a wall/CPU limit,
    so the value does not depend on the machine or its load. Same validity argument (any upper bound on K is valid,
    stopping early only loses tightness)."""
    p1, p2, P = I["p1"], I["p2"], I["prec"]
    m1, m2 = I["m1"], I["m2"]
    P2 = float(p2.sum())
    rmin = float(heads(I).min())
    lb = max(best, rmin + P2 / m2)  # cap bound alone (valid: all stage-2 work runs after rmin)
    full = int(p1[P.any(axis=0)].sum())  # capacity from which K = P2 (if every outbound has preds)
    c = full - 1
    uk = UnionKnapDet(I); ud = UnionDFS(I); n_dfs = 0
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
        if steps >= max_steps:
            exact = False
            break
        r = ud.solve(c, dfs_nodes) if dfs_nodes else None
        if r is not None:
            vub = vinc = float(r[0]); cost = r[1]; opt = True; n_dfs += 1
        else:
            vub, vinc, cost, opt, xs = uk.solve(c, det_tl)
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


