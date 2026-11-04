"""Benchmark parser (2026-11-01): bounds for multi-dock cross-docking P|M_j,prec|Cmax (Cota+2016 instances).

Instance: n1 inbound (m1 docks), n2 outbound (m2 docks); outbound j may start only after all
inbound i with prec[j, i] == 1 are complete. Identical parallel docks per side, no preemption.
"""
import numpy as np


def parse(path):
    tok = np.array(open(path).read().split(), dtype=np.int64)
    n1, n2, m1, m2 = (int(x) for x in tok[:4])
    p1 = tok[4:4 + n1].copy()
    p2 = tok[4 + n1:4 + n1 + n2].copy()
    prec = tok[4 + n1 + n2:4 + n1 + n2 + n1 * n2].reshape(n2, n1).astype(bool)
    assert prec.shape == (n2, n1)
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=p1, p2=p2, prec=prec)


def lb_pch(I):
    """LB of PCH step 2 (Nogueira+ 2020, Alg. 1-2): tails Delta_i, 1 machine of speed m1, Delta desc / LPT ties."""
    p1, p2, prec, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]
    S = prec.T  # (n1, n2) successors of inbound i
    sump = S @ p2
    maxp = np.where(S, p2[None, :], 0).max(axis=1)
    delta = np.maximum(sump / m2, maxp)
    order = sorted(range(I["n1"]), key=lambda i: (-delta[i], -p1[i]))  # stable: ties by J1 order
    ps = np.cumsum(p1[order]) / m1
    return float(max((ps + delta[order]).max(), (p1 + delta).max()))


def heads(I):
    """Earliest start of each outbound: max(max pred p1, sum pred p1 / m1)."""
    P = I["prec"]
    s = P @ I["p1"] / I["m1"]
    mx = np.where(P, I["p1"][None, :], 0).max(axis=1)
    return np.maximum(s, mx)


def lb_stage2_energy(I):
    """Stage-2 relaxation P|r_j,pmtn|Cmax with heads r_j: max_k r_(k) + sum_{r_l >= r_(k)} p2_l / m2; and r_j + p2_j."""
    r = heads(I); p2 = I["p2"]
    o = np.argsort(-r, kind="stable")
    tail = np.cumsum(p2[o]) / I["m2"]  # sum of p2 over jobs with r >= r_(k) (sorted desc)
    return float(max((r[o] + tail).max(), (r + p2).max()))


def lb_load(I):
    """C >= sum p1/m1 (+ min p2 if every inbound has a successor: the last inbound to finish then has one)."""
    P = I["prec"]
    extra = I["p2"][P.any(axis=1)].min() if P.any(axis=0).all() else 0
    return float(max(I["p1"].sum() / I["m1"] + extra, I["p2"].sum() / I["m2"]))


def list_ub(I, order1):
    """Feasible schedule: inbound list-scheduled in order1 on m1 docks; outbound ERD + LPT (PCH steps 4-5)."""
    p1, p2, prec, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]
    free = np.zeros(m1); C1 = np.zeros(I["n1"])
    for i in order1:
        k = free.argmin(); free[k] += p1[i]; C1[i] = free[k]
    r = np.where(prec, C1[None, :], 0).max(axis=1)
    free2 = np.zeros(m2); done = np.zeros(I["n2"], bool); C2 = np.zeros(I["n2"])
    for _ in range(I["n2"]):
        k = free2.argmin(); t = free2[k]
        avail = (~done) & (r <= t)
        if not avail.any():
            t = r[~done].min(); avail = (~done) & (r <= t)
        cand = np.flatnonzero(avail); j = cand[np.argmax(p2[cand])]
        free2[k] = max(t, r[j]) + p2[j]; C2[j] = free2[k]; done[j] = True
    return float(max(C1.max(), C2.max())), C1, C2


def cpsat(I, time_limit=60.0, hint=None, seed=0, horizon=None, strong=False, objective=True, cmax_lb=None):
    from ortools.sat.python import cp_model
    p1, p2, prec, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]
    H = int(p1.sum() + p2.sum()) if horizon is None else int(horizon)
    m = cp_model.CpModel()
    s1 = [m.NewIntVar(0, H, f"s1_{i}") for i in range(I["n1"])]
    s2 = [m.NewIntVar(0, H, f"s2_{j}") for j in range(I["n2"])]
    iv1 = [m.NewFixedSizeIntervalVar(s1[i], int(p1[i]), f"a{i}") for i in range(I["n1"])]
    iv2 = [m.NewFixedSizeIntervalVar(s2[j], int(p2[j]), f"b{j}") for j in range(I["n2"])]
    # identical parallel docks == cumulative with unit demands (interval graphs are m-colourable)
    if m1 >= I["n1"]: pass
    else: m.AddCumulative(iv1, [1] * I["n1"], m1)
    if m2 < I["n2"]: m.AddCumulative(iv2, [1] * I["n2"], m2)
    for j in range(I["n2"]):
        for i in np.flatnonzero(prec[j]):
            m.Add(s2[j] >= s1[i] + int(p1[i]))
    cmax = m.NewIntVar(0, H, "cmax")
    for j in range(I["n2"]):
        m.Add(cmax >= s2[j] + int(p2[j]))
    for i in range(I["n1"]):  # inbound without successors still occupy docks
        m.Add(cmax >= s1[i] + int(p1[i]))
    if cmax_lb is not None: m.Add(cmax >= int(cmax_lb))  # step 1: valid LB as redundant constraint
    if objective: m.Minimize(cmax)
    if hint is not None:
        C1, C2 = hint
        for i in range(I["n1"]): m.AddHint(s1[i], int(C1[i] - p1[i]))
        for j in range(I["n2"]): m.AddHint(s2[j], int(C2[j] - p2[j]))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = time_limit
    sv.parameters.num_workers = 1
    sv.parameters.random_seed = seed
    if strong:
        sv.parameters.use_overload_checker_in_cumulative = True
        sv.parameters.use_timetable_edge_finding_in_cumulative = True
        sv.parameters.use_energetic_reasoning_in_no_overlap_2d = True
    st = sv.Solve(m)
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return dict(status=sv.StatusName(st), ub=sv.ObjectiveValue() if ok else np.nan,
                lb=sv.BestObjectiveBound(), wall=sv.WallTime(),
                sched=([sv.Value(v) for v in s1], [sv.Value(v) for v in s2]) if ok else None)


def check_schedule(I, s1, s2):
    """Independent feasibility check of a schedule; returns makespan."""
    p1, p2 = I["p1"], I["p2"]
    for s, p, m in ((np.array(s1), p1, I["m1"]), (np.array(s2), p2, I["m2"])):
        ev = sorted([(t, 1) for t in s] + [(t, -1) for t in s + p], key=lambda e: (e[0], e[1]))
        c = 0
        for _, d in ev:
            c += d; assert c <= m, "dock capacity violated"
    e1 = np.array(s1) + p1
    for j in range(I["n2"]):
        pr = np.flatnonzero(I["prec"][j])
        if len(pr): assert s2[j] >= e1[pr].max(), "precedence violated"
    return float(max(e1.max(), (np.array(s2) + p2).max()))


def lb_destructive(I, lo, hi, budget=60.0, probe=6.0):
    """Destructive lower bounding: lo is a valid LB, hi a known UB. Probe 'is Cmax <= T feasible?' with CP-SAT
    (horizon T, strong cumulative propagation). Proven INFEASIBLE -> lo = T+1 (valid). FEASIBLE -> hi = T.
    UNKNOWN -> shrink the probe towards lo. Returns (lo, hi, n_probes)."""
    import time
    t0 = time.process_time(); lo = int(np.ceil(lo - 1e-9)); hi = int(hi); ceiling = hi; n = 0
    while lo < ceiling and time.process_time() - t0 < budget:
        T = (lo + ceiling) // 2 if ceiling - lo > 1 else lo
        r = cpsat(I, min(probe, budget - (time.process_time() - t0) + 0.1), horizon=T, strong=True, objective=False)
        n += 1
        if r["status"] == "INFEASIBLE": lo = T + 1
        elif r["status"] in ("FEASIBLE", "OPTIMAL"): hi = min(hi, T); ceiling = T
        else:
            if T == lo: break
            ceiling = T
    return lo, hi, n


def max_startable_work(I, t, time_limit=2.0):
    """Upper bound on the stage-2 work that can be STARTED by time t: max sum p2(A) over outbound sets A whose
    union of predecessors fits the stage-1 capacity m1*t (load bound). Returns the solver's valid upper bound."""
    from ortools.sat.python import cp_model
    p1, p2, P = I["p1"], I["p2"], I["prec"]
    m = cp_model.CpModel()
    y = [m.NewBoolVar("") for _ in range(I["n1"])]; x = [m.NewBoolVar("") for _ in range(I["n2"])]
    for j in range(I["n2"]):
        for i in np.flatnonzero(P[j]): m.AddImplication(x[j], y[i])
    m.Add(sum(int(p1[i]) * y[i] for i in range(I["n1"])) <= int(np.floor(I["m1"] * t)))
    m.Maximize(sum(int(p2[j]) * x[j] for j in range(I["n2"])))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = time_limit; sv.parameters.num_workers = 1
    st = sv.Solve(m)
    return float(sv.BestObjectiveBound()) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float(p2.sum())


def lb_union(I, lo, n_t=12, time_limit=2.0):
    """Union-of-predecessors energetic bound: for any t, C >= t + (P2 - W(t)) / m2, where W(t) bounds the stage-2
    work processed before t: W(t) <= min(max_startable_work(t), m2 * max(0, t - rmin)). Grid of t in [rmin, lo]; lo MUST be a valid LB."""
    P2 = float(I["p2"].sum()); rmin = float(heads(I).min()); best = 0.0
    for t in np.linspace(rmin, max(lo, rmin + 1), n_t):
        W = min(max_startable_work(I, t, time_limit), I["m2"] * max(0.0, t - rmin))
        if P2 - W > 0:  # some stage-2 work must remain after t; t <= lo (a valid LB) keeps t itself valid
            best = max(best, t + (P2 - W) / I["m2"])
    return max(best, lo)


def lb_union_iter(I, lo, n_t=24, time_limit=1.0, max_iter=8):
    """Fixpoint of lb_union (each round's bound is valid, so it can extend the t-grid of the next)."""
    for _ in range(max_iter):
        new = lb_union(I, lo, n_t, time_limit)
        if new <= lo + 1e-6: break
        lo = new
    return lo
