"""Exact-method study (2026-11-02): top-5 acceptance gaps for the cross-dock LB/UB paper.
Shared helpers: instance paths, cached LB_U* values, CP-SAT exact model (the repo's cdc.cpsat model, extended with
an optional LB constraint, optional head/tail cuts and an improvement callback), warm-start UB (constructive + ILS),
one default EACDC run."""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__))
for p in ("eacdc_matched_tim", "benchmark_data", "lower_bounds", "decoder", "eacdc_public/code"):
    q = f"{D}/../{p}"
    if q not in sys.path: sys.path.insert(0, q)
from ortools.sat.python import cp_model  # noqa: E402  (import before numpy-heavy modules, as elsewhere in repo)
import numpy as np  # noqa: E402
import cdc, g1, g2  # noqa: E402

F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
GROUPS = list(F)


def path(g, n1, seed):
    return f"{D}/../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{seed}.dat"


def g1rec(g, n1, seed):
    """Cached step 1 record (published LBs/UBs, LB_U* forward lb_u1, reversed lb_rev, exactness)."""
    return json.load(open(f"{D}/../lower_bounds/out_full/{g}_{n1}_{seed}.json"))


def lbu_star(r):
    """The paper's LB: max(LB_U*, LB_U* of the reversed instance), integral."""
    return float(max(r['lb_u1'], r['lb_rev']))


def heads_tails(I):
    """Integer valid heads of outbound jobs and tails of inbound jobs (used as optional CP-SAT cuts).
    head_j = ceil(max(p1(N(j))/m1, max_{N(j)} p1)); tail_i = ceil(max(p2(S(i))/m2, max_{S(i)} p2))."""
    h = np.ceil(cdc.heads(I) - 1e-9).astype(int)
    S = I["prec"].T
    tl = np.maximum(S @ I["p2"] / I["m2"], np.where(S, I["p2"][None, :], 0).max(axis=1))
    return h, np.ceil(tl - 1e-9).astype(int)


class _CB(cp_model.CpSolverSolutionCallback):
    def __init__(self, cm):
        super().__init__(); self.cm = cm; self.tr = []; self.t0 = time.process_time()

    def on_solution_callback(self):
        self.tr.append((time.process_time() - self.t0, self.WallTime(), int(self.Value(self.cm))))


def cpsat_exact(I, tl, hint=None, cmax_lb=None, cuts=False, horizon=None, seed=0, strong=True):
    """cdc.cpsat model (cumulative per side, precedence, Cmax), 1 worker. hint = (s1, s2) start times.
    cmax_lb: valid LB added as Cmax >= LB (the 'bound' arm). cuts: s2_j >= head_j, s1_i + p1_i + tail_i <= Cmax.
    horizon: valid UB on Cmax (variables' domain). Returns dict(status, ub, lb, cpu, wall, t_opt, trace, sched)."""
    p1, p2, prec, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]
    H = int(p1.sum() + p2.sum()) if horizon is None else int(horizon)
    m = cp_model.CpModel()
    s1 = [m.NewIntVar(0, H, f"s1_{i}") for i in range(I["n1"])]
    s2 = [m.NewIntVar(0, H, f"s2_{j}") for j in range(I["n2"])]
    iv1 = [m.NewFixedSizeIntervalVar(s1[i], int(p1[i]), f"a{i}") for i in range(I["n1"])]
    iv2 = [m.NewFixedSizeIntervalVar(s2[j], int(p2[j]), f"b{j}") for j in range(I["n2"])]
    if m1 < I["n1"]: m.AddCumulative(iv1, [1] * I["n1"], m1)
    if m2 < I["n2"]: m.AddCumulative(iv2, [1] * I["n2"], m2)
    for j in range(I["n2"]):
        for i in np.flatnonzero(prec[j]):
            m.Add(s2[j] >= s1[i] + int(p1[i]))
    cm = m.NewIntVar(0, H, "cmax")
    for j in range(I["n2"]): m.Add(cm >= s2[j] + int(p2[j]))
    for i in range(I["n1"]): m.Add(cm >= s1[i] + int(p1[i]))
    if cmax_lb is not None: m.Add(cm >= int(np.ceil(cmax_lb - 1e-9)))
    if cuts:
        h, tq = heads_tails(I)
        for j in range(I["n2"]): m.Add(s2[j] >= int(h[j]))
        for i in range(I["n1"]): m.Add(cm >= s1[i] + int(p1[i]) + int(tq[i]))
    m.Minimize(cm)
    if hint is not None:
        for i in range(I["n1"]): m.AddHint(s1[i], int(hint[0][i]))
        for j in range(I["n2"]): m.AddHint(s2[j], int(hint[1][j]))
    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = tl
    sv.parameters.num_workers = 1
    sv.parameters.random_seed = seed
    if strong:
        sv.parameters.use_overload_checker_in_cumulative = True
        sv.parameters.use_timetable_edge_finding_in_cumulative = True
        sv.parameters.use_energetic_reasoning_in_no_overlap_2d = True
    cb = _CB(cm); t0 = time.process_time()
    st = sv.Solve(m, cb)
    cpu = time.process_time() - t0
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    sched = ([sv.Value(v) for v in s1], [sv.Value(v) for v in s2]) if ok else None
    if sched is not None:
        mk = cdc.check_schedule(I, *sched); assert abs(mk - sv.ObjectiveValue()) < 1e-6
    return dict(status=sv.StatusName(st), ub=float(sv.ObjectiveValue()) if ok else None,
                lb=float(sv.BestObjectiveBound()), cpu=cpu, wall=sv.WallTime(),
                t_opt=cpu if st == cp_model.OPTIMAL else None, trace=cb.tr, sched=sched)


def warm_ub(I, ils_budget, seed=1):
    """Best step 2 constructive + ILS (frozen constants) for ils_budget CPU s. Returns (ub, (s1, s2), cpu, trace)."""
    import g3
    t0 = time.process_time()
    C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    b, bo, tr = g3.ils_p(I, side, order, max(0.0, ils_budget - (time.process_time() - t0)), seed=100 + seed,
                         t_offset=time.process_time() - t0)
    cm, s1, s2 = g2.schedule_from(I, side, bo)
    assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6
    return float(b), (list(map(int, s1)), list(map(int, s2))), time.process_time() - t0, tr, float(C[kb][0])


def eacdc_default(fp, seed=1):
    """One full default EACDC run (authors' GA.py, static configuration), best schedule ever evaluated."""
    import g3
    e = g3.eacdc(None, None, None, seed, budget=None, fpath=fp)
    return dict(best=float(e["best"]), cpu=e["cpu"], trace=e["trace"])


def exact_hybrid(I, lo, hi, sched, tl, pt0=4.0, cuts=True, seed=0):
    """Exact method used for step 1 (same CP-SAT model). Rounds of: (a) bottom probes in decision form at
    T = lo + dl (horizon T, head/tail cuts; INFEASIBLE -> lo = T + 1 and dl doubles; timeout -> dl halves, stop the
    phase), then (b) an optimisation run with Cmax >= lo, horizon hi, hint = incumbent (improves hi; OPTIMAL ->
    done). Probe/run limits double every round. lo must be a valid LB, sched a feasible schedule of makespan hi.
    Returns dict(opt or None, lo, hi, cpu, t_opt, n_probe, log, sched)."""
    t0 = time.process_time(); lo = int(np.ceil(lo - 1e-9)); hi = int(hi); pt = pt0; log = []; n = 0; dl = 0
    left = lambda: tl - (time.process_time() - t0)
    while lo < hi and left() > 0.05:
        while lo < hi and left() > 0.05:          # (a) bottom probes
            T = min(lo + dl, hi - 1)
            x = cpsat_exact(I, min(pt / 2, left()), horizon=T, cuts=cuts, seed=seed)
            n += 1; log.append(("b", T, x["status"], round(x["cpu"], 2)))
            if x["status"] == "INFEASIBLE": lo = T + 1; dl = max(1, 2 * dl)
            elif x["status"] in ("OPTIMAL", "FEASIBLE"):
                hi = int(x["ub"]); sched = x["sched"]
                if x["status"] == "OPTIMAL": lo = hi
                break
            else: dl //= 2; break
        if lo >= hi or left() <= 0.05: break
        x = cpsat_exact(I, min(pt, left()), hint=sched, horizon=hi, cmax_lb=lo, cuts=cuts, seed=seed)  # (b)
        n += 1; log.append(("o", hi, x["status"], round(x["cpu"], 2)))
        if x["sched"] is not None and x["ub"] < hi: hi = int(x["ub"]); sched = x["sched"]
        if x["status"] == "OPTIMAL": lo = hi
        else: lo = max(lo, int(np.ceil(x["lb"] - 1e-9)))
        pt *= 2
    cpu = time.process_time() - t0
    return dict(opt=hi if lo >= hi else None, lo=lo, hi=hi, cpu=cpu, t_opt=cpu if lo >= hi else None, n_probe=n,
                log=log, sched=sched)


def exact_bisect(I, lo, hi, sched, tl, pt0=5.0, cuts=True, seed=0):
    """Exact two-sided probing with the same CP-SAT model in decision form (horizon T = Cmax <= T, strong
    cumulative propagation, head/tail cuts). lo: valid LB, hi: makespan of `sched` (feasible). Alternates a bottom
    probe at T = lo + dl (galloping dl; INFEASIBLE -> lo = T + 1) and a top probe at T = hi - 1 (FEASIBLE ->
    hi = new makespan, sched updated; INFEASIBLE -> OPT = hi). A probe that times out doubles its limit.
    Returns dict(opt or None, lo, hi, cpu, t_opt, n_probe, log)."""
    t0 = time.process_time(); lo = int(np.ceil(lo - 1e-9)); hi = int(hi); pt = {"b": pt0, "t": pt0}; log = []; n = 0; dl = 0
    side = "b"
    while lo < hi:
        left = tl - (time.process_time() - t0)
        if left <= 0.05: break
        T = min(lo + dl, hi - 1) if side == "b" else hi - 1
        x = cpsat_exact(I, min(pt[side], left), hint=sched if side == "t" else None, horizon=T, cuts=cuts, seed=seed)
        n += 1; log.append((side, T, x["status"], round(x["cpu"], 2)))
        if x["status"] == "INFEASIBLE":
            lo = T + 1 if side == "b" else hi          # no schedule with Cmax <= T
            if side == "b": dl = max(1, 2 * dl)        # gallop upwards after a proven infeasibility
        elif x["status"] == "OPTIMAL":                 # minimisation within the horizon: proven optimum
            hi = lo = int(x["ub"]); sched = x["sched"]
        elif x["status"] == "FEASIBLE":
            hi = int(x["ub"]); sched = x["sched"]
        else:
            pt[side] *= 2
            if side == "b": dl //= 2                   # timed out: probe closer to lo next time
        side = "t" if side == "b" else "b"
    cpu = time.process_time() - t0
    return dict(opt=hi if lo >= hi else None, lo=lo, hi=hi, cpu=cpu, t_opt=cpu if lo >= hi else None, n_probe=n,
                log=log, sched=sched)


def exact_v2(I, lo, hi, sched, tl, probe_frac=0.15, pt=5.0, cuts=True, seed=0):
    """Exact method v2 (after profiling showed that restarting optimisation runs loses CP-SAT's learning):
    (a) bottom probes in decision form at T = lo + dl (galloping, each <= pt s) for at most probe_frac of the
    budget, stopping at the first timeout; (b) ONE optimisation run of the repo model for the rest of the budget,
    with Cmax >= lo (raised), horizon hi, hint = incumbent and head/tail cuts."""
    t0 = time.process_time(); lo = int(np.ceil(lo - 1e-9)); hi = int(hi); log = []; n = 0; dl = 0
    left = lambda: tl - (time.process_time() - t0)
    while lo < hi and (time.process_time() - t0) < probe_frac * tl:
        T = min(lo + dl, hi - 1)
        x = cpsat_exact(I, min(pt, probe_frac * tl - (time.process_time() - t0) + 0.01), horizon=T, cuts=cuts, seed=seed)
        n += 1; log.append(("b", T, x["status"], round(x["cpu"], 2)))
        if x["status"] == "INFEASIBLE": lo = T + 1; dl = max(1, 2 * dl)
        elif x["status"] in ("OPTIMAL", "FEASIBLE"):
            hi = int(x["ub"]); sched = x["sched"]
            if x["status"] == "OPTIMAL": lo = hi
            break
        else: break
    t_probe = time.process_time() - t0
    if lo < hi and left() > 0.05:
        x = cpsat_exact(I, left(), hint=sched, horizon=hi, cmax_lb=lo, cuts=cuts, seed=seed)
        n += 1; log.append(("o", hi, x["status"], round(x["cpu"], 2)))
        if x["sched"] is not None and x["ub"] < hi: hi = int(x["ub"]); sched = x["sched"]
        if x["status"] == "OPTIMAL": lo = hi
        else: lo = max(lo, int(np.ceil(x["lb"] - 1e-9)))
    cpu = time.process_time() - t0
    return dict(opt=hi if lo >= hi else None, lo=lo, hi=hi, cpu=cpu, t_opt=cpu if lo >= hi else None, n_probe=n,
                log=log, sched=sched, t_probe=t_probe)
