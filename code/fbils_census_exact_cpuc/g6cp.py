"""exact v3 development (option A): CP-SAT with LB_U*-style union cuts reified at selected cut times, so the knapsack argument
acts at every search node instead of only at the root (v2lbu adds just Cmax >= LB*).
For a cut time T_k (integer) and makespan variable/constant C:
  forward  : b_jk <-> s2_j <= T_k - 1 (outbound started before T_k); y_ik <-> s1_i + p1_i <= T_k
             b_jk => y_ik (i in N(j));  sum p1_i y_ik <= m1 T_k;  sum p2_j (1 - b_jk) <= m2 (C - T_k)
  reversed : l_ik <-> s1_i + p1_i >= C - T_k + 1 (inbound ends inside the last T_k);  a_jk <-> s2_j >= C - T_k + 1
             l_ik => a_jk (j in S(i));  sum p2_j a_jk <= m2 T_k;  sum p1_i (1 - l_ik) <= m1 (C - T_k)
All are valid for every feasible schedule (forward: work not started before T_k runs in [T_k, C]; the started set's
predecessor union is complete by T_k; reversed = the same on the time-reversed instance). Their LP relaxation at
the root is the LP (concave-envelope) version of LB_U; during search they propagate on partial assignments."""
import sys, os, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../exact_cpsat")
import g5lib as L  # noqa
from ortools.sat.python import cp_model  # noqa
import numpy as np  # noqa


def cut_times(I, lo, hi, k):
    """k integer cut times spread over (rmin, lo]. MUST be <= a proven LB on Cmax: the cuts' right-hand sides
    m*(C - T_k) would otherwise force C >= T_k (bug caught by the known-answer test, 2026-11-02)."""
    rmin = float(L.cdc.heads(I).min()); a = int(np.floor(rmin)) + 1; b = int(lo)
    if b <= a: return []
    return sorted(set(np.linspace(a, b, k + 2)[1:-1].astype(int).tolist()))


def build(I, H, Tk, cmax_lb=None, hint=None, horizon_obj=True, fwd=True, rev=True, ht_cuts=True):
    p1, p2, prec, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]; n1, n2 = I["n1"], I["n2"]
    m = cp_model.CpModel()
    s1 = [m.NewIntVar(0, H, "") for _ in range(n1)]; s2 = [m.NewIntVar(0, H, "") for _ in range(n2)]
    iv1 = [m.NewFixedSizeIntervalVar(s1[i], int(p1[i]), "") for i in range(n1)]
    iv2 = [m.NewFixedSizeIntervalVar(s2[j], int(p2[j]), "") for j in range(n2)]
    if m1 < n1: m.AddCumulative(iv1, [1] * n1, m1)
    if m2 < n2: m.AddCumulative(iv2, [1] * n2, m2)
    E = [(j, i) for j in range(n2) for i in np.flatnonzero(prec[j])]
    for j, i in E: m.Add(s2[j] >= s1[i] + int(p1[i]))
    cm = m.NewIntVar(0, H, "cmax")
    for j in range(n2): m.Add(cm >= s2[j] + int(p2[j]))
    for i in range(n1): m.Add(cm >= s1[i] + int(p1[i]))
    if cmax_lb is not None: m.Add(cm >= int(np.ceil(cmax_lb - 1e-9)))
    if ht_cuts:
        h, tq = L.heads_tails(I)
        for j in range(n2): m.Add(s2[j] >= int(h[j]))
        for i in range(n1): m.Add(cm >= s1[i] + int(p1[i]) + int(tq[i]))
    for T in Tk:
        if fwd:
            b = [m.NewBoolVar("") for _ in range(n2)]; y = [m.NewBoolVar("") for _ in range(n1)]
            for j in range(n2):
                m.Add(s2[j] <= T - 1).OnlyEnforceIf(b[j]); m.Add(s2[j] >= T).OnlyEnforceIf(b[j].Not())
            for i in range(n1):
                m.Add(s1[i] + int(p1[i]) <= T).OnlyEnforceIf(y[i]); m.Add(s1[i] + int(p1[i]) >= T + 1).OnlyEnforceIf(y[i].Not())
            for j, i in E: m.AddImplication(b[j], y[i])
            m.Add(sum(int(p1[i]) * y[i] for i in range(n1)) <= m1 * T)
            m.Add(sum(int(p2[j]) * (1 - b[j]) for j in range(n2)) <= m2 * (cm - T))
        if rev:
            l = [m.NewBoolVar("") for _ in range(n1)]; a = [m.NewBoolVar("") for _ in range(n2)]
            for i in range(n1):
                m.Add(s1[i] + int(p1[i]) >= cm - T + 1).OnlyEnforceIf(l[i]); m.Add(s1[i] + int(p1[i]) <= cm - T).OnlyEnforceIf(l[i].Not())
            for j in range(n2):
                m.Add(s2[j] >= cm - T + 1).OnlyEnforceIf(a[j]); m.Add(s2[j] <= cm - T).OnlyEnforceIf(a[j].Not())
            for j, i in E: m.AddImplication(l[i], a[j])
            m.Add(sum(int(p2[j]) * a[j] for j in range(n2)) <= m2 * T)
            m.Add(sum(int(p1[i]) * (1 - l[i]) for i in range(n1)) <= m1 * (cm - T))
    if horizon_obj: m.Minimize(cm)
    if hint is not None:
        for i in range(n1): m.AddHint(s1[i], int(hint[0][i]))
        for j in range(n2): m.AddHint(s2[j], int(hint[1][j]))
    return m, s1, s2, cm


def solve(I, tl, H, Tk, seed=0, obj=True, **kw):
    m, s1, s2, cm = build(I, H, Tk, horizon_obj=obj, **kw)
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = tl; sv.parameters.num_workers = 1
    sv.parameters.random_seed = seed
    sv.parameters.use_overload_checker_in_cumulative = True
    sv.parameters.use_timetable_edge_finding_in_cumulative = True
    t0 = time.process_time(); st = sv.Solve(m); cpu = time.process_time() - t0
    ok = st in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    sched = ([sv.Value(v) for v in s1], [sv.Value(v) for v in s2]) if ok else None
    ub = None
    if ok:
        ub = float(L.cdc.check_schedule(I, *sched))
    return dict(status=sv.StatusName(st), ub=ub, lb=float(sv.BestObjectiveBound()) if obj else None, cpu=cpu, sched=sched)


def exact_v3(I, lo, hi, sched, tl, k=12, probe_frac=0.15, pt=5.0, seed=0, fwd=True, rev=True):
    """Same control flow as g5lib.exact_v2 (galloping decision probes <= probe_frac of the budget, then ONE optimisation
    run with Cmax >= lo, horizon hi, warm start), but every model carries the reified union cuts at k cut times."""
    t0 = time.process_time(); lo = int(np.ceil(lo - 1e-9)); hi = int(hi); log = []; n = 0; dl = 0
    left = lambda: tl - (time.process_time() - t0)
    while lo < hi and (time.process_time() - t0) < probe_frac * tl:
        T = min(lo + dl, hi - 1)
        x = solve(I, min(pt, probe_frac * tl - (time.process_time() - t0) + 0.01), T, cut_times(I, lo, T, k), seed=seed,
                  obj=False, fwd=fwd, rev=rev)
        n += 1; log.append(("b", T, x["status"], round(x["cpu"], 2)))
        if x["status"] == "INFEASIBLE": lo = T + 1; dl = max(1, 2 * dl)
        elif x["status"] in ("OPTIMAL", "FEASIBLE"):
            hi = int(x["ub"]); sched = x["sched"]
            break
        else: break
    t_probe = time.process_time() - t0
    if lo < hi and left() > 0.05:
        x = solve(I, left(), hi, cut_times(I, lo, hi, k), seed=seed, obj=True, cmax_lb=lo, hint=sched, fwd=fwd, rev=rev)
        n += 1; log.append(("o", hi, x["status"], round(x["cpu"], 2)))
        if x["sched"] is not None and x["ub"] < hi: hi = int(x["ub"]); sched = x["sched"]
        if x["status"] == "OPTIMAL": lo = hi
        else: lo = max(lo, int(np.ceil(x["lb"] - 1e-9)))
    cpu = time.process_time() - t0
    return dict(opt=hi if lo >= hi else None, lo=lo, hi=hi, cpu=cpu, t_opt=cpu if lo >= hi else None, n_probe=n,
                log=log, sched=sched, t_probe=t_probe)
