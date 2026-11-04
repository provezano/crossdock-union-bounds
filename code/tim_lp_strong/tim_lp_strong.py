"""TIM-LP (Cota+ 2016 eqs. 10-15, as in eacdc_matched_tim/tim_lp.py) plus strengthened makespan linking.
Baseline-strengthening check (preregistered; see the paper).

Families (all valid for every integer schedule on [0, T], T >= OPT):
  A  step linking: z_t in [0,1] (t = 0..T-1) = [Cmax > t];  Cmax >= sum_t z_t;  z_t >= z_{t+1};
     m2 z_t >= u2_t (stage-2 occupancy of slot t);  m1 z_{t+q0} >= u1_t,  q0 = min_j q_j,
     q_j = max p2 over outbound successors of inbound j (0 if none).
  B  job step rows (separated): z_t >= sum_{s : s + p_j + q_j > t} x_js   (job j, incl. tail, still running after t).
  C  reversed parallel-machine Queyranne cuts (separated), per stage, horizon shifted by tail theta:
     sum_{j in S} p_j (Cmax - theta - s_j) >= (p(S)^2 / m + sum_{j in S} p_j^2) / 2,  S within {j : q_j >= theta}.
Variants: 'base' (= tim_lp.py), 'A', 'AC', 'ABC' (= TIM-LP+). Solver: HiGHS IPM with crossover, 1 thread, every round
(pilot 2026-11-03: dual simplex > 12 min on one n1 = 20 LP that IPM solves in 65 s; warm-started simplex rounds also slower).  Every LP solved in the loop is a relaxation, so the
objective of the last optimally solved round is a valid lower bound."""
import time
import numpy as np
import highspy

INF = highspy.kHighsInf


def _qtail(I):
    P = np.asarray(I["prec"], bool); p2 = np.asarray(I["p2"], int)
    return np.where(P.T, p2[None, :], 0).max(axis=1).astype(int)  # (n1,)


def tim_lp_strong(I, T=None, variant="ABC", time_limit=None, max_rounds=60, tol=1e-6, per_round=200, solver="ipm"):
    p1 = np.asarray(I["p1"], int); p2 = np.asarray(I["p2"], int); P = np.asarray(I["prec"], bool)
    n1, n2, m1, m2 = len(p1), len(p2), int(I["m1"]), int(I["m2"])
    T = int(p1.sum() + p2.sum()) if T is None else int(T)
    useA = "A" in variant; useB = "B" in variant; useC = "C" in variant
    q1 = _qtail(I); q0 = int(q1.min()) if n1 else 0
    off1 = np.concatenate([[0], np.cumsum(T - p1 + 1)]); N1 = off1[-1]
    off2 = N1 + np.concatenate([[0], np.cumsum(T - p2 + 1)]); N2 = off2[-1]
    ir = N2; ic = N2 + n2; iz = ic + 1; iu = iz + T; ncol = iz + (3 * T if useA else 0)  # z_t, u1_t, u2_t
    rows, cols, vals, lo, hi = [], [], [], [], []
    nr = 0

    def add(cidx, cval, l, h):
        nonlocal nr
        rows.append(np.full(len(cidx), nr)); cols.append(np.asarray(cidx)); vals.append(np.asarray(cval, float))
        lo.append(l); hi.append(h); nr += 1

    def block(stage, e):
        o = off1 if stage == 1 else off2; return np.arange(o[e], o[e + 1])
    for e in range(n1): add(block(1, e), np.ones(T - p1[e] + 1), 1, 1)                      # (10)
    for e in range(n2): add(block(2, e), np.ones(T - p2[e] + 1), 1, 1)
    for stage, p, m, o in ((1, p1, m1, off1), (2, p2, m2, off2)):                            # (11)
        base = nr; R, C = [], []
        for e in range(len(p)):
            s = np.arange(T - p[e] + 1)
            R.append((s[:, None] + np.arange(p[e])[None, :]).ravel()); C.append(np.repeat(o[e] + s, p[e]))
        R = np.concatenate(R); C = np.concatenate(C)
        rows.append(base + R); cols.append(C); vals.append(np.ones(len(R)))
        if not useA:
            lo += [-np.inf] * T; hi += [m] * T; nr += T
        else:  # capacity row becomes u_t - sum x = 0 with 0 <= u_t <= m (same LP), then m z_{t+shift} - u_t >= 0
            uo = iu + (0 if stage == 1 else T); tt = np.arange(T)
            rows.append(base + tt); cols.append(uo + tt); vals.append(-np.ones(T))
            lo += [0.0] * T; hi += [0.0] * T; nr += T
            shift = q0 if stage == 1 else 0; tt = tt[tt + shift <= T - 1]; base = nr
            rows.append(base + np.arange(len(tt))); cols.append(uo + tt); vals.append(-np.ones(len(tt)))
            rows.append(base + np.arange(len(tt))); cols.append(iz + tt + shift); vals.append(np.full(len(tt), float(m)))
            lo += [0.0] * len(tt); hi += [np.inf] * len(tt); nr += len(tt)
    for e in range(n2):                                                                       # (12)-(13)
        for ep in np.flatnonzero(P[e]):
            b = block(1, ep); add(np.r_[ir + e, b], np.r_[1.0, -(np.arange(len(b)) + p1[ep])], 0, np.inf)
        b = block(2, e); add(np.r_[b, ir + e], np.r_[np.arange(len(b)), -1.0], 0, np.inf)
    for stage, p in ((1, p1), (2, p2)):                                                       # (14)-(15)
        for e in range(len(p)):
            b = block(stage, e); add(np.r_[ic, b], np.r_[1.0, -np.arange(len(b), dtype=float)], p[e], np.inf)
    if useA:
        add(np.r_[ic, iz + np.arange(T)], np.r_[1.0, -np.ones(T)], 0, np.inf)                  # Cmax >= sum z
        for t in range(T - 1): add([iz + t, iz + t + 1], [1.0, -1.0], 0, np.inf)               # monotone
    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    order = np.lexsort((R, C)); R, C, V = R[order], C[order], V[order]
    start = np.searchsorted(C, np.arange(ncol + 1))
    lp = highspy.HighsLp(); lp.num_col_ = ncol; lp.num_row_ = nr
    cost = np.zeros(ncol); cost[ic] = 1.0; lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(ncol); cu = np.ones(ncol); cu[ir:iz] = T
    if useA: cu[iu:iu + T] = m1; cu[iu + T:iu + 2 * T] = m2
    lp.col_upper_ = cu
    lp.row_lower_ = np.array(lo, float); lp.row_upper_ = np.array(hi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = start.astype(np.int32); lp.a_matrix_.index_ = R.astype(np.int32); lp.a_matrix_.value_ = V
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    if solver: h.setOptionValue("solver", "ipm" if solver == "ipm_then_simplex" else solver)
    elif ncol > 200000: h.setOptionValue("solver", "ipm")
    h.passModel(lp)
    w0 = time.time(); t0 = time.process_time(); best = None; rounds = 0; ncuts = dict(B=0, C=0); status = None
    tt1 = [np.arange(T - p1[e] + 1, dtype=float) for e in range(n1)]
    tt2 = [np.arange(T - p2[e] + 1, dtype=float) for e in range(n2)]
    while True:
        if time_limit:
            rem = float(time_limit) - (time.time() - w0)  # HiGHS time_limit is wall clock
            if rem <= 0: status = "time_limit"; break
            h.setOptionValue("time_limit", rem)
        h.run(); st = h.getModelStatus()
        if rounds == 0 and solver == "ipm_then_simplex": h.setOptionValue("solver", "simplex")  # warm-start cut rounds
        if st != highspy.HighsModelStatus.kOptimal: status = str(st); break
        best = float(h.getInfo().objective_function_value); status = "optimal"
        if not (useB or useC) or rounds >= max_rounds: break
        x = np.asarray(h.getSolution().col_value); cmax = x[ic]
        new = []  # (idx, val, lower)
        if useC:
            for stage, p, m, tts, q in ((1, p1, m1, tt1, q1), (2, p2, m2, tt2, np.zeros(n2, int))):
                o = off1 if stage == 1 else off2
                s = np.array([x[o[e]:o[e + 1]] @ tts[e] for e in range(len(p))])
                for th in np.unique(q):
                    J = np.flatnonzero(q >= th)
                    cp = cmax - th - s[J]; od = J[np.argsort(cp, kind="stable")]
                    ps = np.cumsum(p[od]); pq = np.cumsum(p[od] ** 2); lhs = np.cumsum(p[od] * (cmax - th - s[od]))
                    rhs = 0.5 * (ps ** 2 / m + pq); viol = rhs - lhs
                    k = int(np.argmax(viol))
                    if viol[k] > tol * max(1.0, rhs[k]):
                        S = od[:k + 1]
                        idx = np.concatenate([[ic]] + [np.arange(o[e], o[e + 1]) for e in S])
                        val = np.concatenate([[float(ps[k])]] + [-p[e] * tts[e] for e in S])
                        new.append((idx, val, float(rhs[k] + th * ps[k]), "C"))
        if useB:
            z = x[iz:iz + T]; cand = []
            for stage, p, q, o in ((1, p1, q1, off1), (2, p2, np.zeros(n2, int), off2)):
                for e in range(len(p)):
                    xe = x[o[e]:o[e + 1]]; suf = np.cumsum(xe[::-1])[::-1]  # suf[s] = sum_{s'>=s} x
                    L = p[e] + q[e]
                    t = np.arange(T); s0 = np.maximum(t - L + 1, 0)
                    run = np.where(s0 < len(xe), suf[np.minimum(s0, len(xe) - 1)], 0.0)
                    v = run - z; k = int(np.argmax(v))
                    if v[k] > tol: cand.append((v[k], stage, e, k, int(s0[k])))
            cand.sort(reverse=True)
            for v, stage, e, k, s0 in cand[:per_round]:
                o = off1 if stage == 1 else off2; b = np.arange(o[e] + s0, o[e + 1])
                new.append((np.r_[iz + k, b], np.r_[1.0, -np.ones(len(b))], 0.0, "B"))
        if not new: break
        for idx, val, l, fam in new:
            h.addRow(l, INF, len(idx), np.asarray(idx, np.int32), np.asarray(val, float)); ncuts[fam] += 1
        rounds += 1
    cpu = time.process_time() - t0; wall = time.time() - w0
    return dict(lb=best, status=status, cpu=cpu, wall=wall, solver=solver, T=T, ncol=int(ncol), nrow=int(h.getNumRow()), rounds=rounds,
                cuts_B=ncuts["B"], cuts_C=ncuts["C"], q0=q0, variant=variant)
