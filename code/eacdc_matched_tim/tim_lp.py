"""LP relaxation of the Cota+ 2016 time-indexed model (TIM), eqs. (10)-(15), solved with HiGHS.
x[i,e,t]=1 if job e of stage i starts at t. Horizon T: any valid upper bound on OPT (default sum of all p).
Instance dict as cdc.parse: p1, p2 (int arrays), prec (n2 x n1 bool), m1, m2."""
import time
import numpy as np
import highspy


def tim_lp(I, T=None, time_limit=None):
    p1 = np.asarray(I["p1"], int); p2 = np.asarray(I["p2"], int); P = np.asarray(I["prec"], bool)
    n1, n2, m1, m2 = len(p1), len(p2), int(I["m1"]), int(I["m2"])
    T = int(p1.sum() + p2.sum()) if T is None else int(T)
    # column layout: x1 blocks, x2 blocks, r2 (n2), Cmax
    off1 = np.concatenate([[0], np.cumsum(T - p1 + 1)]); N1 = off1[-1]
    off2 = N1 + np.concatenate([[0], np.cumsum(T - p2 + 1)]); N2 = off2[-1]
    ir = N2; ic = N2 + n2; ncol = ic + 1
    rows, cols, vals, lo, hi = [], [], [], [], []
    nr = 0

    def add(cidx, cval, l, h):
        nonlocal nr
        rows.append(np.full(len(cidx), nr)); cols.append(np.asarray(cidx)); vals.append(np.asarray(cval, float))
        lo.append(l); hi.append(h); nr += 1

    def block(stage, e):
        o = off1 if stage == 1 else off2; return np.arange(o[e], o[e + 1])
    # (10) assignment
    for e in range(n1): add(block(1, e), np.ones(T - p1[e] + 1), 1, 1)
    for e in range(n2): add(block(2, e), np.ones(T - p2[e] + 1), 1, 1)
    # (11) capacity: x[e,s] counts in rows t=s..s+p-1  (built vectorised)
    for stage, p, m, o in ((1, p1, m1, off1), (2, p2, m2, off2)):
        base = nr; R, C = [], []
        for e in range(len(p)):
            s = np.arange(T - p[e] + 1)
            R.append((s[:, None] + np.arange(p[e])[None, :]).ravel()); C.append(np.repeat(o[e] + s, p[e]))
        R = np.concatenate(R); C = np.concatenate(C)
        rows.append(base + R); cols.append(C); vals.append(np.ones(len(R)))
        lo += [-np.inf] * T; hi += [m] * T; nr += T
    # (12) r2e - sum_t (t+p1e') x1e't >= 0 ; (13) sum_t t x2et - r2e >= 0
    for e in range(n2):
        for ep in np.flatnonzero(P[e]):
            b = block(1, ep); add(np.r_[ir + e, b], np.r_[1.0, -(np.arange(len(b)) + p1[ep])], 0, np.inf)
        b = block(2, e); add(np.r_[b, ir + e], np.r_[np.arange(len(b)), -1.0], 0, np.inf)
    # (14)-(15) Cmax - sum t x >= p
    for stage, p in ((1, p1), (2, p2)):
        for e in range(len(p)):
            b = block(stage, e); add(np.r_[ic, b], np.r_[1.0, -np.arange(len(b), dtype=float)], p[e], np.inf)
    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    order = np.lexsort((R, C)); R, C, V = R[order], C[order], V[order]
    start = np.searchsorted(C, np.arange(ncol + 1))
    lp = highspy.HighsLp()
    lp.num_col_ = ncol; lp.num_row_ = nr
    cost = np.zeros(ncol); cost[ic] = 1.0
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(ncol); cu = np.ones(ncol); cu[ir:] = T; lp.col_upper_ = cu
    lp.row_lower_ = np.array(lo, float); lp.row_upper_ = np.array(hi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = start.astype(np.int32); lp.a_matrix_.index_ = R.astype(np.int32); lp.a_matrix_.value_ = V
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("solver", "ipm") if ncol > 200000 else None
    if time_limit: h.setOptionValue("time_limit", float(time_limit))
    h.passModel(lp)
    t0 = time.process_time(); h.run(); cpu = time.process_time() - t0
    st = h.getModelStatus()
    ok = st == highspy.HighsModelStatus.kOptimal
    info = h.getInfo()
    # a valid LB even if not optimal: dual bound only when optimal; report primal objective if optimal
    return dict(lb=float(info.objective_function_value) if ok else None, status=str(st), cpu=cpu, T=T,
                ncol=int(ncol), nrow=int(nr))


COTA_EXAMPLE = dict(p1=np.array([29, 45, 43]), p2=np.array([17, 39]), m1=2, m2=2,
                    prec=np.array([[0, 0, 1], [1, 1, 1]], bool))  # S1={j13}, S2={all}
