"""Gap 6a: generic energetic-reasoning lower bound for the two-stage cross-dock (Neron, Baptiste & Gupta 2001 style,
specialised to bipartite precedence), as a comparator for LB_U*. No search: pure propagation + destructive
(dichotomic) search on the trial makespan T.
For a trial T: windows inbound [0, d1_i], outbound [r2_j, T]; initial d1_i = T - ceil(tail_i), r2_j = ceil(head_j)
(head/tail as in LB_e2/PCH). Fixpoint of
  (1) energetic satisfiability test per stage on the O(n^2) relevant intervals (Baptiste et al. time points),
  (2) energetic time-bound adjustments (unit demand on capacity m: r_i >= t2 - SL, d_i <= t1 + SL when the
      left/right work of i exceeds the slack SL of the others),
  (3) precedence propagation r2_j >= max_{i in N(j)} r1_i + p1_i, d1_i <= min_{j in S(i)} d2_j - p2_j,
      and the parallel-machine head/tail refreshes r2_j >= ceil(p1(N(j)) / m1) (stage-1 work of N(j) starts at >= 0).
Refuted (some window empty or a test fails) => Cmax > T. Propagation is monotone in T, so the smallest
non-refuted T in [lo, hi] (binary search) is a valid LB: LB_ER."""
import numpy as np, math, time


def _work(r, d, p, t1, t2):
    """W(i, t1, t2) for all jobs and all (t1, t2) pairs: arrays (n1t, n2t, n)."""
    L = t2[None, :, None] - t1[:, None, None]
    left = np.maximum(0, r[None, None, :] + p[None, None, :] - t1[:, None, None])
    right = np.maximum(0, t2[None, :, None] - d[None, None, :] + p[None, None, :])
    return np.minimum(np.minimum(L, p[None, None, :]), np.minimum(left, right)), L, left, right


def _stage(r, d, p, m):
    """One energetic pass on a stage. Returns (feasible, r, d) with adjusted windows."""
    if np.any(r + p > d): return False, r, d
    if m >= len(p): return True, r, d
    T1 = np.unique(np.concatenate([r, d - p, r + p]))
    T2 = np.unique(np.concatenate([d, r + p, d - p]))
    W, L, left, right = _work(r, d, p, T1, T2)
    L2 = L[..., 0]; valid = L2 > 0
    tot = W.sum(axis=2)
    if np.any((tot > m * L2) & valid): return False, r, d
    SL = (m * L2)[..., None] - tot[..., None] + W            # slack left for job i in [t1, t2]
    vm = valid[..., None]
    # adjustments: if left work exceeds slack, job cannot start before t2 - SL (and must fit fully)
    wl = np.minimum(np.minimum(L, p[None, None, :]), left)
    need = vm & (wl > SL)
    if need.any():
        cand = np.where(need, T2[None, :, None] - SL, -np.inf).max(axis=(0, 1))
        r = np.maximum(r, np.ceil(cand - 1e-9))
    wr = np.minimum(np.minimum(L, p[None, None, :]), right)
    need = vm & (wr > SL)
    if need.any():
        cand = np.where(need, T1[:, None, None] + SL, np.inf).min(axis=(0, 1))
        d = np.minimum(d, np.floor(cand + 1e-9))
    return not np.any(r + p > d), r, d


def refute(I, T, max_iter=50):
    """True if energetic propagation proves that no schedule has Cmax <= T."""
    p1, p2, P = I["p1"].astype(float), I["p2"].astype(float), I["prec"]; m1, m2 = I["m1"], I["m2"]
    S = P.T
    head = np.ceil(np.maximum(P @ p1 / m1, np.where(P, p1[None, :], 0).max(axis=1)) - 1e-9)
    tail = np.ceil(np.maximum(S @ p2 / m2, np.where(S, p2[None, :], 0).max(axis=1)) - 1e-9)
    r1 = np.zeros(I["n1"]); d1 = T - tail; r2 = head.copy(); d2 = np.full(I["n2"], float(T))
    for _ in range(max_iter):
        old = (r1.copy(), d1.copy(), r2.copy(), d2.copy())
        ok, r1, d1 = _stage(r1, d1, p1, m1)
        if not ok: return True
        ok, r2, d2 = _stage(r2, d2, p2, m2)
        if not ok: return True
        r2 = np.maximum(r2, np.where(P, (r1 + p1)[None, :], 0).max(axis=1))
        d1 = np.minimum(d1, np.where(S, (d2 - p2)[None, :], np.inf).min(axis=1))
        if all(np.array_equal(a, b) for a, b in zip(old, (r1, d1, r2, d2))): return False
    return False


def lb_er(I, lo, hi):
    """Smallest T in [lo, hi] not refuted (lo valid LB, hi valid UB). Returns (LB_ER, n_checks, cpu)."""
    t0 = time.process_time(); lo, hi = int(math.ceil(lo - 1e-9)), int(hi); n = 0
    if not refute(I, lo): return lo, 1, time.process_time() - t0
    a, b = lo, hi        # refuted(a) True; refuted(b) False (b is feasible)
    while b - a > 1:
        c = (a + b) // 2; n += 1
        if refute(I, c): a = c
        else: b = c
    return b, n + 1, time.process_time() - t0
