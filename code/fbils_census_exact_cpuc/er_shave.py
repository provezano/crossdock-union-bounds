"""RER-style comparator (after Hidri & Haouari 2011 / Neron+ 2001 'global operations'): the energetic propagation of
exact_cpsat/er.py (per-stage energetic tests + time-bound adjustments + cross-stage window propagation) plus
SHAVING: for each job, tentatively restrict its start to the lower (upper) half of its window; if propagation refutes
that, the job's window is cut accordingly (valid: the refuted half contains no schedule with Cmax <= T). Passes repeat
until no window changes (max_pass). Destructive dichotomic search on T as in er.lb_er."""
import sys, os, math, time, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../exact_cpsat")
import er


def _prop(I, W, max_iter=50):
    p1, p2, P = I["p1"].astype(float), I["p2"].astype(float), I["prec"]; S = P.T
    r1, d1, r2, d2 = (w.copy() for w in W)
    for _ in range(max_iter):
        old = (r1.copy(), d1.copy(), r2.copy(), d2.copy())
        ok, r1, d1 = er._stage(r1, d1, p1, I["m1"])
        if not ok: return None
        ok, r2, d2 = er._stage(r2, d2, p2, I["m2"])
        if not ok: return None
        r2 = np.maximum(r2, np.where(P, (r1 + p1)[None, :], 0).max(axis=1))
        d1 = np.minimum(d1, np.where(S, (d2 - p2)[None, :], np.inf).min(axis=1))
        if np.any(r1 + p1 > d1) or np.any(r2 + p2 > d2): return None
        if all(np.array_equal(a, b) for a, b in zip(old, (r1, d1, r2, d2))): break
    return r1, d1, r2, d2


def refute_shave(I, T, max_pass=3, deadline=None):
    p1, p2, P = I["p1"].astype(float), I["p2"].astype(float), I["prec"]; S = P.T
    head = np.ceil(np.maximum(P @ p1 / I["m1"], np.where(P, p1[None, :], 0).max(axis=1)) - 1e-9)
    tail = np.ceil(np.maximum(S @ p2 / I["m2"], np.where(S, p2[None, :], 0).max(axis=1)) - 1e-9)
    W = _prop(I, (np.zeros(I["n1"]), T - tail, head.copy(), np.full(I["n2"], float(T))))
    if W is None: return True
    for _ in range(max_pass):
        changed = False
        for st, p in ((0, p1), (1, p2)):
            for k in range(len(p)):
                if deadline and time.process_time() > deadline: return False
                r, d = W[2 * st], W[2 * st + 1]; lo, hi = r[k], d[k] - p[k]   # start window [lo, hi]
                if hi - lo < 1: continue
                mid = math.floor((lo + hi) / 2)
                Wa = [w.copy() for w in W]; Wa[2 * st + 1][k] = mid + p[k]          # start in [lo, mid]
                if _prop(I, Wa) is None:
                    Wn = [w.copy() for w in W]; Wn[2 * st][k] = mid + 1; W2 = _prop(I, Wn)
                    if W2 is None: return True
                    W = list(W2); changed = True; continue
                Wb = [w.copy() for w in W]; Wb[2 * st][k] = mid + 1                 # start in [mid+1, hi]
                if _prop(I, Wb) is None:
                    Wn = [w.copy() for w in W]; Wn[2 * st + 1][k] = mid + p[k]; W2 = _prop(I, Wn)
                    if W2 is None: return True
                    W = list(W2); changed = True
        if not changed: break
    return False


def lb_rer(I, lo, hi, budget=60.0):
    """Smallest non-refuted T in [lo, hi] by bisection (refutation is monotone in T up to shaving heuristics; we
    only ever return a T with all smaller tested T refuted from lo upward via galloping, so validity holds)."""
    t0 = time.process_time(); lo, hi = int(math.ceil(lo - 1e-9)), int(hi); n = 0
    # validity: each refuted T proves Cmax > T; we keep lo = (largest proven refuted) + 1 among a monotone chain
    a = lo - 1; b = hi
    while b - a > 1 and time.process_time() - t0 < budget:
        c = (a + b) // 2; n += 1
        if refute_shave(I, c, deadline=t0 + budget): a = c
        else: b = c
    # refuted(c) proves Cmax > c directly, so LB = a + 1 is valid regardless of monotonicity
    return a + 1, n, time.process_time() - t0


def stage_chunked(r, d, p, m, chunk_bytes=64e6):
    """Same computation as er._stage, with the (t1, t2, job) arrays processed in t1-chunks (memory-bounded for n1 = 300)."""
    if np.any(r + p > d): return False, r, d
    if m >= len(p): return True, r, d
    T1 = np.unique(np.concatenate([r, d - p, r + p])); T2 = np.unique(np.concatenate([d, r + p, d - p]))
    step = max(1, int(chunk_bytes / (8 * 6 * max(1, T2.size) * len(p))))
    rn, dn = r.copy(), d.copy()
    for s in range(0, T1.size, step):
        t1 = T1[s:s + step]
        W, Lx, left, right = er._work(r, d, p, t1, T2)
        L2 = Lx[..., 0]; valid = L2 > 0; tot = W.sum(axis=2)
        if np.any((tot > m * L2) & valid): return False, r, d
        SL = (m * L2)[..., None] - tot[..., None] + W; vm = valid[..., None]
        wl = np.minimum(np.minimum(Lx, p[None, None, :]), left); need = vm & (wl > SL)
        if need.any(): rn = np.maximum(rn, np.ceil(np.where(need, T2[None, :, None] - SL, -np.inf).max(axis=(0, 1)) - 1e-9))
        wr = np.minimum(np.minimum(Lx, p[None, None, :]), right); need = vm & (wr > SL)
        if need.any(): dn = np.minimum(dn, np.floor(np.where(need, t1[:, None, None] + SL, np.inf).min(axis=(0, 1)) + 1e-9))
    return not np.any(rn + p > dn), rn, dn
