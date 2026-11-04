"""Re-implementation of the multi-dock HgR heuristics H1/H2 of Fonseca, Nogueira & Ravetti (EJOR 2019, Alg. MD, sec. 5.3),
ported from the authors' public AMPL driver MainVFR.run (read-only copy:
code/eacdc_public/third/fonseca_drive/MainVFR.run). Line refs below are to that file.

The AMPL model file it loads (modeloIndexParallelArtigo.md) is NOT public; the LP solved in each round is reconstructed
from the paper's eqs. (37)-(49): the Lagrangian relaxation of the time-indexed model with the cross-docking constraints
(33) dualized, which decouples into one LP per stage (solved here with HiGHS, 1 thread). See HGR_FINDINGS.md for every
interpretation choice. Pure Python; one thread.

Public API:
  hgr(I_path or parsed, mode="code"|"budget", budget=None, h2_bug=True, lp_cap=360) -> dict(ub, ub1, ub2, s1, s2, ...)
"""
import time, heapq, math
import numpy as np
import highspy

# ----------------------------------------------------------------------------------------------- instance + bounds
def load(path):
    tok = [int(x) for x in open(path).read().split()]
    n, m, maq1, maq2 = tok[:4]
    p1 = tok[4:4 + n]; p2 = tok[4 + n:4 + n + m]
    S = [tok[4 + n + m + i * n:4 + n + m + (i + 1) * n] for i in range(m)]  # S[i][j]: outbound i, inbound j
    return dict(n=n, m=m, maq1=maq1, maq2=maq2, p1=p1, p2=p2, S=S,
                pred=[[j for j in range(n) if S[i][j] > 0] for i in range(m)],
                succ=[[i for i in range(m) if S[i][j] > 0] for j in range(n)])


def lb_cs(P):
    """LB block of MainVFR.run (lines 'LB'): ceil(max(LBs, LBr)) -- Chen & Song type."""
    n, m, maq1, maq2, p1, p2 = P["n"], P["m"], P["maq1"], P["maq2"], P["p1"], P["p2"]
    pe1 = [max(sum(p1[i] for i in P["pred"][j]) / maq1, max(p1[i] for i in P["pred"][j])) for j in range(m)]
    per2 = [max(sum(p2[i] for i in P["succ"][j]) / maq2, max(p2[i] for i in P["succ"][j])) if P["succ"][j] else 0.0
            for j in range(n)]
    LBs = (sum(sorted(pe1)[:maq2]) + sum(p2)) / maq2
    LBr = (sum(sorted(per2)[:maq1]) + sum(p1)) / maq1
    return math.ceil(max(LBs, LBr)), pe1


def horizon0(P):
    """Initial UB / horizon T (Alg. MD line 3; MainVFR 'let T := ...')."""
    p1, p2, maq1, maq2 = P["p1"], P["p2"], P["maq1"], P["maq2"]
    return (sum(p1) / maq1 - max(p1) / maq1 + max(p1) + sum(p2) / maq2 - max(p2) / maq2 + max(p2))

# ----------------------------------------------------------------------------------------------- decoder (H1/H2 build)
class Decoder:
    """Stage 1: list scheduling of an inbound list on the least-loaded dock (MainVFR 'Constroi com base em X').
    Stage 2: at the least-loaded outbound dock, if no outbound job is released take the one with the smallest release
    (ties: largest index, AMPL keeps the last), else the longest released job (ties: largest index). Returns the
    stage-2 makespan (= the UB the code records, max TAcumY). Counts evaluations in self.nev."""
    def __init__(self, P):
        self.P = P; self.p1 = P["p1"]; self.p2 = P["p2"]; self.m1 = P["maq1"]; self.m2 = P["maq2"]
        self.pred = P["pred"]; self.nev = 0
        self.lpt2 = sorted(range(P["m"]), key=lambda j: (-self.p2[j], -j))  # longest first, ties largest index

    def __call__(self, L, full=False):
        self.nev += 1
        p1 = self.p1; p2 = self.p2
        h = [0] * self.m1; C1 = [0] * len(p1)
        for i in L:
            t = heapq.heappop(h) + p1[i]; C1[i] = t; heapq.heappush(h, t)
        r = [max(C1[i] for i in pr) for pr in self.pred]
        h = [0] * self.m2; left = list(self.lpt2); C2 = [0] * len(p2); cmax = 0
        while left:
            t = heapq.heappop(h); pick = -1
            for k, j in enumerate(left):
                if r[j] <= t: pick = k; break
            if pick < 0:  # nothing released: smallest release, ties -> largest index
                best = None
                for k, j in enumerate(left):
                    if best is None or r[j] < r[left[best]] or (r[j] == r[left[best]] and j > left[best]): best = k
                pick = best; t = r[left[pick]]
            j = left.pop(pick); c = t + p2[j]; C2[j] = c; heapq.heappush(h, c)
            if c > cmax: cmax = c
        return (cmax, C1, C2) if full else cmax


def h2_list(P, ordY, suc):
    """H2 stage-1 list: for each outbound in ordY, its not-yet-listed predecessors by decreasing Suc (number of
    successors), ties largest index (MainVFR 'Constroi com base em X com base em By'). Inbound jobs without successors
    are never listed by the code; they are appended at the end (they cannot delay any outbound job)."""
    seen = [False] * P["n"]; L = []
    for l in ordY:
        for k in sorted((k for k in P["pred"][l] if not seen[k]), key=lambda k: (-suc[k], -k)):
            seen[k] = True; L.append(k)
    L += [k for k in range(P["n"]) if not seen[k]]
    return L

# ----------------------------------------------------------------------------------------------- LP (Lagrangian subproblems)
LP_OPTS = {}
def _stage_lp(p, a, Tint, m, w, with_cmax, time_limit, stats):
    """LP relaxation of one stage of L_MD(lambda) (eqs. 38-41 or 43-49), in the cumulative form z_jt = sum_{s<=t} x_js
    (a bijective linear change of variables of the time-indexed x; same feasible set and optimum, ~4nT nonzeros
    instead of ~p n T). Objective: sum_j w_j * start_j (+ Cmax if with_cmax). Returns per-job start rounded by
    'last t with x_jt maximal' (MainVFR 'Viabiliza'), or None if no solution."""
    n = len(p); b = [Tint - p[j] for j in range(n)]
    a = [min(a[j], b[j]) for j in range(n)]
    off = [0] * (n + 1)
    for j in range(n): off[j + 1] = off[j] + (b[j] - a[j] + 1)
    nz = off[n] + (1 if with_cmax else 0); ic = off[n]
    cost = np.zeros(nz); lo = np.zeros(nz); hi = np.ones(nz)
    for j in range(n):
        # start_j = b_j - sum_{t=a_j}^{b_j-1} z_jt
        cost[off[j]:off[j + 1] - 1] = -w[j]
        lo[off[j + 1] - 1] = 1.0  # z_{j,b_j} = 1 (assignment)
    if with_cmax: cost[ic] = 1.0; lo[ic] = 0.0; hi[ic] = highspy.kHighsInf
    tb = time.process_time()
    R, Cc, V = [], [], []; nr = 0; rlo = []; rhi = []
    for j in range(n):  # monotone rows: z_{j,t} - z_{j,t-1} >= 0, t = a+1..b
        k = b[j] - a[j]
        if k > 0:
            rr = np.arange(nr, nr + k); cc = off[j] + np.arange(1, k + 1)
            R += [rr, rr]; Cc += [cc, cc - 1]; V += [np.ones(k), -np.ones(k)]; nr += k
    rlo.append(np.zeros(nr)); rhi.append(np.full(nr, np.inf))
    # capacity at period t (t = 0..Tint): sum_j (z_{j,min(t,b_j)} - z_{j,t-p_j} [if t-p_j >= a_j]) <= m
    base = nr
    for j in range(n):
        t = np.arange(a[j], min(b[j] + p[j], Tint + 1))
        R.append(base + t); Cc.append(off[j] + np.minimum(t, b[j]) - a[j]); V.append(np.ones(len(t)))
        t2 = t[t - p[j] >= a[j]]
        R.append(base + t2); Cc.append(off[j] + t2 - p[j] - a[j]); V.append(-np.ones(len(t2)))
    nr += Tint + 1; rlo.append(np.full(Tint + 1, -np.inf)); rhi.append(np.full(Tint + 1, float(m)))
    if with_cmax:  # Cmax >= p_j + start_j  <=>  Cmax + sum_{t<b} z_jt >= p_j + b_j
        for j in range(n):
            k = off[j + 1] - 1 - off[j]
            R.append(np.full(k + 1, nr + j)); Cc.append(np.concatenate([[ic], np.arange(off[j], off[j + 1] - 1)]))
            V.append(np.ones(k + 1))
        rlo.append(np.array([p[j] + b[j] for j in range(n)], float)); rhi.append(np.full(n, np.inf)); nr += n
    import scipy.sparse as sp
    A = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(nr, nz))
    A.eliminate_zeros(); rows_lo = np.concatenate(rlo); rows_hi = np.concatenate(rhi)
    rows_lo[np.isinf(rows_lo)] = -highspy.kHighsInf; rows_hi[np.isinf(rows_hi)] = highspy.kHighsInf
    starts, idx, val = A.indptr, A.indices, A.data
    time_limit = time_limit - (time.process_time() - tb)
    if time_limit <= 0.01:
        stats.append(dict(cols=nz, rows=nr, nnz=int(A.nnz), cpu=time.process_time() - tb, status="build>limit")); return None
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", max(0.01, float(time_limit)))
    for k, v in LP_OPTS.items(): h.setOptionValue(k, v)
    lp = highspy.HighsLp(); lp.num_col_ = nz; lp.num_row_ = len(rows_lo)
    lp.col_cost_ = cost; lp.col_lower_ = lo; lp.col_upper_ = hi
    lp.row_lower_ = rows_lo; lp.row_upper_ = rows_hi
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.array(starts, np.int32); lp.a_matrix_.index_ = np.array(idx, np.int32)
    lp.a_matrix_.value_ = np.array(val, float)
    h.passModel(lp)
    h.run(); dt = time.process_time() - tb
    st = h.getModelStatus(); sol = h.getSolution()
    stats.append(dict(cols=nz, rows=nr, nnz=int(len(idx)), cpu=dt, status=h.modelStatusToString(st)))
    if not sol.value_valid: return None
    z = np.asarray(sol.col_value)
    out = []
    for j in range(n):
        zz = z[off[j]:off[j + 1]]; x = np.diff(np.concatenate([[0.0], zz]))
        mx = x.max(); t = int(np.flatnonzero(x >= mx - 1e-9)[-1])  # last t with maximal x
        out.append(a[j] + t)
    return out

# ----------------------------------------------------------------------------------------------- main loop (Alg. MD)
def hgr(P, mode="code", budget=None, h2_bug=True, lp_cap=360.0, lp_frac=0.5, n_rounds=5, round_stop=50.0, log=None):
    """mode='code': MainVFR semantics (<= 5 rounds; stop when a round brings no improvement or when one round uses
    >= 50 s; LP time limit 360 s). mode='budget': same algorithm under a total CPU budget (process time): the LP of a
    round gets at most lp_frac of the remaining budget, H1/H2 stop when the budget is spent."""
    if isinstance(P, str): P = load(P)
    t0 = time.process_time(); left = (lambda: budget - (time.process_time() - t0)) if budget else (lambda: 1e18)
    n, m = P["n"], P["m"]; p1, p2 = P["p1"], P["p2"]
    LB, pe1 = lb_cs(P); T = horizon0(P)
    corte1 = [math.ceil(v) for v in pe1]  # earliest start of outbound i (head); model file not public -> our choice
    suc = [len(P["succ"][j]) for j in range(n)]
    D = Decoder(P)
    UB1 = UB2 = T; best1 = best2 = None; Bx = [0.0] * n; By = [0.0] * m
    lpstats = []; rounds = []; stop_reason = "rounds"
    for ii in range(n_rounds):
        tr = time.process_time(); melhora = False
        T = min(UB1, UB2); Tint = int(math.floor(T))
        # ---- solve the LP relaxation of both subproblems with the current multipliers (w = -Bx >= 0, v = -By >= 0)
        lshare = lp_frac * left(); tl0 = time.process_time()  # budget mode: both LPs share lp_frac of what is left
        cap = min(lp_cap, max(0.0, left())) if mode == "code" else max(0.0, lshare / 2)
        wx = [-b for b in Bx]; vy = [-b for b in By]
        sc = max([1.0] + vy)  # stage-2 multipliers normalised to max 1 (cf. the normQ normalisation commented out in MainVFR); numerical stability
        X = _stage_lp(p1, [0] * n, Tint, P["maq1"], wx, False, cap, lpstats) if cap > 0.01 else None
        cap2 = min(lp_cap, max(0.0, left())) if mode == "code" else max(0.0, lshare - (time.process_time() - tl0))
        Y = _stage_lp(p2, corte1, Tint, P["maq2"], [v / sc for v in vy], True, cap2, lpstats) if cap2 > 0.01 else None
        # ---- multipliers (MainVFR 'Calcula Multiplicado Lagrange'), normQ = 1
        if X is not None and Y is not None:
            mult = {(i, j): min(Y[i] - p1[j] - X[j], 0) for i in range(m) for j in P["pred"][i]}
            Bx = [(2 * T - LB) * sum(mult[(i, j)] for i in P["succ"][j]) / len(P["succ"][j]) if P["succ"][j] else 0.0
                  for j in range(n)]
            By = [(2 * T - LB) * sum(mult[(i, j)] for j in P["pred"][i]) / len(P["pred"][i]) for i in range(m)]
        lp_ok = X is not None and Y is not None
        # ---- orders: decreasing weight, ties -> largest index (AMPL keeps the last argmax)
        ordX = sorted(range(n), key=lambda j: (-Bx[j], -j)); ordY = sorted(range(m), key=lambda j: (-By[j], -j))
        # ---- H1: first-improvement pairwise swap with restart (MainVFR 'H1')
        g = 0; out = False
        while g <= n - 2 and not out:
            h = g
            while h <= n - 1:
                if left() <= 0: out = True; break
                ordX[g], ordX[h] = ordX[h], ordX[g]
                v = D(ordX)
                if v < UB1:
                    UB1 = v; best1 = list(ordX); melhora = True; g = 0; h = -1
                else:
                    ordX[g], ordX[h] = ordX[h], ordX[g]
                h += 1
            g += 1
        # ---- H2: same on the outbound order, stage-1 list built from it (MainVFR 'H2')
        g = 0; out2 = False
        while g <= m - 2 and not out2:
            h = g
            while h <= m - 1:
                if left() <= 0: out2 = True; break
                ordY[g], ordY[h] = ordY[h], ordY[g]
                L = h2_list(P, ordY, suc)
                if h2_bug:  # the code swaps back before evaluating, then re-applies the swap only if NOT improving
                    ordY[g], ordY[h] = ordY[h], ordY[g]
                v = D(L)
                if v < UB2:
                    UB2 = v; best2 = L; melhora = True; g = 0; h = -1
                    # h2_bug: order stays reverted (improvement not kept in ordY); fixed: keep the swap
                else:
                    ordY[g], ordY[h] = ordY[h], ordY[g]  # bug: re-applies; fixed: reverts
                h += 1
            g += 1
        rounds.append(dict(round=ii + 1, cpu=time.process_time() - tr, UB1=UB1, UB2=UB2, lp_ok=lp_ok, T=T))
        if log: log(rounds[-1])
        if out or out2: stop_reason = "budget"; break
        if mode == "code" and time.process_time() - tr >= round_stop: stop_reason = "round>=50s"; break
        if not melhora: stop_reason = "no_improvement"; break
    # ---- final schedule (best of H1/H2), start times
    if best1 is None and best2 is None:  # budget spent before any evaluation: decode the current H1 order once
        best1 = list(ordX); UB1 = D(best1)
    cands = [(UB1, best1), (UB2, best2)]
    ub, L = min([c for c in cands if c[1] is not None], key=lambda c: c[0])
    cm, C1, C2 = D(L, full=True)
    s1 = [C1[i] - p1[i] for i in range(n)]; s2 = [C2[j] - p2[j] for j in range(m)]
    return dict(ub=max(max(C1), cm), ub_stage2=cm, ub1=UB1, ub2=UB2, LB=LB, s1=s1, s2=s2, nev=D.nev,
                cpu=time.process_time() - t0, rounds=rounds, lp=lpstats, stop=stop_reason)
