"""Upper-bound study (2026-11-01): upper bounds for P|M_j,prec|Cmax cross-docking (Cota+2016).
Heuristics: list decoding (inbound order -> LS; outbound ERD+LPT non-idle), reversal, union-greedy (LB-guided)
orders, iterated local search on the inbound permutation, and CP-SAT LNS (time windows + rigid shifted suffix)."""
import time, heapq
import numpy as np
from cdc import parse, cpsat, check_schedule

def reverse(I):
    return dict(n1=I["n2"], n2=I["n1"], m1=I["m2"], m2=I["m1"], p1=I["p2"], p2=I["p1"], prec=I["prec"].T.copy())

class Dec:
    """Fast pure-python list decoder (lists are faster than numpy for n<=100)."""
    def __init__(self, I):
        self.I = I; self.p1 = I["p1"].tolist(); self.p2 = I["p2"].tolist(); self.m1 = I["m1"]; self.m2 = I["m2"]
        self.preds = [np.flatnonzero(I["prec"][j]).tolist() for j in range(I["n2"])]
        # outbound sorted by LPT (ties by index) for ERD+LPT dispatch
        self.lpt2 = sorted(range(I["n2"]), key=lambda j: (-self.p2[j], j))
    def stage1(self, order):
        p1 = self.p1; h = [0] * self.m1; C1 = [0] * len(p1)
        for i in order:
            t = heapq.heappop(h) + p1[i]; C1[i] = t; heapq.heappush(h, t)
        return C1
    def __call__(self, order, full=False):
        C1 = self.stage1(order); p2 = self.p2
        r = [max((C1[i] for i in pr), default=0) for pr in self.preds]
        # non-idle ERD+LPT: at time t (earliest free dock) pick the longest released job; else wait for min release
        h = [0] * self.m2; left = list(self.lpt2); C2 = [0] * len(p2); cmax = max(C1)
        while left:
            t = heapq.heappop(h)
            pick = None
            for k, j in enumerate(left):
                if r[j] <= t: pick = k; break
            if pick is None:
                rm = min(r[j] for j in left); t = rm
                for k, j in enumerate(left):
                    if r[j] <= t: pick = k; break
            j = left.pop(pick); c = t + p2[j]; C2[j] = c; heapq.heappush(h, c)
            if c > cmax: cmax = c
        return (cmax, C1, C2) if full else cmax

def orders_basic(I):
    p1, p2, P = I["p1"], I["p2"], I["prec"]
    S = P.T
    d = np.maximum(S @ p2 / I["m2"], np.where(S, p2[None], 0).max(1))
    o = {}
    o["pch"] = sorted(range(I["n1"]), key=lambda i: (-d[i], -p1[i]))
    o["lpt"] = sorted(range(I["n1"]), key=lambda i: (-p1[i], i))
    o["succ"] = sorted(range(I["n1"]), key=lambda i: (-S[i].sum(), -p1[i]))
    return o

def union_greedy(I, alpha=1.0, inner="lpt", bundle=True):
    """LB-guided order: repeatedly add the predecessor set N(j)\\U (j outbound) maximizing
    (outbound work newly unlocked by U u N(j))/p1(N(j)\\U)^alpha -- the greedy for the set-union knapsack K(c)
    that defines the union LB. Inner order of the added inbound jobs: LPT (or most-successors)."""
    P = I["prec"]; p1 = I["p1"].astype(float); p2 = I["p2"].astype(float)
    n1, n2 = I["n1"], I["n2"]
    U = np.zeros(n1, bool); unlocked = ~P.any(1); order = []
    nsucc = P.sum(0)
    while not U.all():
        need = P & ~U[None, :]                      # n2 x n1 missing preds per outbound
        cost = need @ p1
        cand = (~unlocked) & (cost > 0)
        if not cand.any():
            rest = np.flatnonzero(~U); rest = rest[np.argsort(-p1[rest], kind="stable")]; order += rest.tolist(); break
        if bundle:
            newU = U[None, :] | P                   # candidate sets (n2 x n1)
            # outbound k unlocked by candidate j: P[k] subset newU[j]  <=> no pred of k outside newU[j]
            gain = ((~(P[None, :, :] & ~newU[:, None, :]).any(2)) & ~unlocked[None, :]) @ p2
        else:
            gain = p2.copy()
        score = np.where(cand, gain / np.maximum(cost, 1e-9) ** alpha, -np.inf)
        j = int(np.argmax(score))
        add = np.flatnonzero(need[j])
        key = (-p1[add]) if inner == "lpt" else (-nsucc[add] * 1e6 - p1[add])
        add = add[np.argsort(key, kind="stable")]
        order += add.tolist(); U[add] = True
        unlocked = ~(P & ~U[None, :]).any(1)
    return order

def all_constructive(I):
    """Returns dict name -> (cmax, side, order) for forward and reversed instance."""
    res = {}
    for side, J in (("f", I), ("r", reverse(I))):
        D = Dec(J); o = orders_basic(J)
        for a in (0.5, 1.0, 2.0):
            o[f"ug{a}"] = union_greedy(J, a)
        o["ug1s"] = union_greedy(J, 1.0, inner="succ")
        for k, v in o.items():
            res[f"{k}_{side}"] = (D(v), side, v)
    return res

def ils(I, side, order, budget, seed=0, D=None):
    """Iterated local search on the inbound permutation: random insertion moves (accept <=), perturbation
    by a random segment reversal when stuck; CPU budget in seconds. Returns (best, order, trace[(cpu, best)])."""
    J = I if side == "f" else reverse(I)
    D = D or Dec(J); rng = np.random.default_rng(seed)
    cur = list(order); fc = D(cur); best, bo = fc, list(cur); n = len(cur)
    t0 = time.process_time(); trace = [(0.0, best)]; stall = 0
    while time.process_time() - t0 < budget:
        a, b = rng.integers(n, size=2)
        if a == b: continue
        nb = list(cur); x = nb.pop(a); nb.insert(b, x)
        f = D(nb)
        if f <= fc:
            stall = 0 if f < fc else stall + 1; cur, fc = nb, f
            if f < best: best, bo = f, list(nb); trace.append((time.process_time() - t0, best))
        else: stall += 1
        if stall > 40 * n:
            cur = list(bo); a, b = sorted(rng.integers(n, size=2)); cur[a:b + 1] = cur[a:b + 1][::-1]
            for _ in range(2):
                a, b = rng.integers(n, size=2); x = cur.pop(a); cur.insert(b, x)
            fc = D(cur); stall = 0
    return best, bo, trace

def schedule_from(I, side, order):
    """Start times (s1, s2) in the ORIGINAL instance for a decoded order (reversed side mirrored in time)."""
    if side == "f":
        D = Dec(I); cm, C1, C2 = D(order, full=True)
        return cm, np.array(C1) - I["p1"], np.array(C2) - I["p2"]
    J = reverse(I); D = Dec(J); cm, C1, C2 = D(order, full=True)
    # reversed stage-1 = original outbound; reversed job ends at C -> original starts at cm - C
    s2 = cm - np.array(C1); s1 = cm - np.array(C2)
    return cm, s1, s2

def lns(I, s1, s2, budget, sub_tl=2.0, seed=0, win_frac=0.25):
    """CP-SAT LNS: choose a time window [a, a+w); jobs starting before a are fixed, jobs inside are free,
    jobs starting after a+w move rigidly by a common shift d (keeps their internal feasibility).
    Accepts improvements (cmax <= best-1 constraint). 1 worker. Returns (best, s1, s2, trace)."""
    from ortools.sat.python import cp_model
    p1, p2, P, m1, m2 = I["p1"], I["p2"], I["prec"], I["m1"], I["m2"]
    s1 = np.array(s1, int); s2 = np.array(s2, int)
    best = int(max((s1 + p1).max(), (s2 + p2).max())); rng = np.random.default_rng(seed)
    t0 = time.process_time(); trace = [(0.0, best)]; wf = win_frac; it = 0
    edges = [(j, i) for j in range(I["n2"]) for i in np.flatnonzero(P[j])]
    while time.process_time() - t0 < budget:
        it += 1
        w = max(1, int(wf * best)); a = int(rng.integers(0, max(1, best - w)))
        m = cp_model.CpModel(); H = best
        d = m.NewIntVar(-w, 0, "d")
        def mk(s, p, tag):
            vs = []
            for k in range(len(s)):
                if s[k] < a: v = m.NewConstant(int(s[k]))
                elif s[k] < a + w: v = m.NewIntVar(a if True else 0, H, f"{tag}{k}"); m.AddHint(v, int(s[k]))
                else:
                    v = m.NewIntVar(0, H, f"{tag}{k}"); m.Add(v == int(s[k]) + d)
                vs.append(v)
            return vs
        v1 = mk(s1, p1, "a"); v2 = mk(s2, p2, "b")
        iv1 = [m.NewFixedSizeIntervalVar(v1[i], int(p1[i]), "") for i in range(I["n1"])]
        iv2 = [m.NewFixedSizeIntervalVar(v2[j], int(p2[j]), "") for j in range(I["n2"])]
        m.AddCumulative(iv1, [1] * I["n1"], m1); m.AddCumulative(iv2, [1] * I["n2"], m2)
        for j, i in edges: m.Add(v2[j] >= v1[i] + int(p1[i]))
        cm = m.NewIntVar(0, H, "c")
        for j in range(I["n2"]): m.Add(cm >= v2[j] + int(p2[j]))
        for i in range(I["n1"]): m.Add(cm >= v1[i] + int(p1[i]))
        m.Add(cm <= best - 1); m.Minimize(cm); m.AddHint(d, 0)
        sv = cp_model.CpSolver(); sv.parameters.num_workers = 1; sv.parameters.random_seed = int(seed + it)
        sv.parameters.max_time_in_seconds = min(sub_tl, max(0.05, budget - (time.process_time() - t0)))
        st = sv.Solve(m)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            s1 = np.array([sv.Value(v) for v in v1]); s2 = np.array([sv.Value(v) for v in v2])
            best = int(sv.ObjectiveValue()); trace.append((time.process_time() - t0, best))
        else:
            wf = min(0.6, wf * 1.1)  # enlarge the neighbourhood when stuck
    return best, s1, s2, trace
