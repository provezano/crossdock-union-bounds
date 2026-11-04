"""EACDCml-ours (2026-11-02): re-implementation of EACDCml / EACDC (Nogueira et al. 2024, Evol. Intell.
doi:10.1007/s12065-024-00992-x) from the PAPER text (preprint + VoR). Where the paper is silent, the public code
(github.com/yotitinogs/EA-ML) or a logged choice is used; every choice is listed in SPEC (bottom) and in
the paper supplement (choices D1-D9).

Algorithm (paper Alg. 1, Sects. 4.2-4.5):
  individual = permutation of the n1 inbound trucks (priority order);
  decoder    = list scheduling of inbound trucks on the earliest-free inbound dock, then outbound ERD + LPT
               non-idle dispatch (paper Sect. 4.3) == g2.Dec (verified identical to the authors' evaluator);
  init       = PCH order (Delta_j desc, LPT ties), its improving swap neighbours, random fill up to mu;
  variation  = lambda offspring; each from uniformly chosen parent(s); crossover w.p. CXPB (PMX | OX), else
               mutation (swap | inversion | displacement | split&swap); operator within class chosen by the
               adaptive success-proportional rule p_i = (1+NS_i)/sum_j(1+NS_j);
  survival   = mu tournaments of size ts on P u L ((mu+lambda) scheme);
  stop       = NGEN generations or 0.25*NGEN consecutive generations without improving the best of P.
"""
import sys, os, time, random
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../decoder")
import numpy as np
import g2, cdc  # noqa: E402  (g2.Dec = our fast list decoder; cdc.parse / check_schedule / lb_pch)

# Variant selector for all drivers: EACDCML_VARIANT="<init>,<early_stop>" (default paper-literal "paper,1").
_V = os.environ.get("EACDCML_VARIANT", "paper,1").split(",")
DEFAULT_INIT, DEFAULT_ES = _V[0], _V[1] == "1"
STATIC = dict(NGEN=150, CXPB=0.71, ts=3, mu=75, lam=100)            # paper Table 2 (EACDC static tuning)
RANGES = dict(NGEN=(50, 200), CXPB=(0.0, 1.0), mu=(50, 200), lam=(50, 200), ts=(1, 20))  # paper Sect. 4.7


# ----------------------------------------------------------------------------------------------- PCH seed
def pch_delta(I):
    """Delta_i = max(sum_{j in S(i)} p2_j / m2, max_{j in S(i)} p2_j) (PCH 2020, step 2)."""
    S = I["prec"].T; p2 = I["p2"]
    return np.maximum(S @ p2 / I["m2"], np.where(S, p2[None, :], 0).max(axis=1))


def pch_order(I):
    """PCH priority order: Delta descending, LPT (larger p1) tie-break, then index (stable sort), as in PCH 2020
    Alg. 2 and the public code. (The EACDCml text says 'ascending'; see SPEC D1.)"""
    d = pch_delta(I); p1 = I["p1"]
    return sorted(range(I["n1"]), key=lambda i: (-float(d[i]), -int(p1[i])))


# ----------------------------------------------------------------------------------------------- operators
def pmx(p1, p2, a, b):
    """Partially mapped crossover, first child O1 (segment [a, b] inclusive from p1)."""
    n = len(p1); o = [None] * n; o[a:b + 1] = p1[a:b + 1]
    seg = set(p1[a:b + 1]); pos2 = {v: k for k, v in enumerate(p2)}
    for i in range(a, b + 1):
        j = p2[i]
        if j in seg: continue
        k = pos2[p1[i]]
        while a <= k <= b: k = pos2[p1[k]]
        o[k] = j
    for i in range(n):
        if o[i] is None: o[i] = p2[i]
    return o


def ox(p1, p2, a, b):
    """Order crossover, first child O1: segment [a, b] from p1, rest in p2's order starting after b (wrapping)."""
    n = len(p1); o = [None] * n; o[a:b + 1] = p1[a:b + 1]; seg = set(p1[a:b + 1])
    fill = [p2[(b + 1 + k) % n] for k in range(n)]; fill = [v for v in fill if v not in seg]
    for k, v in enumerate(fill): o[(b + 1 + k) % n] = v
    return o


def mut_swap(v, i, j):
    o = list(v); o[i], o[j] = o[j], o[i]; return o


def mut_inv(v, i, j):
    i, j = min(i, j), max(i, j); o = list(v); o[i:j + 1] = o[i:j + 1][::-1]; return o


def mut_disp(v, s, L, p):
    """Remove the sub-permutation v[s:s+L] and re-insert it starting at position p of the remainder."""
    seg = v[s:s + L]; rest = v[:s] + v[s + L:]; return rest[:p] + seg + rest[p:]


def mut_split(v, k):
    """Split & Swap: break at k (1..n-1) and swap the two parts."""
    return list(v[k:]) + list(v[:k])


def _two_points(rng, n):
    a, b = rng.randrange(n), rng.randrange(n)
    return (a, b) if a <= b else (b, a)


def apply_x(op, rng, p1, p2):
    a, b = _two_points(rng, len(p1))
    return pmx(p1, p2, a, b) if op == 0 else ox(p1, p2, a, b)


def apply_m(op, rng, v):
    n = len(v)
    if op == 0:
        i, j = rng.sample(range(n), 2); return mut_swap(v, i, j)
    if op == 1:
        i, j = rng.sample(range(n), 2); return mut_inv(v, i, j)
    if op == 2:
        L = rng.randint(1, max(1, n // 2)); s = rng.randrange(n - L + 1); p = rng.randrange(n - L + 1)
        return mut_disp(v, s, L, p)
    return mut_split(v, rng.randint(1, n - 1))


class AOS:
    """Adaptive operator selection (paper Sect. 4.5): p_i = C_i / sum C_j, C_i = 1 + NS_i (cumulative successes)."""
    def __init__(self, k): self.C = [1] * k
    def probs(self): s = sum(self.C); return [c / s for c in self.C]
    def pick(self, rng):
        r = rng.random() * sum(self.C); acc = 0.0
        for i, c in enumerate(self.C):
            acc += c
            if r < acc: return i
        return len(self.C) - 1
    def success(self, i): self.C[i] += 1


# ----------------------------------------------------------------------------------------------- the EA
class Counter:
    """Decoder wrapper counting evaluations."""
    def __init__(self, dec): self.dec = dec; self.n = 0
    def __call__(self, order): self.n += 1; return self.dec(order)


def init_population(I, mu, rng, ev):
    """Paper Sect. 4.4: PCH individual; swap neighbours of it (first level, i<j lexicographic) that are strictly
    better than PCH are inserted until |P| = mu or all n1(n1-1)/2 swaps are evaluated; random fill."""
    s0 = pch_order(I); f0 = ev(s0); P = [(s0, f0)]; n = I["n1"]; n_ls = 0
    for i in range(n - 1):
        if len(P) >= mu: break
        for j in range(i + 1, n):
            if len(P) >= mu: break
            s = list(s0); s[i], s[j] = s[j], s[i]; f = ev(s); n_ls += 1
            if f < f0: P.append((s, f))
    n_from_ls = len(P) - 1
    while len(P) < mu:
        s = list(range(n)); rng.shuffle(s); P.append((s, ev(s)))
    return P, dict(f_pch=f0, n_ls_evals=n_ls, n_ls_indiv=n_from_ls, n_random=mu - 1 - n_from_ls)


def init_population_code(I, mu, rng, ev):
    """Public-code initial population (GA.py load_individuals, Improved == 0), on the inbound permutation:
    first-improvement swap local search from the PCH order that RESTARTS (i1 = 0, i2 = 1) after every improvement
    and keeps the improving swap; i2 runs from i1 (the no-op swap included, as in the code); EVERY evaluated
    neighbour is added to the population (uncapped); survivor selection reduces it to mu after generation 1."""
    s1 = pch_order(I); n = I["n1"]; P = []; fbest = float("inf"); f0 = None
    i1 = 0
    while i1 < n - 1:
        i2 = i1
        while i2 <= n - 1:
            s1[i1], s1[i2] = s1[i2], s1[i1]; f = ev(s1)
            if f0 is None: f0 = f
            P.append((list(s1), f))
            if f < fbest:
                fbest = f; i1 = 0; i2 = 0
            else:
                s1[i1], s1[i2] = s1[i2], s1[i1]
            i2 += 1
        i1 += 1
    return P, dict(f_pch=f0, n_ls_evals=len(P), n_ls_indiv=len(P), n_random=0, ls_best=fbest)


def run(I, params, seed=169, dec=None, t0=None, cpu_cap=None, init=None, early_stop=None):
    """One EACDC(ml) run. params: NGEN, CXPB, mu, lam, ts. Returns dict with best (Alg. 1 output = best individual
    of P), order, cpu, evals, gens, trace [(cpu, evals, best)], best_ever (diagnostic: best of every evaluated)."""
    t0 = time.process_time() if t0 is None else t0
    init = DEFAULT_INIT if init is None else init; early_stop = DEFAULT_ES if early_stop is None else early_stop
    NGEN, CXPB, mu, lam, ts = int(params["NGEN"]), float(params["CXPB"]), int(params["mu"]), int(params["lam"]), int(params["ts"])
    rng = random.Random(seed)
    ev = Counter(dec or g2.Dec(I))
    P, info = (init_population if init == "paper" else init_population_code)(I, mu, rng, ev)
    bi = min(range(len(P)), key=lambda k: P[k][1]); best_o, best = list(P[bi][0]), P[bi][1]
    best_ever = best; trace = [(time.process_time() - t0, ev.n, best)]
    aosX, aosM = AOS(2), AOS(4); no_evo = 0; g = 0; stop = "NGEN"
    for g in range(1, NGEN + 1):
        L = []
        for _ in range(lam):
            if rng.random() < CXPB:
                i1, i2 = rng.sample(range(len(P)), 2); op = aosX.pick(rng)
                child = apply_x(op, rng, P[i1][0], P[i2][0]); f = ev(child)
                if f < min(P[i1][1], P[i2][1]): aosX.success(op)
            else:
                i1 = rng.randrange(len(P)); op = aosM.pick(rng)
                child = apply_m(op, rng, P[i1][0]); f = ev(child)
                if f < P[i1][1]: aosM.success(op)
            L.append((child, f))
            if f < best_ever: best_ever = f
        U = P + L
        P = [min(rng.sample(U, ts), key=lambda x: x[1]) for _ in range(mu)]
        cb = min(P, key=lambda x: x[1])
        if cb[1] < best:
            best, best_o = cb[1], list(cb[0]); no_evo = 0; trace.append((time.process_time() - t0, ev.n, best))
        else:
            no_evo += 1
            if early_stop and no_evo >= 0.25 * NGEN: stop = "noimp"; break
        if cpu_cap is not None and time.process_time() - t0 >= cpu_cap: stop = "cap"; break
    return dict(best=float(best), order=best_o, cpu=time.process_time() - t0, evals=ev.n, gens=g, stop=stop,
                trace=trace, best_ever=float(best_ever), init=info, aosX=aosX.C, aosM=aosM.C, variant=dict(init=init, early_stop=early_stop),
                params=dict(NGEN=NGEN, CXPB=CXPB, mu=mu, lam=lam, ts=ts))


# ----------------------------------------------------------------------------------------------- features
FEATURES = ["n1", "n2", "p1", "p2", "m1", "m2", "Sn"]


def features(I):
    """Paper Sect. 4.7: n1, n2, mean p1, mean p2, m1, m2, Sn = average number of outbound successors per inbound."""
    return [float(I["n1"]), float(I["n2"]), float(np.mean(I["p1"])), float(np.mean(I["p2"])), float(I["m1"]),
            float(I["m2"]), float(I["prec"].sum() / I["n1"])]


def clip_params(raw):
    """Map a raw predicted vector [NGEN, CXPB, mu, lam, ts] to valid parameters: clip to the paper's Optuna
    ranges, round the integer ones (SPEC D9)."""
    out = {}
    for k, v in zip(["NGEN", "CXPB", "mu", "lam", "ts"], raw):
        lo, hi = RANGES[k]; v = min(max(float(v), lo), hi)
        out[k] = v if k == "CXPB" else int(round(v))
    return out


def gap_ub(ub, lb):
    return (ub - lb) / ub


def verify(I, order, best):
    """Rebuild the schedule and check it with the independent checker; returns checked makespan."""
    cm, s1, s2 = g2.schedule_from(I, "f", order)
    c = cdc.check_schedule(I, s1, s2); assert abs(c - best) < 1e-6, (c, best); return c
