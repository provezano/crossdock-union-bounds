"""Matched-comparison study (2026-11-02): time-matched / tuned UB comparison for P|M_j,prec|Cmax (Cota+2016).
- ils_p: parametrised copy of g2.ils (defaults reproduce g2.ils exactly: stall_mult=40, n_pert=2, seg_rev=True)
- lbprefix_order: LB-guided constructive from the union-knapsack sets of the LB walk (task 5c)
- eacdc: authors' GA.py (unmodified file; verified bit-identical fast evaluator/loader) with an anytime
  wrapper on the evaluator: records (cpu, best) and stops at a CPU budget (truncation = best-so-far);
  restarts with fresh seeds while budget remains (EACDC-R). Also returns the best individual.
- memetic: EACDC for a fraction of the budget, then ILS intensification from the best of
  {EACDC best (derived inbound order), LB-prefix order, best step 2 constructive}."""
import sys, os, time, random, io, contextlib
D = os.path.dirname(os.path.abspath(__file__))
for p in ("benchmark_data", "lower_bounds", "decoder", "eacdc_public/code"):
    sys.path.insert(0, f"{D}/../{p}")
from ortools.sat.python import cp_model  # noqa
import numpy as np
import cdc, g1, g2

F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
def path(g, n1, seed): return f"{D}/../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{seed}.dat"


def ils_p(I, side, order, budget, seed=0, stall_mult=40, n_pert=2, seg_rev=True, t_offset=0.0):
    J = I if side == "f" else g2.reverse(I)
    Dd = g2.Dec(J); rng = np.random.default_rng(seed)
    cur = list(order); fc = Dd(cur); best, bo = fc, list(cur); n = len(cur)
    t0 = time.process_time(); trace = [(t_offset, best)]; stall = 0
    while time.process_time() - t0 < budget:
        a, b = rng.integers(n, size=2)
        if a == b: continue
        nb = list(cur); x = nb.pop(a); nb.insert(b, x)
        f = Dd(nb)
        if f <= fc:
            stall = 0 if f < fc else stall + 1; cur, fc = nb, f
            if f < best: best, bo = f, list(nb); trace.append((t_offset + time.process_time() - t0, best))
        else: stall += 1
        if stall > stall_mult * n:
            cur = list(bo)
            if seg_rev:
                a, b = sorted(rng.integers(n, size=2)); cur[a:b + 1] = cur[a:b + 1][::-1]
            for _ in range(n_pert):
                a, b = rng.integers(n, size=2); x = cur.pop(a); cur.insert(b, x)
            fc = Dd(cur); stall = 0
    return best, bo, trace


def lbprefix_order(I, budget=2.0, tl=0.25, n_caps=8):
    """Stage-1 order built from union-knapsack optimal sets A_c at increasing capacities c, including the
    breakpoint c* where the LB walk attains its max. Inbound jobs of N(A_c) \\ U are appended (LPT),
    the rest in PCH-tail order. Cost is CPU-capped by `budget` (LB walk) + n_caps knapsack solves."""
    P = I["prec"]; p1 = I["p1"]; m1, m2 = I["m1"], I["m2"]; P2 = float(I["p2"].sum())
    r = g1.lb_union_exact(I, 0.0, tl=tl, budget=budget, return_trace=True)
    rmin = float(cdc.heads(I).min()); best_c, best_term = None, -1
    for (c, cost, vub, vinc, opt) in r["trace"]:
        th = (c + 1) / m1
        if vub < P2:
            term = th + (P2 - min(vub, m2 * max(th - rmin, 0))) / m2
            if term > best_term: best_term, best_c = term, c
    full = int(p1[P.any(axis=0)].sum())
    caps = set(np.linspace(0, full, n_caps + 2)[1:-1].astype(int).tolist())
    if best_c is not None: caps.add(int(best_c))
    uk = g1.UnionKnap(I); U = np.zeros(I["n1"], bool); order = []
    for c in sorted(caps):
        _, _, _, _, xs = uk.solve(c, tl)
        if xs is None: continue
        add = np.flatnonzero(P[xs].any(axis=0) & ~U)
        add = add[np.argsort(-p1[add], kind="stable")]; order += add.tolist(); U[add] = True
    rest = [i for i in g2.orders_basic(I)["pch"] if not U[i]]
    return order + rest, dict(c_star=best_c, lb_raw=r["raw"])


def best_constructive(I, with_lbp=False, lbp_budget=2.0):
    C = g2.all_constructive(I)
    if with_lbp:
        for side, J in (("f", I), ("r", g2.reverse(I))):
            o, _ = lbprefix_order(J, budget=lbp_budget); C[f"lbp_{side}"] = (g2.Dec(J)(o), side, o)
    k = min(C, key=lambda k: C[k][0]); return k, C


# ---------------- EACDC (authors' GA.py) with anytime wrapper ----------------
class _Stop(Exception): pass
_GA = None
_REAL_SEED = random.seed  # captured once (fixed 2026-11-02: re-capturing chained lambdas -> RecursionError after ~1000 calls)
def _load_ga():
    global _GA
    if _GA is None:
        import importlib.util, warnings; warnings.filterwarnings("ignore")
        S = f"{D}/../eacdc_public/code"
        spec = importlib.util.spec_from_file_location("run_eaml", f"{S}/run_eaml.py"); RE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RE)
        import fast_eval, fast_load
        RE.GA.load_individuals = fast_load.make(RE.GA)
        _GA = (RE, RE.GA, fast_eval.make(RE.GA))
    return _GA

DEFAULT = (3, 150, 75, 100, 0.95, 0.4)  # GA.py / paper Table 2 static configuration

def eacdc(g, n1, iseed, seed, budget=None, params=DEFAULT, restart=True, fpath=None):
    """Returns dict(best, trace[(cpu,best)], best_perm (outbound), runs, cpu). budget None = one full run."""
    RE, GA, fe = _load_ga()
    I = RE.leitura(fpath or path(g, n1, iseed))
    st = dict(best=float("inf"), perm=None, trace=[], trace_e=[], t0=time.process_time(), nev=0, ninit=0)
    def ev(ind):
        f = fe(ind); c = time.process_time() - st["t0"]; st["nev"] += 1
        if f[0] < st["best"]:
            st["best"] = f[0]; st["perm"] = list(ind); st["trace"].append((c, float(f[0])))
            st["trace_e"].append((st["nev"], st["ninit"], float(f[0])))
        if budget is not None and c > budget: raise _Stop()
        return f
    GA.evalPop = ev
    if not hasattr(GA, "_fast_load_raw"): GA._fast_load_raw = GA.load_individuals
    def li(creator, n):  # counts the schedules built inside the deterministic PCH-swap initialisation
        out = GA._fast_load_raw(creator, n); st["ninit"] += len(out); return out
    GA.load_individuals = li
    _rs = random.Random.seed; runs = 0; s = seed
    orig = _REAL_SEED
    while True:
        runs += 1; ss = s
        GA.random.seed = lambda x=None, *a, **k: orig(ss if x == 169 else x)
        try:
            with contextlib.redirect_stdout(io.StringIO()): GA.GA(*I, *params)
        except _Stop:
            break
        if budget is None or not restart or time.process_time() - st["t0"] >= budget: break
        s += 1000
    return dict(best=st["best"], trace=st["trace"], perm=st["perm"], runs=runs, cpu=time.process_time() - st["t0"],
                n_evalpop=st["nev"], n_init=st["ninit"], trace_e=st["trace_e"])


def inbound_from_perm(I, perm):
    """Inbound order the GA decoder derives from an outbound permutation (same rule as fast_eval)."""
    P = I["prec"]; p1 = I["p1"]; n2 = I["n2"]
    pos = np.empty(n2, np.int64); pos[np.asarray(perm)] = np.arange(n2)
    first = np.where(P, pos[:, None], n2 + 1).min(axis=0)
    idx = np.flatnonzero(P.any(axis=0))
    return idx[np.lexsort((idx, -p1[idx], first[idx]))].tolist()


def memetic(g, n1, iseed, seed, budget, ga_frac=0.4):
    I = cdc.parse(path(g, n1, iseed)); t0 = time.process_time()
    e = eacdc(g, n1, iseed, seed, budget=ga_frac * budget, restart=False)
    cands = {"eacdc": ("f", inbound_from_perm(I, e["perm"]))}
    o, _ = lbprefix_order(I, budget=min(2.0, 0.1 * budget)); cands["lbp"] = ("f", o)
    C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); cands["constr"] = (C[kb][1], C[kb][2])
    vals = {k: (g2.Dec(I if s == "f" else g2.reverse(I))(v)) for k, (s, v) in cands.items()}
    kbest = min(vals, key=vals.get); side, order = cands[kbest]
    used = time.process_time() - t0
    b, bo, tr = ils_p(I, side, order, max(0.0, budget - used), seed=100 + seed, t_offset=used)
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6
    trace = e["trace"] + [(used, min(vals.values()))] + tr
    return dict(best=float(min(b, e["best"])), trace=trace, start_from=kbest, cpu=time.process_time() - t0, ga_best=e["best"])
