"""anytime profile experiment shared helpers. Frozen search library = fbils_frozen (byte-identical ILS v2 / IG, SHA256SUMS there);
decoder/constructives = decoder/g2.py (frozen hash 480e79cc...)."""
import sys, os, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../decoder"); sys.path.insert(0, f"{D}/../fbils_frozen")
import g2, cdc, srch  # noqa: E402

BIGCAP = 7200.0  # CPU cap (s) for eval-budgeted searches; never expected to bind (flagged if it does)


def start(I):
    """Best of the 14 step 2 constructives (as every earlier ILS/IG arm). Returns J (searched side), side, order, ncons, tcon."""
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I)
    return J, side, order, len(C), time.process_time() - t0


def at(tr, lim):
    """Best value of an anytime trace [(cpu, evals, best)] after at most `lim` evaluations (incl. constructives)."""
    v = [x[2] for x in tr if x[1] <= lim]
    return min(v) if v else None


def cpu_at(tr, lim):
    v = [x[0] for x in tr if x[1] <= lim]
    return max(v) if v else None


def check(I, J, side, bo, b):
    assert g2.Dec(J)(bo) == b
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6


def run_fbils(I, ev_b, seed):
    """FB-ILS = frozen ILS v2 configuration (pfbi 0, stall 40, rounds 3, wfrac 0.25, seed 100+s, nev0 = 14),
    run to an evaluation budget only (cpu_budget 0), anytime trace returned."""
    J, side, order, n0, tcon = start(I)
    t0 = time.process_time()
    b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, 0.0, int(ev_b), cap=BIGCAP, seed=100 + seed, nev0=n0,
                                 pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    cpu = time.process_time() - t0; check(I, J, side, bo, b)
    return dict(best=float(b), trace=[(c + tcon, e, v) for c, e, v in tr], nev=int(ne), tcon=tcon, cpu=cpu + tcon,
                capped=bool(ne < ev_b))


def run_ig(I, ev_b, seed, d, T, ls):
    """Ruiz & Stuetzle (2007) iterated greedy as frozen in srch.c (algo 1): destroy d random jobs, greedy best-position
    reinsertion, optional insertion local search to a local optimum (ls = 1), SA-like acceptance with temperature
    T * (sum p) / ((n1 + n2) * 10). Start = best constructive; seed 100 + s."""
    J, side, order, n0, tcon = start(I)
    t0 = time.process_time()
    b, bo, tr, ne = srch.run(J, order, "ig", 0.0, int(ev_b), cap=BIGCAP, seed=100 + seed, nev0=n0, ip1=int(d), fp1=float(T),
                             fp2=float(ls))
    cpu = time.process_time() - t0; check(I, J, side, bo, b)
    return dict(best=float(b), trace=[(c + tcon, e, v) for c, e, v in tr], nev=int(ne), tcon=tcon, cpu=cpu + tcon,
                capped=bool(ne < ev_b))
