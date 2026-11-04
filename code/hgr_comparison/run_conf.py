"""Preregistered run: FB-ILS (ILS v2, frozen) and base ILS (ILS v1, g2.ils) vs HgR
(Fonseca, Nogueira & Ravetti 2019; our re-implementation hgr.py) at matched process-CPU budget.
All code is imported from ./frozen (byte copies; sha256 in the prereg). The runner logic for each arm is the one of
bound_analyses/hgr/match.py (run_hgr / run_ils1 / run_ils2), with the seed passed through.

One task = (instance, seed); every method of the task runs back to back in ONE process; the method order is rotated
by the task index q (methods[q % k:] + methods[:q % k]). Every final schedule is checked with cdc.check_schedule.
usage: run_conf.py LIST.json BUDGET OUTDIR METHODS SEEDS SHARD NSHARDS
  e.g. run_conf.py list210.json 10 out/b10 hgr,hgrnolp,ils1,ils2 41,42,43 0 4"""
import sys, os, json, time, platform, socket
D = os.path.dirname(os.path.abspath(__file__)); FZ = f"{D}/frozen"
sys.path.insert(0, FZ)
import hgr, cdc, g2, srch  # noqa: E402  (all from ./frozen)
hgr.LP_OPTS = dict(solver="ipm")  # as in match.py: HiGHS IPM + crossover, 1 thread


def run_hgr(path, B, variant):
    P = hgr.load(path)
    r = hgr.hgr(P, mode="budget", budget=B, h2_bug=True, lp_frac=(0.0 if variant == "nolp" else 0.5),
                lp_cap=(0.0 if variant == "nolp" else 360.0))
    lp_ok = [x["lp_ok"] for x in r["rounds"]]
    return r["ub"], r["s1"], r["s2"], dict(ub1=r["ub1"], ub2=r["ub2"], nev=r["nev"], stop=r["stop"],
                                           n_rounds=len(r["rounds"]), lp_ok=lp_ok,
                                           lp_status=[x["status"] for x in r["lp"]])


def run_ils1(I, B, seed):
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0])
    tcon = time.process_time() - t0
    b, bo, tr = g2.ils(I, C[kb][1], C[kb][2], max(0.0, B - tcon), seed=100 + seed)
    cm, s1, s2 = g2.schedule_from(I, C[kb][1], bo)
    return b, s1, s2, dict(tcon=tcon, constr=kb)


def run_ils2(I, B, seed):
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0])
    side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I); tcon = time.process_time() - t0
    b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, max(0.0, B - tcon), 0, cap=B + 1, seed=100 + seed,
                                 nev0=len(C), pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    assert g2.Dec(J)(bo) == b
    cm, s1, s2 = g2.schedule_from(I, side, bo)
    return b, s1, s2, dict(tcon=tcon, constr=kb, nev=ne)


def _nice():
    try: return os.getpriority(os.PRIO_PROCESS, 0)
    except OSError: return None


def main():
    lst, B, out, methods, seeds, sh, ns = (sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4].split(","),
                                          [int(x) for x in sys.argv[5].split(",")], int(sys.argv[6]), int(sys.argv[7]))
    os.makedirs(out, exist_ok=True)
    L = json.load(open(lst)); jobs = [(r, s) for s in seeds for r in L]
    env = dict(python=sys.version.split()[0], exe=sys.executable, host=socket.gethostname(), node=platform.node(),
               machine=platform.machine(), omp=os.environ.get("OMP_NUM_THREADS"), nice=_nice())
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{s}.json"
        if os.path.exists(fo): continue
        path = r["path"] if os.path.isabs(r["path"]) else f"{D}/{r['path']}"
        I = cdc.parse(path); k = q % len(methods); order = methods[k:] + methods[:k]; res = {}
        for meth in order:
            t0 = time.process_time(); w0 = time.time()
            if meth == "hgr": ub, s1, s2, extra = run_hgr(path, B, "")
            elif meth == "hgrnolp": ub, s1, s2, extra = run_hgr(path, B, "nolp")
            elif meth == "ils1": ub, s1, s2, extra = run_ils1(I, B, s)
            elif meth == "ils2": ub, s1, s2, extra = run_ils2(I, B, s)
            else: raise ValueError(meth)
            cpu = time.process_time() - t0; wall = time.time() - w0
            mk = cdc.check_schedule(I, [int(x) for x in s1], [int(x) for x in s2])
            assert abs(mk - ub) < 1e-6, (r["key"], s, meth, mk, ub)
            res[meth] = dict(ub=float(ub), checked=True, cpu=cpu, wall=wall, **extra)
            print(r["key"], s, meth, ub, round(cpu, 2), flush=True)
        rec = dict(key=r["key"], g=r["g"], n1=r["n1"], seed=s, q=q, budget=B, order=order, n2=int(I["n2"]),
                   m1=int(I["m1"]), m2=int(I["m2"]), env=env, **res)
        json.dump(rec, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done", flush=True)


if __name__ == "__main__":
    main()
