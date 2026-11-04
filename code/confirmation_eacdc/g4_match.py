"""Review fixes B1-B3 / C (2026-11-02): per-run matched ILS vs EACDC, interleaved in ONE process.
For each (instance, seed): one full, untruncated EACDC run (authors' GA.py + static config, fast identical
evaluator/loader, seed redirect only) and one ILS run, back to back in the same interpreter; the arm that runs
first alternates with the task index. ILS budget = that EACDC run's own CPU (constructive time counted) AND at
least as many decoder evaluations as EACDC made (init schedules + evalPop calls); cap 3x CPU + 1 s.
usage: g4_match.py LIST.json OUTDIR SEEDS(comma) SHARD NSH
LIST.json: [{"key":..., "path":...}] (path absolute or relative to this file's dir)."""
import sys, os, json, time, platform, socket
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../eacdc_matched_tim")
import numpy as np
import g3
g2, cdc = g3.g2, g3.cdc


def ils_e(I, side, order, cpu_budget, eval_budget, seed, t0, n0, cap):
    """g3.ils_p (default constants: stall 40, 2 perturbation moves, segment reversal) with an evaluation
    counter; trace entries (cpu since t0, evals since start incl. constructive, best)."""
    J = I if side == "f" else g2.reverse(I)
    Dd = g2.Dec(J); rng = np.random.default_rng(seed)
    nev = n0
    cur = list(order); fc = Dd(cur); nev += 1; best, bo = fc, list(cur); n = len(cur)
    trace = [(time.process_time() - t0, nev, best)]; stall = 0
    while True:
        c = time.process_time() - t0
        if (c >= cpu_budget and nev >= eval_budget) or c >= cap: break
        a, b = rng.integers(n, size=2)
        if a == b: continue
        nb = list(cur); x = nb.pop(a); nb.insert(b, x)
        f = Dd(nb); nev += 1
        if f <= fc:
            stall = 0 if f < fc else stall + 1; cur, fc = nb, f
            if f < best: best, bo = f, list(nb); trace.append((time.process_time() - t0, nev, best))
        else: stall += 1
        if stall > 40 * n:
            cur = list(bo)
            a, b = sorted(rng.integers(n, size=2)); cur[a:b + 1] = cur[a:b + 1][::-1]
            for _ in range(2):
                a, b = rng.integers(n, size=2); x = cur.pop(a); cur.insert(b, x)
            fc = Dd(cur); nev += 1; stall = 0
    return best, bo, trace, nev, time.process_time() - t0


HORIZON = {20: 1.5, 30: 2.7, 40: 3.8, 50: 7.5, 60: 11.5, 70: 30.5, 80: 35.0}


def run_ils(I, cpu_budget, eval_budget, seed):
    t0 = time.process_time()
    C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); n0 = len(C)
    side, order = C[kb][1], C[kb][2]
    b, bo, tr, nev, cpu = ils_e(I, side, order, cpu_budget, eval_budget, 100 + seed, t0, n0, 3 * cpu_budget + 1.0)
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6
    return dict(best=float(b), trace=tr, nev=nev, cpu=cpu, constr=float(C[kb][0]), n_constr=n0)


def run_eacdc(fp, seed):
    w = time.time(); e = g3.eacdc(None, None, None, seed, budget=None, fpath=fp)
    return dict(best=float(e["best"]), cpu=e["cpu"], n_init=e["n_init"], n_evalpop=e["n_evalpop"],
                trace=e["trace"], trace_e=e["trace_e"], wall=time.time() - w)


def main():
    lst, out, seeds, sh, ns = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(out, exist_ok=True)
    L = json.load(open(lst)); jobs = [(r, s) for s in seeds for r in L]
    env = dict(python=sys.version.split()[0], impl=platform.python_implementation(), host=socket.gethostname(),
               machine=platform.machine(), exe=sys.executable)
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{s}.json"
        if os.path.exists(fo): continue
        fp = r["path"] if os.path.isabs(r["path"]) else f"{D}/{r['path']}"
        I = cdc.parse(fp); first = "eacdc" if q % 2 == 0 else "ils"
        t_start = time.time()
        if first == "eacdc":
            e = run_eacdc(fp, s); i = run_ils(I, e["cpu"], e["n_init"] + e["n_evalpop"], s)
        else:
            # ILS first: its matched budget is not known yet, so ILS runs to a fixed per-n1 horizon
            # (1.5 x the max full-EACDC CPU seen on sample105) and is read a posteriori at EACDC's CPU and
            # evaluation count (the anytime value at B is a prefix of the same seeded trajectory). If EACDC
            # exceeds the horizon, ILS is read at the horizon (conservative against ILS; flagged).
            H = HORIZON[I["n1"]]
            i = run_ils(I, H, 0, s); e = run_eacdc(fp, s); i["horizon"] = H
        json.dump(dict(key=r["key"], seed=s, first=first, eacdc=e, ils=i, env=env, t_start=t_start,
                       t_end=time.time()), open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo)
    print("done")


if __name__ == "__main__":
    main()
