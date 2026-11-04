"""Matched comparison: HgR (H1/H2 re-implementation, budget mode) vs ILS v1 (g2.ils, the paper's ILS) vs ILS v2
(srch.run_fbi, compiled C), same machine, same per-instance CPU budget (process time, constructives included),
interleaved per instance in ONE process (method order rotates per instance). One JSONL line per (instance, method).
usage: match.py LIST.json BUDGET OUT.jsonl [methods=hgr,ils1,ils2] [mode=budget|code]
  mode=code runs only HgR with MainVFR semantics, capped at the published Tempo_MD of the instance (fidelity run)."""
import sys, os, json, time, platform
H = os.path.dirname(os.path.abspath(__file__)); FD = os.path.dirname(os.path.dirname(H))
sys.path.insert(0, H); sys.path.insert(1, f"{FD}/decoder"); sys.path.insert(2, f"{FD}/fbils_census_exact_cpuc")
import hgr, cdc, g2  # noqa: E402  (cdc/g2 imported read-only from decoder)
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
DATA = f"{FD}/benchmark_data/data/Dados"
hgr.LP_OPTS = dict(solver="ipm")  # IPM + crossover (vertex solution, like CPLEX's default simplex); see findings


def run_hgr(path, B, mode, variant=""):
    """variant '': as coded (H2 swap bug kept); 'fix': H2 swap bug corrected; 'nolp': LP skipped (multipliers stay
    null, orders by index tie-break), whole budget to H1/H2 + swap LS."""
    P = hgr.load(path)
    r = hgr.hgr(P, mode=mode, budget=B, h2_bug=(variant != "fix"), lp_frac=(0.0 if variant == "nolp" else 0.5),
                lp_cap=(0.0 if variant == "nolp" else 360.0))
    return r["ub"], r["s1"], r["s2"], dict(cpu=r["cpu"], ub1=r["ub1"], ub2=r["ub2"], nev=r["nev"], stop=r["stop"],
                                           rounds=r["rounds"], lp=r["lp"])


def run_ils1(I, B, seed):
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0])
    tcon = time.process_time() - t0
    b, bo, tr = g2.ils(I, C[kb][1], C[kb][2], max(0.0, B - tcon), seed=100 + seed)
    cm, s1, s2 = g2.schedule_from(I, C[kb][1], bo)
    return b, s1, s2, dict(cpu=time.process_time() - t0, tcon=tcon, constr=kb)


def run_ils2(I, B, seed):
    import srch
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I); tcon = time.process_time() - t0
    b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, max(0.0, B - tcon), 0, cap=B + 1, seed=100 + seed, nev0=len(C),
                                 pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    cm, s1, s2 = g2.schedule_from(I, side, bo)
    return b, s1, s2, dict(cpu=time.process_time() - t0, tcon=tcon, constr=kb, nev=ne)


def main():
    L = json.load(open(sys.argv[1])); B = float(sys.argv[2]); out = sys.argv[3]
    methods = (sys.argv[4] if len(sys.argv) > 4 else "hgr,ils1,ils2").split(",")
    mode = sys.argv[5] if len(sys.argv) > 5 else "budget"
    pub = json.load(open(f"{H}/pub_hgr.json"))
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            try: d = json.loads(ln); done.add((d["key"], d["method"]))
            except Exception: pass
    f = open(out, "a")
    for q, r in enumerate(L):
        key = f"{r['g']}_{r['n1']}_{r['seed']}"; path = f"{DATA}/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat"
        I = cdc.parse(path); pr = pub[key]
        ms = methods[q % len(methods):] + methods[:q % len(methods)]  # rotate order per instance
        for meth in ms:
            if (key, meth) in done: continue
            budget = B if mode == "budget" else min(B, float(pr["time"]))
            t0 = time.process_time(); w0 = time.time()
            if meth.startswith("hgr"): ub, s1, s2, extra = run_hgr(path, budget, mode, meth[3:])
            elif meth == "ils1": ub, s1, s2, extra = run_ils1(I, budget, 1)
            elif meth == "ils2": ub, s1, s2, extra = run_ils2(I, budget, 1)
            cpu = time.process_time() - t0; wall = time.time() - w0
            mk = cdc.check_schedule(I, list(map(int, s1)), list(map(int, s2)))
            assert abs(mk - ub) < 1e-6, (key, meth, mk, ub)
            rec = dict(key=key, g=r["g"], n1=r["n1"], seed=r["seed"], n2=I["n2"], m1=I["m1"], m2=I["m2"], method=meth,
                       mode=mode, budget=budget, ub=float(ub), checked=True, cpu=cpu, wall=wall, pub_h1=pr["h1"],
                       pub_h2=pr["h2"], pub_time=pr["time"], lb_cs=pr["lb_md"], py=platform.python_version(),
                       host=platform.node(), **{k: v for k, v in extra.items() if k != "cpu"})
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(key, meth, ub, round(cpu, 2), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
