"""step 5: CP-SAT (repo model, 1 worker) as a matched-CPU UB comparator vs ILS on sample105, 4 s and 10 s.
Per (instance, seed), in ONE process, arm order rotated by task index:
  ils  : best step 2 constructive + ILS (frozen constants), 10 s CPU incl. constructive; anytime trace.
  cpc  : CP-SAT cold (no hint, horizon sum p), 10 s; improvement trace by process CPU (callback).
  cpw  : CP-SAT warm: same best constructive schedule as hint and horizon = its makespan; constructive CPU counted.
Values at 4 s and 10 s are read from the traces (no fallback: None if no solution yet).
usage: g5_ub.py OUTDIR SEEDS SHARD NSH"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g5lib as L
import numpy as np
cdc, g2 = L.cdc, L.g2
import g3  # noqa: E402

B = 10.0


def run_ils(I, seed):
    t0 = time.process_time()
    C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    tc = time.process_time() - t0
    b, bo, tr = g3.ils_p(I, side, order, B - tc, seed=100 + seed, t_offset=tc)
    return dict(best=float(b), trace=[(float(c), float(v)) for c, v in tr], constr=float(C[kb][0]), t_constr=tc)


def run_cp(I, seed, warm):
    t0 = time.process_time(); hint = None; H = None; tc = 0.0
    if warm:
        C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0])
        cm, s1, s2 = g2.schedule_from(I, C[kb][1], C[kb][2]); hint = (s1, s2); H = int(cm)
        tc = time.process_time() - t0
    x = L.cpsat_exact(I, B - tc, hint=hint, horizon=H, seed=seed, strong=True)
    tr = [(tc + c, float(v)) for c, w, v in x["trace"]]
    if warm: tr = [(tc, float(H))] + tr
    return dict(best=x["ub"], status=x["status"], lb=x["lb"], trace=tr, t_constr=tc)


def at(tr, b):
    v = [x[1] for x in tr if x[0] <= b + 1e-9]
    return min(v) if v else None


def main():
    out, seeds, sh, ns = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    S = json.load(open(f"{D}/../lower_bounds/sample105.json"))
    jobs = [(r, s) for s in seeds for r in S]
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        key = f"{r['g']}_{r['n1']}_{r['seed']}"; fo = f"{out}/{key}_s{s}.json"
        if os.path.exists(fo): continue
        I = cdc.parse(L.path(r['g'], r['n1'], r['seed']))
        arms = ["ils", "cpc", "cpw"]; k = q % 3; arms = arms[k:] + arms[:k]
        res = {}
        for a in arms:
            res[a] = run_ils(I, s) if a == "ils" else run_cp(I, s, warm=(a == "cpw"))
            for b in (4.0, 10.0): res[a][f"at{int(b)}"] = at(res[a]["trace"], b)
        json.dump(dict(key=key, g=r['g'], n1=r['n1'], seed=s, order=arms, py=platform.python_version(), **res),
                  open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo)
    print("done")


if __name__ == "__main__":
    main()
