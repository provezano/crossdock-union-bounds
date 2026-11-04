"""step 1 main: exact optima for n1 = 20 and 30 on a stratified sample (30 instances per (dock group, n1) cell,
10 cells -> 300 instances; rng 20261102; the 3 sample105 instances of each cell are included first).
Per instance (one process): EACDC default run (seed 1), ILS 10 s (seed 1, schedule = warm start), then the exact
arms listed (default only the primary arm):
  v2lbu: g5lib.exact_v2 (short galloping bottom probes, then ONE optimisation run with Cmax >= raised LB)
  v2cs : same, started from ceil(Chen-Song LB)
  hlbu : hybrid CP-SAT probing (g5lib.exact_hybrid) started from LB* = max(LB_U*, reversed LB_U*)
  hcs  : same, started from ceil(Chen-Song LB as reported by HgR)  [without-LB_U* comparator]
  onone: the repo model (cdc.cpsat) in optimisation mode, warm start, no LB  [plain baseline]
  olbu : same, with Cmax >= LB*
usage: exact_main.py OUTDIR TL ARMS SHARD NSH [SUBSET]   (SUBSET k: only the first k instances of every cell)"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g5lib as L
import numpy as np


def sample(k=30):
    R = json.load(open(f"{D}/../lower_bounds/allbounds.json"))
    S105 = {(r['g'], r['n1'], r['seed']) for r in json.load(open(f"{D}/../lower_bounds/sample105.json"))}
    rng = np.random.default_rng(20261102); out = []
    for n1 in (20, 30):
        for g in L.GROUPS:
            cell = sorted(r['seed'] for r in R if r['g'] == g and r['n1'] == n1)
            first = sorted(s for s in cell if (g, n1, s) in S105)
            rest = [s for s in rng.permutation(cell).tolist() if s not in first]
            out.append([(g, n1, s) for s in (first + rest)[:k]])
    return out


def main():
    out, tl, arms, sh, ns = sys.argv[1], float(sys.argv[2]), [a for a in sys.argv[3].split(",") if a and a != "base"], int(sys.argv[4]), int(sys.argv[5])
    sub = int(sys.argv[6]) if len(sys.argv) > 6 else 30
    os.makedirs(out, exist_ok=True)
    cells = sample(); inst = [x for c in cells for x in c[:sub]]
    if os.environ.get("N1ONLY"): inst = [x for x in inst if x[1] == int(os.environ["N1ONLY"])]
    inst = [inst[i] for i in np.random.default_rng(5).permutation(len(inst))]  # mix cells across shards
    todo = [(q, x) for q, x in enumerate(inst) if q % ns == sh]
    if os.environ.get("REVERSE"): todo = todo[::-1]   # second worker on the same shard, from the other end
    for q, (g, n1, sd) in todo:
        key = f"{g}_{n1}_{sd}"
        base = f"{out}/{key}_base.json"
        I = L.cdc.parse(L.path(g, n1, sd)); r = L.g1rec(g, n1, sd)
        if os.path.exists(base):
            B = json.load(open(base)); sched = (B["s1"], B["s2"])
        else:
            e = L.eacdc_default(L.path(g, n1, sd), seed=1)
            ub, sched, cpu, tr, constr = L.warm_ub(I, 10.0, seed=1)
            ub4 = min(v for c, v in tr if c <= 4.0 + 1e-9)
            B = dict(key=key, g=g, n1=n1, seed=sd, m1=I["m1"], m2=I["m2"], n2=I["n2"], lbu=L.lbu_star(r),
                     lb_u1=r["lb_u1"], lb_rev=r["lb_rev"], lb_cs=r["lb_hgr"], lb_pch=r["lb_pch"], ub_hgr=r["ub_hgr"],
                     ub_eacdc=e["best"], eacdc_cpu=e["cpu"], ub_ils10=ub, ub_ils4=ub4, constr=constr,
                     s1=list(map(int, sched[0])), s2=list(map(int, sched[1])), py=platform.python_version())
            json.dump(B, open(base + ".tmp", "w")); os.replace(base + ".tmp", base)
        for arm in arms:
            fo = f"{out}/{key}_{arm}_tl{int(tl)}.json"
            if os.path.exists(fo): continue
            ub = B["ub_ils10"]
            if arm[0] in "hv":
                lo = B["lbu"] if arm.endswith("lbu") else float(np.ceil(B["lb_cs"] - 1e-9))
                x = (L.exact_v2 if arm.startswith("v2") else L.exact_hybrid)(I, lo, ub, sched, tl)
                res = dict(opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"], t_opt=x["t_opt"], n_probe=x["n_probe"], log=x["log"])
            else:
                kw = dict(cmax_lb=B["lbu"]) if arm == "olbu" else {}
                x = L.cpsat_exact(I, tl, hint=sched, horizon=int(ub), **kw)
                res = dict(opt=x["ub"] if x["status"] == "OPTIMAL" else None, lo=x["lb"], hi=x["ub"], cpu=x["cpu"],
                           t_opt=x["t_opt"], status=x["status"])
            json.dump(dict(key=key, arm=arm, tl=tl, **res), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
            print(fo, res["opt"], res["lo"], res["hi"], round(res["cpu"], 1), flush=True)
    print("done")


if __name__ == "__main__":
    main()
