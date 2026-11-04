"""exact v3 development main run: exact_v3 (CP-SAT + reified LB_U*-style union cuts) on the SAME 300 step 1 instances
(exact_cpsat/exact_main.sample), SAME 300 s limit, SAME start (LB* = max(LB_U*, reversed)) and warm start (the
step 1 base ILS-10 s schedule). Optimal schedules are saved and re-checked. usage: exact_cpuc_main.py OUT TL SHARD NSH [K]"""
import sys, os, json, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, f"{D}/../exact_cpsat")
import g6cp, exact_main, g5lib as L  # noqa
import numpy as np  # noqa

def main():
    out, tl, sh, ns = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); k = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    os.makedirs(out, exist_ok=True)
    inst = [x for c in exact_main.sample() for x in c]
    inst = [inst[i] for i in np.random.default_rng(5).permutation(len(inst))]
    for q, (g, n1, sd) in enumerate(inst):
        if q % ns != sh: continue
        key = f"{g}_{n1}_{sd}"; fo = f"{out}/{key}_v3k{k}_tl{int(tl)}.json"
        if os.path.exists(fo): continue
        B = json.load(open(f"{D}/../exact_cpsat/out_exact/{key}_base.json"))
        I = L.cdc.parse(L.path(g, n1, sd))
        x = g6cp.exact_v3(I, B["lbu"], B["ub_ils10"], (B["s1"], B["s2"]), tl, k=k)
        if x["sched"] is not None: assert abs(L.cdc.check_schedule(I, *x["sched"]) - x["hi"]) < 1e-6
        res = dict(key=key, arm=f"v3k{k}", tl=tl, opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"], t_opt=x["t_opt"],
                   t_probe=x["t_probe"], n_probe=x["n_probe"], log=x["log"], py=platform.python_version(), host=platform.node(),
                   sched=[list(map(int, x["sched"][0])), list(map(int, x["sched"][1]))] if x["sched"] is not None else None)
        json.dump(res, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
        print(key, res["opt"], res["lo"], res["hi"], round(res["cpu"], 1), flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
