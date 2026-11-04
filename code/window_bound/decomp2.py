"""Versions 3-4 on the solved set: (a) window shaving + nested portfolio (refute_combo), (b) dock-consistent nesting.
Starts from lo = LB* (and separately from the decomposition's N_port value for the stacked value). Asserts <= OPT.
usage: decomp2.py OUT SHARD NSH"""
import sys; sys.dont_write_bytecode = True
import os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset
def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    O = sset.load(); keys = sorted(k for k, v in O.items() if v["opt"] is not None)
    for qd, k in enumerate(keys):
        if qd % ns != sh: continue
        fo = f"{out}/{k}.json"
        if os.path.exists(fo): continue
        v = O[k]; I = lbx.parse(sset.path(v["g"], v["n1"], v["seed"])); opt = int(v["opt"]); h, q = lbx.heads_tails_bp(I); cp = lbx.CutPicker(I, h, q)
        lo, hi = int(v["lbu"]), int(v["best_hi"]); r = dict(key=k, g=v["g"], n1=v["n1"], opt=opt, lbstar=lo)
        lb, n, cpu = lbx.destructive(lambda T: lbx.refute_shave(I, T, h, q, det=0.1, budget=60), lo, hi, budget=300); r["shave"] = dict(lb=lb, n=n, cpu=round(cpu, 1))
        lb, n, cpu = lbx.destructive(lambda T: lbx.refute_combo(I, T, h, q, cp), lo, hi, budget=600); r["combo"] = dict(lb=lb, n=n, cpu=round(cpu, 1))
        lb, n, cpu = lbx.destructive(lambda T: lbx.refute_nested_docks(I, T, h, q, cp.pick(T, 8, "score"), det=5.0), lo, hi, budget=300); r["docks"] = dict(lb=lb, n=n, cpu=round(cpu, 1))
        for nm in ("shave", "combo", "docks"): assert r[nm]["lb"] <= opt, ("VALIDITY", k, nm, r[nm]["lb"], opt)
        json.dump(r, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo); print(k, flush=True)
    print("done", flush=True)
if __name__ == "__main__":
    main()
