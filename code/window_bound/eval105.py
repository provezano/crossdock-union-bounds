"""Phase 2 at scale: new destructive bounds on the dev sample105 (n1 = 20..80). lo = LB* (step 1 cache, fwd/rev),
hi = best known UB (min of HgR, step 5 ILS/CP-SAT 3 seeds). Asserts LB <= best UB. usage: eval105.py OUT SHARD NSH [BUDGET]"""
import sys; sys.dont_write_bytecode = True
import os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset

def ub105():
    U = {}
    for f in glob.glob(f"{D}/snap/out_ub/*.json"):
        d = json.load(open(f)); k = d["key"]
        for a in ("ils", "cpc", "cpw"): U[k] = min(U.get(k, 1e18), d[a]["best"])
    return U

def portfolio(I, T, h, q, cp, det, st):
    if lbx.refute_single(I, T, lbx.Cfg(), h, q, cuts=sorted(set(cp.pick(T, 48, "score")) | set(cp.pick(T, 16, "uniform"))), det=0.5, stats=st): return True
    for K, mode in ((8, "score"), (8, "mix"), (16, "uniform")):
        if lbx.refute_nested(I, T, lbx.Cfg(), h, q, cuts=cp.pick(T, K, mode), det=det, stats=st): return True
    return False

def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); budget = float(sys.argv[4]) if len(sys.argv) > 4 else 120.0
    os.makedirs(out, exist_ok=True); S = json.load(open(f"{D}/../lower_bounds/sample105.json")); U = ub105()
    for q_, r in enumerate(S):
        if q_ % ns != sh: continue
        k = f"{r['g']}_{r['n1']}_{r['seed']}"; fo = f"{out}/{k}.json"
        if os.path.exists(fo): continue
        G = json.load(open(f"{D}/../lower_bounds/out_full/{k}.json"))
        lbs = float(max(G["lb_u1"], G["lb_rev"])); hi = int(min(U.get(k, 1e18), r["ub_hgr"]))
        I = lbx.parse(sset.path(r["g"], r["n1"], r["seed"])); t0 = time.process_time()
        h, q = lbx.heads_tails_bp(I); cp = lbx.CutPicker(I, h, q); res = dict(key=k, g=r["g"], n1=r["n1"], n2=I["n2"], m1=I["m1"], m2=I["m2"], lbstar=lbs, ub=hi)
        st = {}; lb, n, cpu = lbx.destructive(lambda T: lbx.refute_single(I, T, lbx.Cfg(), h, q, cuts=sorted(set(cp.pick(T, 48, "score")) | set(cp.pick(T, 16, "uniform"))), det=0.5, stats=st), lbs, hi, budget=budget)
        res["J_dff"] = dict(lb=lb, n=n, cpu=round(cpu, 2), st=st)
        st = {}; lb2, n, cpu = lbx.destructive(lambda T: portfolio(I, T, h, q, cp, 3.0, st), max(lbs, lb), hi, budget=budget)
        res["N_port"] = dict(lb=lb2, n=n, cpu=round(cpu, 2), st=st); res["pre"] = lbx.destructive(lambda T: lbx.refute_pmtn(I, T, h, q), lbs - 200, hi)[0]
        res["cpu"] = round(time.process_time() - t0, 1)
        assert lb <= hi and lb2 <= hi, ("VALIDITY", k, lb, lb2, hi)
        json.dump(res, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo); print(k, res["cpu"], flush=True)
    print("done", flush=True)

if __name__ == "__main__":
    main()
