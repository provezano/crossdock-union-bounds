"""Census on all Cota instances with n1 in {20, 30} (3,000): portfolio bound (single-cut J+DFF, then nested) from
lo = LB* (step 1 cache), hi = best known UB = min(HgR published, step 5 ILS-10 s bases, step 6 ILS census snapshot).
Asserts LB <= best UB (validity). usage: census.py OUT SHARD NSH [BUDGET]"""
import sys; sys.dont_write_bytecode = True
import os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset, eval105
def best_ub():
    U = {}
    for r in json.load(open(f"{D}/../lower_bounds/allbounds.json")): U[f"{r['g']}_{r['n1']}_{r['seed']}"] = float(r["ub_hgr"])
    for f in glob.glob(f"{D}/snap/out_exact/*_base.json"):
        b = json.load(open(f)); U[b["key"]] = min(U[b["key"]], float(b["ub_ils10"]))
    for f in glob.glob(f"{D}/snap/out_ils/*.jsonl"):
        for l in open(f):
            try: r = json.loads(l); U[r["key"]] = min(U[r["key"]], float(r["ub10"]))
            except Exception: pass
    for k, v in eval105.ub105().items(): U[k] = min(U[k], float(v))
    return U
def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); budget = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
    os.makedirs(out, exist_ok=True); U = best_ub(); fo = f"{out}/census_{sh}.jsonl"
    done = {json.loads(l)["key"] for l in open(fo)} if os.path.exists(fo) else set(); f = open(fo, "a")
    keys = sorted(k for k in U if int(k.split("_")[1]) in (20, 30))
    for q_, k in enumerate(keys):
        if q_ % ns != sh or k in done: continue
        g, n1, sd = k.split("_"); G = json.load(open(f"{D}/../lower_bounds/out_full/{k}.json"))
        lo = float(max(G["lb_u1"], G["lb_rev"])); hi = int(U[k]); I = lbx.parse(sset.path(g, int(n1), int(sd))); t0 = time.process_time()
        h, q = lbx.heads_tails_bp(I); cp = lbx.CutPicker(I, h, q); st = {}
        lb, n, cpu = lbx.destructive(lambda T: eval105.portfolio(I, T, h, q, cp, 2.0, st), lo, hi, budget=budget)
        assert lb <= hi, ("VALIDITY", k, lb, hi)
        f.write(json.dumps(dict(key=k, g=g, n1=int(n1), lbstar=lo, lb=lb, ub=hi, n=n, cpu=round(time.process_time() - t0, 2))) + "\n"); f.flush()
    print("done", flush=True)
if __name__ == "__main__":
    main()
