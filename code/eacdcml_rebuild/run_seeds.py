"""Seed-to-seed spread of the static arms on repro210: public GA.py (static (3,150,75,100,0.95,0.4)) and EACDC-paper
v2 (Table 2, code-style LS init, no early stop), seeds given. cwd = third/EA-ML, python -I.
usage: run_seeds.py LIST OUTDIR SEEDS(comma) SHARD NSH"""
import sys, os, json
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1]); sys.path.insert(0, D); sys.path.insert(0, f"{D}/../eacdc_matched_tim")
import g3, eacdcml as E, cdc
lst, out = sys.argv[1], sys.argv[2]; seeds = [int(x) for x in sys.argv[3].split(",")]; sh, ns = int(sys.argv[4]), int(sys.argv[5])
os.makedirs(out, exist_ok=True); L = json.load(open(lst)); jobs = [(r, s) for s in seeds for r in L]
for q, (r, s) in enumerate(jobs):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}_s{s}.json"
    if os.path.exists(fo): continue
    fp = f"{D}/{r['path']}"; I = cdc.parse(fp); lb = cdc.lb_pch(I)
    e = g3.eacdc(None, None, None, s, budget=None, fpath=fp)
    v = E.run(I, E.STATIC, seed=s, init="code", early_stop=False)
    json.dump(dict(key=r["key"], g=r["g"], n1=r["n1"], seed=s, lb_pch=lb,
                   gapy=dict(best=float(e["best"]), gap=E.gap_ub(e["best"], lb), cpu=e["cpu"]),
                   v2=dict(best=v["best"], gap=E.gap_ub(v["best"], lb), cpu=v["cpu"])), open(fo + ".tmp", "w"))
    os.replace(fo + ".tmp", fo)
print("done")
