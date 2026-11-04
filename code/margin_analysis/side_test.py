"""Exploratory H(d) follow-up: is the side choice (time reversal) the responsible component? For each n1 = 30 structured
matched instance and seeds 11-13, run the base ILS (diag_d.ils_generic, frozen constants) from the best constructive
of EACH side separately, at EACDC's evaluation count (from d_out). usage: side_test.py OUT"""
import sys, os, json, glob
F = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, f"{F}/fbils_census_exact_cpuc"); sys.path.insert(0, f"{F}/exact_cpsat")
import g5lib as L; g2 = L.g2
import importlib.util
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_d.py")).read()
ns = {}; exec(src.split("jobs = [")[0].split("out, sh, ns =")[0] + src.split("out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)")[1].split("jobs = [")[0], ns)
ils = ns["ils_generic"]; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob("d_out/*.json")):
    d = json.load(open(f)); fo = f"{out}/{os.path.basename(f)}"
    if os.path.exists(fo): continue
    I = L.cdc.parse(f"{F}/fbils_census_exact_cpuc/data_struct/{d['key']}.dat"); C = g2.all_constructive(I); res = {}
    for side in ("f", "r"):
        ks = [k for k in C if C[k][1] == side]; kb = min(ks, key=lambda k: C[k][0])
        J = I if side == "f" else g2.reverse(I); b, o = ils(g2.Dec(J), C[kb][2], d["evb"], 100 + d["seed"])
        res[side] = dict(constr=float(C[kb][0]), ils=float(b))
    json.dump(dict(key=d["key"], seed=d["seed"], eacdc=d["eacdc_best"], chosen=d["side"], **res), open(fo, "w"))
    print(d["key"], d["seed"], d["eacdc_best"], res, flush=True)
