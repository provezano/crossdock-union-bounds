"""Faithfulness of EACDCml-code on repro210: GA.py static (3,150,75,100,0.95,0.4) and GA.py + model_code.pt, same
process/machine, seed 169. cwd = third/EA-ML, python -I. usage: run_repro_code.py LIST OUTDIR SHARD NSH"""
import sys, os, json, time
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1]); sys.path.insert(0, D); sys.path.insert(0, f"{D}/../eacdc_matched_tim")
import g3, eacdcml as E, cdc
from mlp import Predictor
from code_arm import run_code
lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
PC = Predictor(f"{D}/model_code.pt")
for q, r in enumerate(json.load(open(lst))):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): continue
    fp = f"{D}/{r['path']}"; I = cdc.parse(fp); lb = cdc.lb_pch(I)
    t0 = time.process_time(); p = E.clip_params(PC.raw(E.features(I))); m = run_code(fp, p, 169); m["cpu"] = time.process_time() - t0
    s = g3.eacdc(None, None, None, 169, budget=None, fpath=fp)
    st = dict(best=float(s["best"]), cpu=s["cpu"], evals=s["n_init"] + s["n_evalpop"])
    m.pop("trace", None); m.pop("trace_e", None)
    for x in (m, st): x["gap"] = E.gap_ub(x["best"], lb)
    json.dump(dict(key=r["key"], g=r["g"], n1=r["n1"], lb_pch=lb, mlc=m, gapy=st), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
print("done")
