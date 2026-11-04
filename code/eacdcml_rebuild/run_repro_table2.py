"""Faithfulness of EACDCml-ours v2 on repro210: ML arm (model_table2.pt) and static arm (Table 2), same process and
machine, seed 169, code-style LS init, no early stop. usage: run_repro_table2.py LIST OUTDIR SHARD NSH"""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import eacdcml as E, cdc
from mlp import Predictor
lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
P = Predictor(f"{D}/model_table2.pt")
def arm(I, pred):
    t0 = time.process_time(); p = E.clip_params(pred.raw(E.features(I))) if pred else dict(E.STATIC)
    r = E.run(I, p, seed=169, t0=t0, init="code", early_stop=False); E.verify(I, r["order"], r["best"]); r.pop("order"); r.pop("trace"); return r
for q, r in enumerate(json.load(open(lst))):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/{r['path']}"); lb = cdc.lb_pch(I)
    a, b = (arm(I, P), arm(I, None)) if q % 2 == 0 else (arm(I, P), arm(I, None))[::1]
    for x in (a, b): x["gap"] = E.gap_ub(x["best"], lb)
    json.dump(dict(key=r["key"], g=r["g"], n1=r["n1"], lb_pch=lb, ml=a, static=b), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
print("done")
