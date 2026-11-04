"""Faithfulness of the frozen ML arms on repro210 (seed 169): EACDCml-ours v2 (model_table2.pt) and EACDCml-code
(GA.py + model_code.pt). Static v2 is in out_diag_code_es0. cwd = third/EA-ML, python -I.
usage: run_repro2.py LIST OUTDIR SHARD NSH ARMS(ml,mlc)"""
import sys, os, json
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1]); sys.path.insert(0, D); sys.path.insert(0, f"{D}/../confirmation_eacdc")
import eacdcml as E, cdc
from mlp import Predictor
from match import arm_e, arm_code
lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
arms = sys.argv[5].split(",") if len(sys.argv) > 5 else ["ml", "mlc"]
P2 = Predictor(f"{D}/model_table2.pt") if "ml" in arms else None; PC = Predictor(f"{D}/model_code.pt") if "mlc" in arms else None
for q, r in enumerate(json.load(open(lst))):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): continue
    fp = f"{D}/{r['path']}"; I = cdc.parse(fp); lb = cdc.lb_pch(I)
    res = {}
    if "ml" in arms: res["ml"] = arm_e(I, P2, 169, "code", False)
    if "mlc" in arms: res["mlc"] = arm_code(I, fp, PC, 169)
    for x in res.values(): x["gap"] = E.gap_ub(x["best"], lb)
    json.dump(dict(key=r["key"], g=r["g"], n1=r["n1"], lb_pch=lb, **res), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
print("done")
