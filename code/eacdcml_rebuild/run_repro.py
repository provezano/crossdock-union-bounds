"""Faithfulness runs: EACDCml-ours (MLP-predicted parameters) and EACDC-paper (static Table 2 parameters) on the
stratified reproduction sample, fixed seed 169 (paper: fixed seed), back to back in one process, arm order
alternating by index. Gap = (UB - LB_PCH)/UB as in the paper.
usage: run_repro.py LIST.json MODEL.pt OUTDIR SHARD NSH [SEED]"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import eacdcml as E, cdc
from mlp import Predictor


def arm_ml(I, pred, seed):
    t0 = time.process_time(); raw = pred.raw(E.features(I)); p = E.clip_params(raw)
    r = E.run(I, p, seed=seed, t0=t0); r["raw_pred"] = raw; E.verify(I, r["order"], r["best"]); return r


def arm_static(I, seed):
    r = E.run(I, E.STATIC, seed=seed); E.verify(I, r["order"], r["best"]); return r


def main():
    lst, model, out, sh, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 169
    os.makedirs(out, exist_ok=True); pred = Predictor(model)
    L = json.load(open(lst))
    for q, r in enumerate(L):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{seed}.json"
        if os.path.exists(fo): continue
        I = cdc.parse(f"{D}/{r['path']}"); lb = cdc.lb_pch(I)
        if q % 2 == 0: ml = arm_ml(I, pred, seed); st = arm_static(I, seed)
        else: st = arm_static(I, seed); ml = arm_ml(I, pred, seed)
        for a in (ml, st): a.pop("order"); a["gap"] = E.gap_ub(a["best"], lb)
        json.dump(dict(key=r["key"], g=r["g"], n1=r["n1"], seed=seed, lb_pch=lb, ml=ml, static=st,
                       first="ml" if q % 2 == 0 else "static",
                       env=dict(python=sys.version.split()[0], host=platform.node())), open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo)
    print("done")


if __name__ == "__main__":
    main()
