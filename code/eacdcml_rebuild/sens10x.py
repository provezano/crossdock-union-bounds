"""Exploratory 10x-calibration sensitivity arm (not part of the prereg families).
EACDCml-ours v2 driven by model_table2_10x.pt (MLP trained on out_tune10x, 8,750 trials), same arm definition as
match.arm_e(..., init="code", es=False) with prediction CPU counted.
usage: sens10x.py repro LIST OUTDIR SHARD NSH          (seed 169, gap vs LB_PCH)
       sens10x.py match LIST OUTDIR SEEDS SHARD NSH    (rival 'ml10' on the confirmation instances)"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import torch; torch.set_num_threads(1)
import eacdcml as E, cdc
from mlp import Predictor
P = Predictor(f"{D}/model_table2_10x.pt")


def arm(I, seed):
    t0 = time.process_time(); p = E.clip_params(P.raw(E.features(I)))
    r = E.run(I, p, seed=seed, t0=t0, init="code", early_stop=False); E.verify(I, r["order"], r["best"])
    r.pop("order"); r.pop("trace", None); r["params"] = p; return r


def main():
    mode = sys.argv[1]; lst, out = sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
    if mode == "repro": seeds, sh, ns = [169], int(sys.argv[4]), int(sys.argv[5])
    else: seeds, sh, ns = [int(x) for x in sys.argv[4].split(",")], int(sys.argv[5]), int(sys.argv[6])
    env = dict(python=sys.version.split()[0], host=platform.node())
    jobs = [(r, s) for s in seeds for r in json.load(open(lst))]
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{s}.json"
        if os.path.exists(fo): continue
        base = D if mode == "repro" else f"{D}/../confirmation_eacdc"
        fp = r["path"] if os.path.isabs(r["path"]) else f"{base}/{r['path']}"
        I = cdc.parse(fp); x = arm(I, s)
        if mode == "repro": x["gap"] = E.gap_ub(x["best"], cdc.lb_pch(I))
        json.dump(dict(key=r["key"], n1=r["n1"], seed=s, ml10=x, env=env), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done")


if __name__ == "__main__":
    main()
