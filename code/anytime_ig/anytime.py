"""anytime profile experiment (preregistered; see the paper): anytime profile at 1x, 10x, 100x matched evaluations.
One process per TEST instance (seed s = 51), run with cwd = eacdc_public/third/EA-ML and python -I:
  eacdcR : EACDC static (authors' GA.py, Table-2 configuration, fast identical evaluator, seed redirect), 100 independent
           restarts with seeds s + 1000 r (r = 0..99). Level k = best of the first k runs; its evaluation count = sum of
           all evaluations (init + evalPop) of those runs. Level 1 = the single static run of the earlier preregs.
  mlc    : EACDCml-code (GA.py driven by the frozen model_code.pt MLP), generations scaled: NGEN x k for k = 1, 10, 100
           (three separate runs, seed s; the code's crossover-switch period NGEN/4 scales with NGEN, as in the code).
  then the searches, each run ONCE (seeded) to the largest target and read on its anytime trace at every target:
  fbils  : frozen FB-ILS (ILS v2) configuration;   igdef : IG with the paper's earlier setting (d 4, T 0.4, LS on).
Targets = eacdcR evals at k, mlc evals at k, and k x nev(HgR) from hgr_any.py output (k = 1, 10, 100).
usage: anytime.py LIST OUT HGRDIR SHARD NSH"""
import sys, os, json, time, platform
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1]); sys.path.insert(0, D)
import common as C  # noqa: E402
sys.path.insert(0, f"{D}/../eacdcml_rebuild"); sys.path.insert(0, f"{D}/../eacdc_matched_tim")
import torch  # noqa: E402
torch.set_num_threads(1)
import g3, eacdcml as E  # noqa: E402
from mlp import Predictor  # noqa: E402
from code_arm import run_code  # noqa: E402
SEED = 51; KS = (1, 10, 100)


def eacdc_restarts(fp, s):
    runs = []
    for r in range(max(KS)):
        e = g3.eacdc(None, None, None, s + 1000 * r, budget=None, fpath=fp)
        runs.append(dict(best=float(e["best"]), cpu=e["cpu"], ev=int(e["n_init"] + e["n_evalpop"])))
    lev = {}
    for k in KS:
        R = runs[:k]; lev[str(k)] = dict(best=min(x["best"] for x in R), ev=sum(x["ev"] for x in R), cpu=sum(x["cpu"] for x in R))
    return dict(levels=lev, runs=runs)


def mlc_scaled(I, fp, pred, s):
    t0 = time.process_time(); p = E.clip_params(pred.raw(E.features(I))); tp = time.process_time() - t0; lev = {}
    for k in KS:
        pk = dict(p); pk["NGEN"] = int(p["NGEN"]) * k
        x = run_code(fp, pk, s)
        lev[str(k)] = dict(best=float(x["best"]), ev=int(x["evals"]), cpu=x["cpu"] + tp, ngen=pk["NGEN"])
    return dict(levels=lev, params=p, cpu_pred=tp)


def main():
    lst, out, hdir, sh, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(out, exist_ok=True); PC = Predictor(f"{D}/../eacdcml_rebuild/model_code.pt")
    env = dict(python=sys.version.split()[0], host=platform.node(), machine=platform.machine())
    for q, r in enumerate(json.load(open(lst))):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}.json"
        if os.path.exists(fo): continue
        H = json.load(open(f"{hdir}/{r['key']}.json"))       # HgR must have finished for this instance
        fp = f"{D}/{r['path']}"; I = C.cdc.parse(fp); s = SEED; t_start = time.time()
        if q % 2 == 0: R = eacdc_restarts(fp, s); M = mlc_scaled(I, fp, PC, s); rorder = "eacdcR,mlc"
        else: M = mlc_scaled(I, fp, PC, s); R = eacdc_restarts(fp, s); rorder = "mlc,eacdcR"
        T = {f"eacdcR_{k}": R["levels"][str(k)]["ev"] for k in KS}
        T.update({f"mlc_{k}": M["levels"][str(k)]["ev"] for k in KS}); T.update({f"hgr_{k}": k * H["hgr"]["nev"] for k in KS})
        T.update({f"hgrnolp_{k}": k * H["hgrnolp"]["nev"] for k in KS})
        emax = max(T.values()); S = {}
        sorder = ["fbils", "igdef"] if (q // 2) % 2 == 0 else ["igdef", "fbils"]
        for a in sorder:
            x = C.run_fbils(I, emax, s) if a == "fbils" else C.run_ig(I, emax, s, 4, 0.4, 1)
            S[a] = dict(best=x["best"], nev=x["nev"], cpu=x["cpu"], tcon=x["tcon"], capped=x["capped"],
                        at={t: C.at(x["trace"], e) for t, e in T.items()}, cpu_at={t: C.cpu_at(x["trace"], e) for t, e in T.items()},
                        trace=x["trace"])
        json.dump(dict(key=r["key"], n1=r["n1"], g=r["g"], seed=s, q=q, rival_order=rorder, search_order=sorder, targets=T,
                       eacdcR=R, mlc=M, hgr=H, search=S, env=env, t_start=t_start, t_end=time.time()), open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo); print(r["key"], "done", round(time.time() - t_start), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
