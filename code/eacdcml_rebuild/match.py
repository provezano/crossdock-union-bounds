"""Preregistered comparison: our ILS (frozen g4_match.run_ils, constants unchanged)
vs four EACDC-line rivals, per (instance, seed), all interleaved in ONE process:
  ml   = EACDCml-ours v2 (paper spec + code-style LS init, no early stop; MLP model_table2.pt)       [primary rival]
  st   = EACDC-paper v2 (same algorithm, static Table 2 parameters)
  mlc  = EACDCml-code (authors' public GA.py driven by MLP model_code.pt)
  ml1  = EACDCml-ours v1 (paper-literal; MLP model.pt)                                         [secondary]
Order: even task index -> the four rivals, then ILS run until it has passed every rival's CPU and evaluation count
(cap 3x max CPU + 1 s, g4_match); odd -> ILS first to the frozen per-n1 horizon (horizon.json), then the rivals.
ILS is read a posteriori on its seeded anytime trace at each rival's own CPU time and evaluation count.
Run with cwd = eacdc_public/third/EA-ML and python -I (third-party GA.py is executed in-process).
usage: match.py LIST.json OUTDIR SEEDS(comma) SHARD NSH"""
import sys, os, json, time, platform
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1])
sys.path.insert(0, D); sys.path.insert(0, f"{D}/../confirmation_eacdc")
import g4_match                      # frozen ILS v1 harness
sys.path.insert(0, f"{D}/../fbils_frozen")
import confirm_fbils as CV              # frozen ILS v2 driver (byte-identical, fbils_frozen/SHA256SUMS); imports frozen srch
import eacdcml as E, cdc
from mlp import Predictor
from code_arm import run_code

HORIZON = json.load(open(f"{D}/horizon.json"))


def arm_e(I, pred, seed, init, es):
    t0 = time.process_time(); p = E.clip_params(pred.raw(E.features(I))) if pred else dict(E.STATIC)
    r = E.run(I, p, seed=seed, t0=t0, init=init, early_stop=es); E.verify(I, r["order"], r["best"]); r.pop("order")
    r.pop("trace", None); return r


def arm_code(I, fp, pred, seed):
    t0 = time.process_time(); p = E.clip_params(pred.raw(E.features(I))); x = run_code(fp, p, seed)
    tot = time.process_time() - t0; x["cpu_pred"] = tot - x["cpu"]; x["cpu"] = tot   # prediction CPU counted
    x.pop("trace", None); x.pop("trace_e", None); return x


def run_ils2(I, cpu_b, ev_b, seed):
    """ILS v2 = CV.c_arm(I, "v2", ...) line for line (frozen constants: pfbi 0, stall 40, rounds 3, wfrac 0.25,
    seed 100+s, nev0 = 14 constructive decodes, cap 3*cpu_b + 2), but returning the anytime trace (cpu incl.
    constructive, evals incl. every FBI and window-insertion decode, best) so it can be read at several budgets."""
    g2 = CV.g2; t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I); tcon = time.process_time() - t0
    b, bo, tr, ne = CV.srch.run_fbi(J, g2.reverse(J), order, max(0.0, cpu_b - tcon), ev_b, cap=3 * cpu_b + 2, seed=100 + seed,
                                    nev0=len(C), pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    assert g2.Dec(J)(bo) == b
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(CV.L.cdc.check_schedule(I, s1, s2) - b) < 1e-6
    return dict(best=float(b), trace=[(c + tcon, e, v) for c, e, v in tr], nev=ne, tcon=tcon, cpu=time.process_time() - t0)


def main():
    lst, out = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")]; sh, ns = int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(out, exist_ok=True)
    P2, PC, P1 = Predictor(f"{D}/model_table2.pt"), Predictor(f"{D}/model_code.pt"), Predictor(f"{D}/model.pt")
    L = json.load(open(lst)); jobs = [(r, s) for s in seeds for r in L]
    env = dict(python=sys.version.split()[0], host=platform.node(), machine=platform.machine())
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{s}.json"
        if os.path.exists(fo): continue
        fp = r["path"] if os.path.isabs(r["path"]) else f"{D}/../confirmation_eacdc/{r['path']}"
        I = cdc.parse(fp)
        def rivals():
            return dict(ml=arm_e(I, P2, s, "code", False), st=arm_e(I, None, s, "code", False),
                        mlc=arm_code(I, fp, PC, s), ml1=arm_e(I, P1, s, "paper", True))
        flip = (q // 2) % 2 == 1          # order of ILS v1 / v2 within the ILS block
        if q % 2 == 0:
            R = rivals(); cb = max(x["cpu"] for x in R.values()); eb = max(x["evals"] for x in R.values())
            if flip: ils2 = run_ils2(I, cb, eb, s); ils = g4_match.run_ils(I, cb, eb, s)
            else: ils = g4_match.run_ils(I, cb, eb, s); ils2 = run_ils2(I, cb, eb, s)
            first = "rivals"
        else:
            H = HORIZON[str(I["n1"])]
            if flip: ils2 = run_ils2(I, H, 0, s); ils = g4_match.run_ils(I, H, 0, s)
            else: ils = g4_match.run_ils(I, H, 0, s); ils2 = run_ils2(I, H, 0, s)
            ils["horizon"] = ils2["horizon"] = H; R = rivals(); first = "ils"
        json.dump(dict(key=r["key"], n1=I["n1"], seed=s, first=first, ils_order="v2v1" if flip else "v1v2", rivals=R, ils=ils,
                       ils2=ils2, env=env), open(fo + ".tmp", "w"))
        os.replace(fo + ".tmp", fo)
    print("done")


if __name__ == "__main__":
    main()
