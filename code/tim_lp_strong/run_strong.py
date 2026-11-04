"""Run TIM-LP (old) and strengthened variants on the instance sets of the TIM-LP comparison.
usage: run_strong.py small OUT SHARD NSH     300 generated instances (regenerated, old m1 = m2 rule, rng 20261102)
       run_strong.py real  OUT SHARD NSH     the 50 n1 = 20 benchmark instances of eacdc_matched_tim out_tim_real (T = HgR UB)
       run_strong.py large OUT SHARD NSH TL  the 9 n1 = 30-50 instances of eacdc_matched_tim out_tim_large
UPB (lb_u1), OPT and HgR UB are read from the eacdc_matched_tim result files (same instances, same values as in tab_tim)."""
import sys, os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); G3 = f"{D}/../eacdc_matched_tim"
sys.path.insert(0, G3); sys.path.insert(0, f"{D}/../decoder")
from ortools.sat.python import cp_model  # noqa: load before highspy (dylib clash)
import numpy as np
import cdc
from tim_lp import tim_lp
from tim_lp_strong import tim_lp_strong

F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}


def gen_old(rng, n1, mt):  # verbatim copy of the earlier eacdc_matched_tim/run_tim.py gen() (m1 = m2 rule)
    n2 = int(rng.integers(int(np.ceil(0.8 * n1)), int(np.floor(1.2 * n1)) + 1))
    m = {'m2': 2, 'm4': 4, 'u24': int(rng.integers(2, 5))}[mt]
    P = rng.random((n2, n1)) < 0.5
    for j in range(n2):
        if not P[j].any(): P[j, rng.integers(n1)] = True
    for i in range(n1):
        if not P[:, i].any(): P[rng.integers(n2), i] = True
    return dict(n1=n1, n2=n2, m1=m, m2=m, p1=rng.integers(10, 101, n1), p2=rng.integers(10, 101, n2), prec=P)


def small_instances():
    rng = np.random.default_rng(20261102); out = []
    for n1 in (4, 5, 6, 7, 8):
        for mt in ('m2', 'm4', 'u24'):
            for k in range(20):
                out.append((f"{mt}_{n1}_{k}", gen_old(rng, n1, mt)))
    return out


def solve_all(I, T, tl, variants):
    res = {}
    if "base" in variants:
        w = time.time(); b = tim_lp(I, T=T, time_limit=tl); res["base"] = dict(lb=b["lb"], status=b["status"], cpu=b["cpu"], wall=time.time() - w)
    for v in variants:
        if v == "base": continue
        r = tim_lp_strong(I, T=T, variant=v, time_limit=tl); res[v] = r
    return res


def main():
    mode, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out, exist_ok=True)
    if mode == "small":
        for q, (name, I) in enumerate(small_instances()):
            if q % ns != sh: continue
            fo = f"{out}/{name}.json"
            if os.path.exists(fo): continue
            old = json.load(open(f"{G3}/out_tim_small/{name}.json"))
            assert old["n2"] == I["n2"] and old["m"] == I["m1"], name
            res = solve_all(I, int(old["T"]), None, ("base", "A", "AC", "ABC"))
            json.dump(dict(name=name, n1=old["n1"], mt=old["mt"], opt=old["opt"], opt_status=old["opt_status"],
                           lb_u1=old["lb_u1"], old_tim=old["tim"], T=old["T"], res=res), open(fo, "w"))
    else:
        src = "out_tim_real" if mode == "real" else "out_tim_large"
        tl = 1800.0 if mode == "real" else float(sys.argv[5])
        files = sorted(glob.glob(f"{G3}/{src}/*.json"))
        for q, f in enumerate(files):
            if q % ns != sh: continue
            r = json.load(open(f)); fo = f"{out}/{os.path.basename(f)}"
            if os.path.exists(fo): continue
            I = cdc.parse(f"{G3}/../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
            res = solve_all(I, int(r["ub_hgr"]), tl, ("base", "ABC"))
            json.dump(dict(g=r["g"], n1=r["n1"], seed=r["seed"], ub_hgr=r["ub_hgr"], lb_u1=r["lb_u1"],
                           old_tim=r["tim"], old_tim_cpu=r["tim_cpu"], T=int(r["ub_hgr"]), res=res), open(fo, "w"))
            print(os.path.basename(f), {k: (v["lb"], round(v["cpu"], 1)) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
