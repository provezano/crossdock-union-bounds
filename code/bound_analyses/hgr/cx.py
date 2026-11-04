"""Empirical complexity of the HgR re-implementation (hgr.py) vs n1, on the 105 dev instances (3 per cell).
Measures: one decode (H1 build) time + op count; one H2 build (list + decode); evaluations and CPU of one full
first-improvement swap LS (H1, from the index order, no LP); LP size (cols, rows, nnz, CSR bytes) of both stage LPs at
the initial horizon T0 (built, not solved). LP solve times come from the fidelity run (out/fid_code35.jsonl).
Writes out/cx.json and prints log-log slope fits vs n1.  usage: cx.py"""
import sys, os, json, time, math
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import hgr  # noqa: E402
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
DATA = f"{H}/../../benchmark_data/data/Dados"


def lp_size(P, Tint):
    """Count LP dimensions exactly as hgr._stage_lp builds them (z-form), and the x-form of the paper's model."""
    out = {}
    for st, p, n, a in (("1", P["p1"], P["n"], [0] * P["n"]), ("2", P["p2"], P["m"], None)):
        if a is None:
            _, pe1 = hgr.lb_cs(P); a = [math.ceil(v) for v in pe1]
        b = [Tint - x for x in p]; a = [min(a[j], b[j]) for j in range(n)]
        cols = sum(b[j] - a[j] + 1 for j in range(n)); mono = sum(b[j] - a[j] for j in range(n))
        capnz = sum(len(range(a[j], min(b[j] + p[j], Tint + 1))) + max(0, min(b[j] + p[j], Tint + 1) - (a[j] + p[j]))
                    for j in range(n))
        rows = mono + Tint + 1 + (n if st == "2" else 0)
        nnz = 2 * mono + capnz + (sum(b[j] - a[j] + 1 for j in range(n)) if st == "2" else 0)
        xnnz = cols + sum(min(p[j], Tint + 1) * (b[j] - a[j] + 1) for j in range(n))  # assignment + capacity (x-form)
        out[st] = dict(cols=cols + (1 if st == "2" else 0), rows=rows, nnz=nnz, x_rows=n + Tint + 1 + (n if st == "2" else 0),
                       x_nnz=xnnz + (n * 0))
    return out


def main():
    L = json.load(open(f"{H}/list105.json")); rng = np.random.default_rng(0); recs = []
    for r in L:
        P = hgr.load(f"{DATA}/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat"); n, m = P["n"], P["m"]
        D = hgr.Decoder(P); E = sum(len(x) for x in P["pred"])
        orders = [list(rng.permutation(n)) for _ in range(200)]
        t = time.process_time()
        for o in orders: D(o)
        tdec = (time.process_time() - t) / 200
        suc = [len(P["succ"][j]) for j in range(n)]; oy = [list(rng.permutation(m)) for _ in range(100)]
        t = time.process_time()
        for o in oy: D(hgr.h2_list(P, o, suc))
        th2 = (time.process_time() - t) / 100
        # one full H1 swap LS from the index-tie order (as the 'nolp' variant), single round, counting evaluations
        D.nev = 0; t = time.process_time()
        r1 = hgr.hgr(P, mode="budget", budget=1e9, lp_frac=0.0, lp_cap=0.0, n_rounds=1)
        tls = time.process_time() - t
        T0 = int(math.floor(hgr.horizon0(P))); sz = lp_size(P, T0)
        recs.append(dict(key=f"{r['g']}_{r['n1']}_{r['seed']}", g=r["g"], n1=n, n2=m, m1=P["maq1"], m2=P["maq2"], E=E, T0=T0,
                         t_decode=tdec, t_h2build=th2, ls_evals=r1["nev"], ls_cpu=tls,
                         ops_decode=n * max(1, math.log2(P["maq1"])) + E + m * m, lp=sz))
    json.dump(recs, open(f"{H}/out/cx.json", "w"))

    def slope(xs, ys):
        A = np.vstack([np.log(xs), np.ones(len(xs))]).T; s, c = np.linalg.lstsq(A, np.log(ys), rcond=None)[0]
        res = np.log(ys) - (s * np.log(xs) + c); boot = []
        for _ in range(2000):
            i = rng.integers(len(xs), size=len(xs)); Ab = A[i]
            boot.append(np.linalg.lstsq(Ab, np.log(np.asarray(ys))[i], rcond=None)[0][0])
        return s, np.percentile(boot, [2.5, 97.5])
    n1 = np.array([x["n1"] for x in recs], float)
    for name, ys in [("t_decode", [x["t_decode"] for x in recs]), ("ops_decode", [x["ops_decode"] for x in recs]),
                     ("t_h2build", [x["t_h2build"] for x in recs]), ("ls_evals", [x["ls_evals"] for x in recs]),
                     ("ls_cpu", [x["ls_cpu"] for x in recs]), ("T0", [x["T0"] for x in recs]),
                     ("lp1_cols", [x["lp"]["1"]["cols"] for x in recs]), ("lp1_nnz", [x["lp"]["1"]["nnz"] for x in recs]),
                     ("lp2_cols", [x["lp"]["2"]["cols"] for x in recs]), ("lp2_nnz", [x["lp"]["2"]["nnz"] for x in recs]),
                     ("lp1_x_nnz", [x["lp"]["1"]["x_nnz"] for x in recs])]:
        s, ci = slope(n1, ys); print(f"{name:12s} slope {s:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
    for g in ("m2", "m10"):
        sel = [x for x in recs if x["g"] == g]; xs = np.array([x["n1"] for x in sel], float)
        for name in ("lp1_cols", "lp2_nnz", "ls_evals", "t_decode"):
            ys = [x["lp"]["1"]["cols"] if name == "lp1_cols" else x["lp"]["2"]["nnz"] if name == "lp2_nnz" else x[name] for x in sel]
            s, ci = slope(xs, ys); print(f"  {g:4s} {name:10s} slope {s:5.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")


if __name__ == "__main__":
    main()
