"""Faithfulness analysis: our group-mean gaps (UB - LB_PCH)/UB vs the paper's Table 1/3 group averages.
Reproduction error = mean over the 35 cells of (ours - paper), CI95 by stratified instance bootstrap (within cell).
usage: ana_repro.py OUTDIR [SEEDS(comma)]"""
import sys, os, json, glob, re
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
# plain-text export of the EACDCml paper (Nogueira et al. 2024); not redistributed, set EACDCML_PAPER_TXT
TXT = os.environ.get("EACDCML_PAPER_TXT", f"{D}/eacdcml_paper.txt")
GROUPS = ["m2", "m4", "m10", "u24", "u210"]


def paper_table():
    rows = [l.split() for l in open(TXT, encoding="utf-8").read().splitlines()
            if re.match(r"^[2-8]0 \[\d+,\d+\] ", l)]
    assert len(rows) == 35
    T = {}
    for k, r in enumerate(rows):
        v = [float(x) for x in r[2:]]; assert len(v) == 23
        T[(GROUPS[k // 7], int(r[0]))] = dict(eacdc=v[12], eacdc_T=v[14], ml=v[20], ml_T=v[22], hgr=v[10], pch=v[8])
    return T


def main():
    out = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/*.json"))]
    if seeds: R = [r for r in R if r["seed"] in seeds]
    T = paper_table(); cells = sorted({(r["g"], r["n1"]) for r in R}, key=lambda c: (GROUPS.index(c[0]), c[1]))
    by = {c: [r for r in R if (r["g"], r["n1"]) == c] for c in cells}
    rng = np.random.default_rng(0); B = 10000
    def strat(fn):
        """point and CI of mean over cells of mean over cell instances of fn(r)."""
        vals = {c: np.array([fn(r) for r in by[c]]) for c in cells}
        pt = np.mean([v.mean() for v in vals.values()])
        bs = np.zeros(B)
        for c, v in vals.items(): bs += v[rng.integers(0, len(v), (B, len(v)))].mean(1)
        bs /= len(cells); return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    res = dict(n_inst=len(R), n_cells=len(cells))
    for arm, pk in (("ml", "ml"), ("static", "eacdc")):
        ours = strat(lambda r: 100 * r[arm]["gap"])
        paper = np.mean([T[c][pk] for c in cells])
        err = (ours[0] - paper, ours[1] - paper, ours[2] - paper)
        res[arm] = dict(ours_mean=ours, paper_mean=paper, repro_error_pp=err,
                        cell_abs_err_mean=float(np.mean([abs(100 * np.mean([r[arm]["gap"] for r in by[c]]) - T[c][pk]) for c in cells])))
    res["ml_minus_static_pp"] = strat(lambda r: 100 * (r["ml"]["gap"] - r["static"]["gap"]))
    res["paper_ml_minus_static_pp"] = (-0.27, -0.2958, -0.2470)
    res["ml_minus_static_makespan_pct"] = strat(lambda r: 100 * (r["ml"]["best"] - r["static"]["best"]) / r["static"]["best"])
    w = sum(r["ml"]["best"] < r["static"]["best"] for r in R); l = sum(r["ml"]["best"] > r["static"]["best"] for r in R)
    res["ml_vs_static_WTL"] = (w, len(R) - w - l, l)
    res["by_n1"] = {}
    for n1 in sorted({c[1] for c in cells}):
        cs = [c for c in cells if c[1] == n1]
        res["by_n1"][n1] = dict(
            ours_ml=float(np.mean([100 * r["ml"]["gap"] for c in cs for r in by[c]])),
            paper_ml=float(np.mean([T[c]["ml"] for c in cs])),
            ours_st=float(np.mean([100 * r["static"]["gap"] for c in cs for r in by[c]])),
            paper_st=float(np.mean([T[c]["eacdc"] for c in cs])),
            cpu_ml=float(np.mean([r["ml"]["cpu"] for c in cs for r in by[c]])),
            cpu_st=float(np.mean([r["static"]["cpu"] for c in cs for r in by[c]])),
            paper_T_ml=float(np.mean([T[c]["ml_T"] for c in cs])), paper_T_st=float(np.mean([T[c]["eacdc_T"] for c in cs])))
    res["by_group"] = {g: dict(ours_ml=float(np.mean([100 * r["ml"]["gap"] for c in cells if c[0] == g for r in by[c]])),
                               paper_ml=float(np.mean([T[c]["ml"] for c in cells if c[0] == g])),
                               ours_st=float(np.mean([100 * r["static"]["gap"] for c in cells if c[0] == g for r in by[c]])),
                               paper_st=float(np.mean([T[c]["eacdc"] for c in cells if c[0] == g]))) for g in GROUPS}
    res["diag"] = dict(
        stop_ml={s: sum(r["ml"]["stop"] == s for r in R) for s in ("NGEN", "noimp", "cap")},
        stop_st={s: sum(r["static"]["stop"] == s for r in R) for s in ("NGEN", "noimp", "cap")},
        random_fill_frac_ml=float(np.mean([r["ml"]["init"]["n_random"] / r["ml"]["params"]["mu"] for r in R])),
        random_fill_frac_st=float(np.mean([r["static"]["init"]["n_random"] / 75 for r in R])),
        any_random_fill_st=float(np.mean([r["static"]["init"]["n_random"] > 0 for r in R])),
        best_ever_below_alg1_out_ml=float(np.mean([r["ml"]["best_ever"] < r["ml"]["best"] for r in R])),
        params_ml_mean={k: float(np.mean([r["ml"]["params"][k] for r in R])) for k in ("NGEN", "CXPB", "mu", "lam", "ts")},
        cpu_ml_mean=float(np.mean([r["ml"]["cpu"] for r in R])), cpu_st_mean=float(np.mean([r["static"]["cpu"] for r in R])),
        evals_ml_mean=float(np.mean([r["ml"]["evals"] for r in R])), evals_st_mean=float(np.mean([r["static"]["evals"] for r in R])))
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(f"{out.rstrip('/')}_analysis.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
