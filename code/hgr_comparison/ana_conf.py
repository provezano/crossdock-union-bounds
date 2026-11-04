"""Analysis of the preregistered confirmation (run once, after all cells are complete).
Unit = instance; value d_i = mean over seeds 41-43 of 100*(ILS - HgR)/HgR. CI95 = percentile instance bootstrap
(10,000 resamples, rng seed 0). Test = one-sided Wilcoxon signed-rank (alternative 'less', zero_method 'wilcox').
Holm over H1-H4 at alpha 0.05. Confirmed iff Holm p < 0.05 and CI95 upper < 0. W/T/L on d_i with tol 1e-9.
usage: ana_conf.py  -> writes analysis.json and analysis.txt"""
import os, json, glob, collections
import numpy as np
from scipy.stats import wilcoxon
D = os.path.dirname(os.path.abspath(__file__)); SEEDS = (41, 42, 43)


def load(sub, methods, keys):
    R = collections.defaultdict(dict)
    for f in glob.glob(f"{D}/out/{sub}/*.json"):
        r = json.load(open(f)); R[r["key"]][r["seed"]] = r
    miss = [(k, s) for k in keys for s in SEEDS if s not in R.get(k, {})]
    assert not miss, (sub, len(miss), miss[:5])
    for k in keys:
        for s in SEEDS:
            for m in methods: assert R[k][s][m]["checked"], (k, s, m)
    return R


def d_inst(R, keys, a, b):
    out = []
    for k in keys:
        v = []
        for s in SEEDS:
            x = R[k][s]; ua = x[a]["ub"]
            ub = min(x["hgr"]["ub"], x["hgrnolp"]["ub"]) if b == "hgrbest" else x[b]["ub"]
            v.append(100.0 * (ua - ub) / ub)
        out.append(float(np.mean(v)))
    return np.array(out)


def summ(d):
    rng = np.random.default_rng(0); n = len(d)
    bs = d[rng.integers(0, n, size=(10000, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(wilcoxon(d, alternative="less", zero_method="wilcox").pvalue) if np.any(np.abs(d) > 1e-9) else 1.0
    tol = 1e-9
    return dict(n=n, mean=float(d.mean()), median=float(np.median(d)), ci=[float(lo), float(hi)], p=p,
                W=int((d < -tol).sum()), T=int((np.abs(d) <= tol).sum()), L=int((d > tol).sum()))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def main():
    L210 = json.load(open(f"{D}/list210.json")); L60 = json.load(open(f"{D}/list60.json"))
    k210 = [r["key"] for r in L210]; k60 = [r["key"] for r in L60]
    R10 = load("b10", ("hgr", "hgrnolp", "ils1", "ils2"), k210); R60 = load("b60", ("hgr", "ils2"), k60)
    H = dict(H1=summ(d_inst(R10, k210, "ils2", "hgr")), H2=summ(d_inst(R10, k210, "ils1", "hgr")),
             H3=summ(d_inst(R10, k210, "ils2", "hgrbest")), H4=summ(d_inst(R60, k60, "ils2", "hgr")))
    adj = holm(np.array([H[h]["p"] for h in ("H1", "H2", "H3", "H4")]))
    for h, a in zip(("H1", "H2", "H3", "H4"), adj):
        H[h]["holm_p"] = float(a); H[h]["confirmed"] = bool(a < 0.05 and H[h]["ci"][1] < 0)
    # secondary (no test claims)
    S = {}
    S["ils1_vs_hgrbest_10s"] = summ(d_inst(R10, k210, "ils1", "hgrbest"))
    S["ils2_vs_ils1_10s"] = summ(np.array([np.mean([100 * (R10[k][s]["ils2"]["ub"] - R10[k][s]["ils1"]["ub"])
                                                    / R10[k][s]["ils1"]["ub"] for s in SEEDS]) for k in k210]))
    S["hgr_vs_hgrnolp_10s"] = summ(np.array([np.mean([100 * (R10[k][s]["hgr"]["ub"] - R10[k][s]["hgrnolp"]["ub"])
                                                       / R10[k][s]["hgrnolp"]["ub"] for s in SEEDS]) for k in k210]))
    by = {}
    for grp, f in (("g", lambda r: r["g"]), ("n1", lambda r: r["n1"])):
        for v in sorted({f(r) for r in L210}, key=str):
            ks = [r["key"] for r in L210 if f(r) == v]
            by[f"{grp}={v}"] = {h: float(d_inst(R10, ks, a, b).mean()) for h, a, b in
                                (("H1", "ils2", "hgr"), ("H2", "ils1", "hgr"), ("H3", "ils2", "hgrbest"))}
    for v in sorted({r["n1"] for r in L60}):
        ks = [r["key"] for r in L60 if r["n1"] == v]; by[f"H4 n1={v} (n={len(ks)})"] = float(d_inst(R60, ks, "ils2", "hgr").mean())
    # integrity
    def cells(R, keys, ms): return [(R[k][s], m) for k in keys for s in SEEDS for m in ms]
    c10 = cells(R10, k210, ("hgr", "hgrnolp", "ils1", "ils2")); c60 = cells(R60, k60, ("hgr", "ils2"))
    hosts = sorted({x["env"]["host"] for x, _ in c10 + c60}); pys = sorted({x["env"]["python"] for x, _ in c10 + c60})
    integ = dict(n_runs=len(c10) + len(c60), all_checked=all(x[m]["checked"] for x, m in c10 + c60), hosts=hosts, python=pys,
                 cpu_h=sum(x[m]["cpu"] for x, m in c10 + c60) / 3600,
                 max_cpu_over_budget={b: max(x[m]["cpu"] - x["budget"] for x, m in c) for b, c in (("10", c10), ("60", c60))},
                 hgr_stop_10={s: sum(1 for x, m in c10 if m == "hgr" and x[m]["stop"] == s) for s in
                              {x[m]["stop"] for x, m in c10 if m == "hgr"}},
                 hgr_stop_60={s: sum(1 for x, m in c60 if m == "hgr" and x[m]["stop"] == s) for s in
                              {x[m]["stop"] for x, m in c60 if m == "hgr"}},
                 hgr_any_lp_ok_10=sum(1 for x, m in c10 if m == "hgr" and any(x[m]["lp_ok"])),
                 hgr_any_lp_ok_60=sum(1 for x, m in c60 if m == "hgr" and any(x[m]["lp_ok"])),
                 hgr_cpu_median_10=float(np.median([x["hgr"]["cpu"] for x, m in c10 if m == "hgr"])),
                 hgr_cpu_median_60=float(np.median([x["hgr"]["cpu"] for x, m in c60 if m == "hgr"])),
                 hgrnolp_cpu_median_10=float(np.median([x["hgrnolp"]["cpu"] for x, m in c10 if m == "hgrnolp"])),
                 hgr_det_spread_10=sum(1 for k in k210 if len({R10[k][s]["hgr"]["ub"] for s in SEEDS}) > 1),
                 first_method_10=dict(collections.Counter(R10[k][s]["order"][0] for k in k210 for s in SEEDS)),
                 first_method_60=dict(collections.Counter(R60[k][s]["order"][0] for k in k60 for s in SEEDS)))
    out = dict(confirmatory=H, secondary=S, by=by, integrity=integ)
    json.dump(out, open(f"{D}/analysis.json", "w"), indent=1)
    lines = ["endpoint | n | mean % | CI95 | median | W/T/L | p (raw) | Holm p | confirmed"]
    for h in ("H1", "H2", "H3", "H4"):
        x = H[h]; lines.append(f"{h} | {x['n']} | {x['mean']:+.3f} | [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {x['median']:+.3f} | "
                               f"{x['W']}/{x['T']}/{x['L']} | {x['p']:.2e} | {x['holm_p']:.2e} | {x['confirmed']}")
    for k, x in S.items():
        lines.append(f"[sec] {k} | {x['n']} | {x['mean']:+.3f} | [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {x['W']}/{x['T']}/{x['L']}")
    lines += [f"[by] {k}: {v}" for k, v in by.items()] + [f"[integrity] {k}: {v}" for k, v in integ.items()]
    open(f"{D}/analysis.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
