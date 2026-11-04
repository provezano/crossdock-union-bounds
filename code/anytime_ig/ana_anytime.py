"""Anytime profile analysis (run once; preregistered). Unit = TEST instance (70; one seed).
d_i = 100 (A_i - R_i) / R_i, negative = A better. Endpoints (15), Holm over all 15 at alpha 0.05:
  A_k  FB-ILS vs EACDC-R (restarts) at EACDC-R's evaluations at level k            one-sided (mean < 0)
  B_k  FB-ILS vs EACDCml-code (generations x k) at its evaluations                  one-sided (mean < 0)
  C_k  FB-ILS at k x nev(HgR) vs HgR (own stopping rule)                            one-sided (mean < 0)
  T_k  tuned IG vs EACDC-R at EACDC-R's evaluations                                  two-sided
  U_k  FB-ILS vs tuned IG, both at EACDC-R's evaluations                             two-sided
Test: sign-flip permutation, 100,000 flips (rng 0); CI95 percentile bootstrap over instances (10,000, rng 0).
Confirmed iff Holm-adjusted p < 0.05 and the CI excludes 0 (one-sided: upper < 0; two-sided: report the side).
usage: ana_anytime.py ANYDIR IGTDIR OUTPREFIX [N, smoke tests only]"""
import sys, json, glob, numpy as np, collections
KS = ("1", "10", "100")
def flip_p(d, two, rng):
    d = np.asarray(d, float); n = len(d); m = d.mean(); B = 100000; out = 0; done = 0
    while done < B:
        b = min(20000, B - done); S = rng.choice([-1.0, 1.0], size=(b, n)); mm = (S * np.abs(d)).mean(1)
        out += int((np.abs(mm) >= abs(m) - 1e-12).sum()) if two else int((mm <= m + 1e-12).sum()); done += b
    return (out + 1) / (B + 1)
def boot(d, rng):
    d = np.asarray(d, float); i = rng.integers(0, len(d), size=(10000, len(d))); m = d[i].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def wtl(d): d = np.asarray(d); return f"{int((d < -1e-9).sum())}/{int((np.abs(d) <= 1e-9).sum())}/{int((d > 1e-9).sum())}"
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj
def main():
    adir, tdir, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    A = {x["key"]: x for x in (json.load(open(f)) for f in sorted(glob.glob(f"{adir}/*.json")))}
    T = {x["key"]: x for x in (json.load(open(f)) for f in sorted(glob.glob(f"{tdir}/*.json")))}
    NEXP = int(sys.argv[4]) if len(sys.argv) > 4 else 70
    keys = sorted(A); assert len(keys) == NEXP and set(T) == set(keys), (len(keys), len(T))
    rel = lambda a, r: 100.0 * (a - r) / r
    E = collections.OrderedDict()
    for k in KS:
        E[f"A{k}"] = ([rel(A[i]["search"]["fbils"]["at"][f"eacdcR_{k}"], A[i]["eacdcR"]["levels"][k]["best"]) for i in keys], False)
    for k in KS:
        E[f"B{k}"] = ([rel(A[i]["search"]["fbils"]["at"][f"mlc_{k}"], A[i]["mlc"]["levels"][k]["best"]) for i in keys], False)
    for k in KS:
        E[f"C{k}"] = ([rel(A[i]["search"]["fbils"]["at"][f"hgr_{k}"], A[i]["hgr"]["hgr"]["ub"]) for i in keys], False)
    for k in KS:
        E[f"T{k}"] = ([rel(T[i]["at"][f"eacdcR_{k}"], A[i]["eacdcR"]["levels"][k]["best"]) for i in keys], True)
    for k in KS:
        E[f"U{k}"] = ([rel(A[i]["search"]["fbils"]["at"][f"eacdcR_{k}"], T[i]["at"][f"eacdcR_{k}"]) for i in keys], True)
    rng = np.random.default_rng(0); res = collections.OrderedDict()
    for name, (d, two) in E.items():
        lo, hi = boot(d, rng); res[name] = dict(n=len(d), mean=float(np.mean(d)), median=float(np.median(d)), ci=[lo, hi],
                                                 wtl=wtl(d), p=flip_p(d, two, rng), two_sided=two)
    adj = holm(np.array([r["p"] for r in res.values()]))
    for (name, r), a in zip(res.items(), adj):
        r["p_holm"] = float(a); lo, hi = r["ci"]
        if r["two_sided"]: r["verdict"] = ("CONFIRMED " + ("(A better)" if hi < 0 else "(A worse)")) if (a < 0.05 and (hi < 0 or lo > 0)) else "NOT CONFIRMED"
        else: r["verdict"] = "CONFIRMED" if (a < 0.05 and hi < 0) else "NOT CONFIRMED"
    # secondary (no claims)
    sec = collections.OrderedDict()
    for fam in "ABC":
        d = np.array(E[f"{fam}100"][0]) - np.array(E[f"{fam}1"][0]); lo, hi = boot(d, rng)
        sec[f"trend_{fam}_100minus1"] = dict(mean=float(d.mean()), ci=[lo, hi])
    for k in KS:
        d = [rel(A[i]["search"]["igdef"]["at"][f"eacdcR_{k}"], A[i]["eacdcR"]["levels"][k]["best"]) for i in keys]
        sec[f"igdef_vs_eacdcR_{k}"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)), wtl=wtl(d))
        d = [rel(T[i]["at"][f"eacdcR_{k}"], A[i]["search"]["igdef"]["at"][f"eacdcR_{k}"]) for i in keys]
        sec[f"igtuned_vs_igdef_{k}"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)), wtl=wtl(d))
        d = [rel(A[i]["search"]["fbils"]["at"][f"hgrnolp_{k}"], min(A[i]["hgr"]["hgr"]["ub"], A[i]["hgr"]["hgrnolp"]["ub"])) for i in keys]
        sec[f"fbils_vs_hgrbest_{k}"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)), wtl=wtl(d), note="at k x nev(HgR-noLP)")
        d = [rel(A[i]["eacdcR"]["levels"][k]["best"], A[i]["eacdcR"]["levels"]["1"]["best"]) for i in keys]
        sec[f"eacdcR_gain_{k}_vs_1"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)))
        d = [rel(A[i]["mlc"]["levels"][k]["best"], A[i]["mlc"]["levels"]["1"]["best"]) for i in keys]
        sec[f"mlc_gain_{k}_vs_1"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)))
        d = [rel(A[i]["search"]["fbils"]["at"][f"eacdcR_{k}"], A[i]["search"]["fbils"]["at"]["eacdcR_1"]) for i in keys]
        sec[f"fbils_gain_{k}_vs_1"] = dict(mean=float(np.mean(d)), ci=list(boot(d, rng)))
    # by n1 (pattern only)
    n1s = sorted({A[i]["n1"] for i in keys}); byn = {}
    for name in ("A1", "A100", "B1", "B100", "C1", "C100", "T1", "T100", "U1", "U100"):
        d = np.array(E[name][0]); byn[name] = {str(n): float(d[[A[i]["n1"] == n for i in keys]].mean()) for n in n1s}
    # best-known-relative profile (convergence figure data): mean RPD vs the best value any method found on the instance
    bk = {i: min([A[i]["eacdcR"]["levels"]["100"]["best"], A[i]["mlc"]["levels"]["100"]["best"], A[i]["hgr"]["hgr"]["ub"],
                  A[i]["hgr"]["hgrnolp"]["ub"], A[i]["search"]["fbils"]["best"], A[i]["search"]["igdef"]["best"], T[i]["best"]]) for i in keys}
    prof = {}
    for k in KS:
        prof[k] = dict(
            eacdcR=float(np.mean([rel(A[i]["eacdcR"]["levels"][k]["best"], bk[i]) for i in keys])),
            mlc=float(np.mean([rel(A[i]["mlc"]["levels"][k]["best"], bk[i]) for i in keys])),
            hgr=float(np.mean([rel(A[i]["hgr"]["hgr"]["ub"], bk[i]) for i in keys])),
            fbils_at_eacdcR=float(np.mean([rel(A[i]["search"]["fbils"]["at"][f"eacdcR_{k}"], bk[i]) for i in keys])),
            igt_at_eacdcR=float(np.mean([rel(T[i]["at"][f"eacdcR_{k}"], bk[i]) for i in keys])),
            igdef_at_eacdcR=float(np.mean([rel(A[i]["search"]["igdef"]["at"][f"eacdcR_{k}"], bk[i]) for i in keys])),
            fbils_at_mlc=float(np.mean([rel(A[i]["search"]["fbils"]["at"][f"mlc_{k}"], bk[i]) for i in keys])),
            fbils_at_hgr=float(np.mean([rel(A[i]["search"]["fbils"]["at"][f"hgr_{k}"], bk[i]) for i in keys])),
            evals_eacdcR=float(np.mean([A[i]["targets"][f"eacdcR_{k}"] for i in keys])),
            evals_mlc=float(np.mean([A[i]["targets"][f"mlc_{k}"] for i in keys])),
            evals_hgr=float(np.mean([A[i]["targets"][f"hgr_{k}"] for i in keys])),
            cpu_eacdcR=float(np.mean([A[i]["eacdcR"]["levels"][k]["cpu"] for i in keys])),
            cpu_mlc=float(np.mean([A[i]["mlc"]["levels"][k]["cpu"] for i in keys])),
            cpu_fbils_at_eacdcR=float(np.mean([A[i]["search"]["fbils"]["cpu_at"][f"eacdcR_{k}"] for i in keys])),
            cpu_igt_at_eacdcR=float(np.mean([T[i]["cpu_at"][f"eacdcR_{k}"] for i in keys])))
    integ = dict(
        capped=[i for i in keys if A[i]["search"]["fbils"]["capped"] or A[i]["search"]["igdef"]["capped"] or T[i]["capped"]],
        hgr_stop=dict(collections.Counter(A[i]["hgr"]["hgr"]["stop"] for i in keys)),
        hgr_cpu_median=float(np.median([A[i]["hgr"]["hgr"]["cpu"] for i in keys])),
        hosts=dict(collections.Counter(A[i]["env"]["host"] for i in keys)), igt_hosts=dict(collections.Counter(T[i]["env"]["host"] for i in keys)),
        hgr_hosts=dict(collections.Counter(A[i]["hgr"]["env"]["host"] for i in keys)),
        cpu_h=dict(anytime=float(sum(A[i]["eacdcR"]["levels"]["100"]["cpu"] + sum(A[i]["mlc"]["levels"][k]["cpu"] for k in KS)
                                     + A[i]["search"]["fbils"]["cpu"] + A[i]["search"]["igdef"]["cpu"] for i in keys) / 3600),
                   hgr=float(sum(A[i]["hgr"]["hgr"]["cpu"] + A[i]["hgr"]["hgrnolp"]["cpu"] for i in keys) / 3600),
                   igt=float(sum(T[i]["cpu"] for i in keys) / 3600)),
        igt_cfg=T[keys[0]]["cfg"])
    json.dump(dict(endpoints=res, secondary=sec, by_n1=byn, profile=prof, integrity=integ), open(outp + ".json", "w"), indent=1)
    with open(outp + ".txt", "w") as f:
        for name, r in res.items():
            f.write(f"{name:5s} mean {r['mean']:+.3f} CI [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] med {r['median']:+.3f} W/T/L {r['wtl']} "
                    f"p {r['p']:.2e} holm {r['p_holm']:.2e} {r['verdict']}\n")
        f.write(json.dumps(dict(secondary=sec, by_n1=byn, profile=prof, integrity=integ), indent=1))
    print(open(outp + ".txt").read()[:4000])
if __name__ == "__main__":
    main()
