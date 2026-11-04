"""Analysis (run once) of run_strong.py outputs. usage: ana_strong.py DIR [DIR ...]"""
import json, glob, sys
import numpy as np


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float); m = np.random.default_rng(seed).choice(x, (B, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


for d in sys.argv[1:]:
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json"))]
    print(f"== {d}: n={len(R)}")
    if not R: continue
    u = np.array([r["lb_u1"] for r in R], float)
    ref = np.array([r.get("opt", r.get("ub_hgr")) for r in R], float)
    for v in R[0]["res"]:
        lbs = [r["res"][v]["lb"] for r in R]; ok = np.array([x is not None for x in lbs])
        lb = np.array([np.ceil(x - 1e-6) if x is not None else np.nan for x in lbs])
        raw = np.array([x if x is not None else np.nan for x in lbs])
        cpu = np.array([r["res"][v]["cpu"] for r in R])
        s = ok
        print(f"  [{v}] solved {s.sum()}/{len(R)}")
        if s.sum() == 0: continue
        print("    LP > UPB: %d/%d   LP < UPB: %d   LP == UPB: %d" % ((lb[s] > u[s] + 1e-9).sum(), s.sum(),
              (lb[s] < u[s] - 1e-9).sum(), (lb[s] == u[s]).sum()))
        print("    LP/UPB mean %.4f [%.4f, %.4f]" % boot(lb[s] / u[s]))
        print("    LP/ref mean %.4f [%.4f, %.4f]  (ref = OPT or HgR UB)" % boot(lb[s] / ref[s]))
        viol = raw[s] > ref[s] + 1e-6
        print("    VALIDITY: raw LP > ref on %d" % viol.sum())
        print("    CPU s mean %.2f median %.2f max %.1f" % (cpu.mean(), np.median(cpu), cpu.max()))
        if v != "base":
            b = np.array([r["res"]["base"]["lb"] if r["res"]["base"]["lb"] is not None else np.nan for r in R])
            g = s & ~np.isnan(b)
            print("    %s/base mean %.4f; strictly higher on %d/%d" % (v, np.mean(raw[g] / b[g]), (raw[g] > b[g] + 1e-6).sum(), g.sum()))
            for k in ("rounds", "cuts_B", "cuts_C"):
                print(f"    {k} mean {np.mean([r['res'][v][k] for r in R]):.1f}", end="")
            print()
    if "old_tim" in R[0]:
        b = [r["res"]["base"]["lb"] for r in R]
        dif = max(abs(x - r["old_tim"]) for x, r in zip(b, R) if x is not None and r["old_tim"] is not None)
        print(f"  base vs eacdc_matched_tim recorded TIM-LP: max |diff| = {dif:.2e}")
    for r in R:
        if "g" in r and r["n1"] > 20:
            print("   ", r["g"], r["n1"], r["seed"], "UPB", r["lb_u1"],
                  {v: (None if r["res"][v]["lb"] is None else round(r["res"][v]["lb"], 1), round(r["res"][v]["cpu"], 1), r["res"][v]["status"]) for v in r["res"]})
    if "n1" in R[0] and "mt" in R[0]:
        for n1 in sorted({r["n1"] for r in R}):
            ix = [i for i, r in enumerate(R) if r["n1"] == n1]
            a = np.array([np.ceil(R[i]["res"]["ABC"]["lb"] - 1e-6) for i in ix])
            print(f"   n1={n1}: ABC/UPB {np.mean(a / u[ix]):.3f}  ABC>UPB {(a > u[ix]).sum()}  ABC/OPT {np.mean(a / ref[ix]):.3f}")
