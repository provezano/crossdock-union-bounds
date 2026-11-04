"""Analyses for step 1 (exact), step 2a (2-dock set), step 5 (CP-SAT vs ILS). usage: ana_g5.py {exact DIR TL | dock DIR | ub DIR}"""
import sys, os, json, glob
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
import numpy as np
from ana_lib import boot, wtl


def pct(x): return f"{100 * x:.2f}%"


def exact(d, tl):
    B = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob(f"{d}/*_base.json")}
    arms = sorted({os.path.basename(f).split("_")[3] for f in glob.glob(f"{d}/*_tl{tl}.json")})
    A = {a: {} for a in arms}
    for f in glob.glob(f"{d}/*_tl{tl}.json"):
        r = json.load(open(f)); A[r["arm"]][r["key"]] = r
    print(f"instances with base: {len(B)}; arms: { {a: len(v) for a, v in A.items()} }")
    prim = "v2lbu" if "v2lbu" in A else arms[0]
    R = A[prim]; keys = sorted(k for k in R if k in B)
    # best known UB / best proven LB over all arms
    def best_hi(k): return min([B[k]["ub_ils10"], B[k]["ub_hgr"], B[k]["ub_eacdc"]] + [A[a][k]["hi"] for a in A if k in A[a] and A[a][k]["hi"] is not None])
    def best_lo(k): return max([B[k]["lbu"]] + [A[a][k]["lo"] for a in A if k in A[a] and A[a][k]["lo"] is not None])
    opt = {k: best_hi(k) for k in keys if best_lo(k) >= best_hi(k) - 1e-9}
    print(f"\n== primary arm {prim}, tl {tl}: n = {len(keys)}; solved (by any arm) {len(opt)} = {pct(len(opt) / len(keys))}; "
          f"solved by primary {sum(R[k]['opt'] is not None for k in keys)}")
    for n1 in (20, 30):
        ks = [k for k in keys if B[k]["n1"] == n1]
        print(f"  n1={n1}: n {len(ks)}, solved {sum(k in opt for k in ks)}, by group: " +
              ", ".join(f"{g} {sum(k in opt for k in ks if B[k]['g'] == g)}/{sum(B[k]['g'] == g for k in ks)}" for g in ("m2", "m4", "m10", "u24", "u210")))
    ks = sorted(opt)
    if ks:
        eq = [B[k]["lbu"] >= opt[k] - 1e-9 for k in ks]
        print(f"\nLB* (max fwd, rev) = OPT on {sum(eq)}/{len(ks)} solved = {pct(np.mean(eq))}; "
              f"LB_U* fwd = OPT on {sum(B[k]['lb_u1'] >= opt[k] - 1e-9 for k in ks)}")
        g = np.array([(opt[k] - B[k]["lbu"]) / opt[k] for k in ks])
        print(f"LB* gap to OPT (OPT-LB)/OPT on solved: mean {pct(g.mean())}, median {pct(np.median(g))}, max {pct(g.max())}")
        gcs = np.array([(opt[k] - np.ceil(B[k]["lb_cs"] - 1e-9)) / opt[k] for k in ks])
        print(f"Chen-Song LB gap to OPT on solved: mean {pct(gcs.mean())}")
        for lab, col in (("HgR", "ub_hgr"), ("EACDC (1 default run)", "ub_eacdc"), ("ILS 4 s", "ub_ils4"), ("ILS 10 s", "ub_ils10")):
            x = np.array([(B[k][col] - opt[k]) / opt[k] for k in ks]); m, lo, hi = boot(x)
            print(f"  true gap {lab:22s} mean {pct(m)} [{pct(lo)}, {pct(hi)}], median {pct(np.median(x))}, max {pct(x.max())}, at OPT {np.mean(x < 1e-9):.3f}")
        for n1 in (20, 30):
            kk = [k for k in ks if B[k]["n1"] == n1]
            if kk:
                print(f"  n1={n1} (solved {len(kk)}): LB*=OPT {np.mean([B[k]['lbu'] >= opt[k] - 1e-9 for k in kk]):.3f}; "
                      + "; ".join(f"{lab} {pct(np.mean([(B[k][c] - opt[k]) / opt[k] for k in kk]))}" for lab, c in (("HgR", "ub_hgr"), ("EACDC", "ub_eacdc"), ("ILS10", "ub_ils10"))))
    # unsolved: remaining certified gap
    un = [k for k in keys if k not in opt]
    if un:
        gu = np.array([(best_hi(k) - best_lo(k)) / best_hi(k) for k in un])
        raise_lb = np.array([(best_lo(k) - B[k]["lbu"]) / B[k]["lbu"] for k in un])
        print(f"\nunsolved {len(un)}: residual (bestUB - bestLB)/bestUB mean {pct(gu.mean())}, max {pct(gu.max())}; "
              f"exact method raised LB* by mean {pct(raise_lb.mean())}")
    # LB raise from the exact method on ALL instances, and decomposition of the ILS-LB* gap
    tot = np.array([(B[k]["ub_ils10"] - B[k]["lbu"]) / B[k]["lbu"] for k in keys])
    print(f"\nILS10 vs LB*: mean gap (UB-LB)/LB {pct(tot.mean())}")
    if ks:
        lbpart = np.array([(opt[k] - B[k]["lbu"]) / B[k]["lbu"] for k in ks]); ubpart = np.array([(B[k]["ub_ils10"] - opt[k]) / B[k]["lbu"] for k in ks])
        print(f"  on solved: LB structure share {pct(lbpart.mean())} + UB slack {pct(ubpart.mean())}")
    # with vs without LB_U*: matched pairs on instances where both arms ran
    print("\n== time to prove optimality, paired subsets")
    pairs = [("v2lbu", "v2cs"), ("v2lbu", "onone"), ("olbu", "onone"), ("hlbu", "hcs"), ("v2lbu", "olbu")]
    for a, b in pairs:
        if a not in A or b not in A: continue
        kk = sorted(set(A[a]) & set(A[b]) & set(B))
        if not kk: continue
        sa = [A[a][k]["opt"] is not None for k in kk]; sb = [A[b][k]["opt"] is not None for k in kk]
        ta = np.array([A[a][k]["t_opt"] if A[a][k]["opt"] is not None else float(tl) for k in kk])
        tb = np.array([A[b][k]["t_opt"] if A[b][k]["opt"] is not None else float(tl) for k in kk])
        both = [i for i in range(len(kk)) if sa[i] and sb[i]]
        print(f"  {a} vs {b}: n {len(kk)}; solved {sum(sa)} vs {sum(sb)}; only-{a} {sum(x and not y for x, y in zip(sa, sb))}, only-{b} {sum(y and not x for x, y in zip(sa, sb))}; "
              f"censored mean time {ta.mean():.1f} vs {tb.mean():.1f} s; median {np.median(ta):.1f} vs {np.median(tb):.1f}; "
              f"on both-solved ({len(both)}): median {np.median(ta[both]) if both else float('nan'):.1f} vs {np.median(tb[both]) if both else float('nan'):.1f} s, "
              f"geo-mean ratio {np.exp(np.mean(np.log(np.maximum(ta[both], .01) / np.maximum(tb[both], .01)))) if both else float('nan'):.2f}")
        la = np.array([A[a][k]["lo"] for k in kk], float); lb_ = np.array([A[b][k]["lo"] for k in kk], float)
        print(f"     final proven LB: {a} > {b} in {np.mean(la > lb_ + 1e-9):.2f}, < in {np.mean(la < lb_ - 1e-9):.2f}")


def dock(d):
    R = [json.load(open(f)) for f in glob.glob(f"{d}/*.json")]
    print(f"n = {len(R)}")
    T4 = {("Grupo1", 5): (23.7, 29.3, 33.6, 34.7), ("Grupo1", 10): (39.4, 60.9, 66.2, 70.0), ("Grupo1", 20): (70.7, 119.7, 140.2, 143.4),
          ("Grupo1", 40): (132.9, 225.5, 293.5, 294.9), ("Grupo1", 60): (193.9, 338.0, 444.3, 444.3),
          ("Grupo2", 5): (232.8, 288.9, 337.4, 347.0), ("Grupo2", 10): (379.1, 635.9, 667.6, 706.9), ("Grupo2", 20): (705.7, 1203.0, 1412.7, 1442.4),
          ("Grupo2", 40): (None, 2253.2, 2871.8, 2883.2), ("Grupo2", 60): (None, 3373.5, 4452.3, 4452.3)}
    print("group n | ours: LB1 LB2 | paper T4: LB1 LB2 LB_HgR | ours: LB_U* fwd, max(fwd,rev), LB* (incl LB1/2) | LB*/LB_HgR-1 | exact fwd/rev | ILS4 gap (UB-LB*)/UB | n with OPT (n<=10): LB*=OPT")
    for (gname, n), t in sorted(T4.items()):
        rr = [r for r in R if r["key"].startswith(gname) and r["n"] == n]
        if not rr: continue
        l1 = np.mean([r["lb1"] for r in rr]); l2 = np.mean([r["lb2"] for r in rr])
        u = np.mean([r["lb_u1"] for r in rr]); ur = np.mean([max(r["lb_u1"], r["lb_rev"]) for r in rr])
        ls = [max(r["lb_u1"], r["lb_rev"], r["lb1"], r["lb2"]) for r in rr]
        gap = np.mean([(r["ub_ils4"] - L) / r["ub_ils4"] for r, L in zip(rr, ls)])
        ex = f"{np.mean([r['u1_exact'] for r in rr]):.2f}/{np.mean([r['rev_exact'] for r in rr]):.2f}"
        wo = [r for r in rr if r.get("opt") is not None]
        eq = f"{sum(max(r['lb_u1'], r['lb_rev'], r['lb1'], r['lb2']) >= r['opt'] - 1e-9 for r in wo)}/{len(wo)}" if n <= 10 else "-"
        print(f"{gname} {n:2d} (k={len(rr)}) | {l1:.1f} {l2:.1f} | {t[1]} {t[2]} {t[3]} | {u:.1f} {ur:.1f} {np.mean(ls):.1f} | {100 * (np.mean(ls) / t[3] - 1):+.2f}% | {ex} | {pct(gap)} | {eq}")
    bad = [r["key"] for r in R if max(r["lb_u1"], r["lb_rev"]) > r["ub_ils4"] + 1e-9]
    print("validity: LB > ILS UB in", len(bad), bad[:5])
    print("every inbound has a successor:", np.mean([r["every_in_has_succ"] for r in R]))
    print("instance gain of LB* over max(LB1, LB2): mean", pct(np.mean([max(r['lb_u1'], r['lb_rev'], r['lb1'], r['lb2']) / max(r['lb1'], r['lb2']) - 1 for r in R])))


def ub(d):
    R = [json.load(open(f)) for f in glob.glob(f"{d}/*.json")]
    keys = sorted({r["key"] for r in R}); print(f"runs {len(R)}, instances {len(keys)}")
    for b in (4, 10):
        for arm in ("cpc", "cpw"):
            rows = []; miss = 0
            for k in keys:
                rr = [r for r in R if r["key"] == k]
                xi = [r["ils"][f"at{b}"] for r in rr]; xc = [r[arm][f"at{b}"] for r in rr]
                if any(v is None for v in xc): miss += 1; continue
                rows.append((np.mean(xc) / np.mean(xi) - 1, rr[0]["n1"], np.mean(xc), np.mean(xi)))
            x = np.array([r[0] for r in rows]); m, lo, hi = boot(x); w, t, l = wtl([r[2] for r in rows], [r[3] for r in rows])
            print(f"{arm} vs ILS at {b} s: CP-SAT/ILS - 1 = {pct(m)} [{pct(lo)}, {pct(hi)}], CP-SAT W/T/L {w}/{t}/{l}, n={len(rows)}, no-solution instances {miss}")
            print("   by n1: " + ", ".join(f"{n1}: {pct(np.mean([r[0] for r in rows if r[1] == n1]))} (W/T/L {'/'.join(map(str, wtl([r[2] for r in rows if r[1] == n1], [r[3] for r in rows if r[1] == n1])))})" for n1 in range(20, 81, 10) if any(r[1] == n1 for r in rows)))
    opt = sum(any(r[a]["status"] == "OPTIMAL" for a in ("cpc", "cpw")) for r in R)
    print("CP-SAT runs that proved optimality within 10 s:", opt, "of", len(R))


def erana(fn):
    R = [json.loads(l) for l in open(fn)]; print(f"n = {len(R)}")
    def row(rr, lab):
        L = np.array([max(r["lb_u1"], r["lb_rev"]) for r in rr]); E = np.array([r["lb_er"] for r in rr]); C = np.array([np.ceil(r["lb_cs"] - 1e-9) for r in rr])
        print(f"{lab:10s} n={len(rr):5d}  ER/ceil(CS)-1 {pct(np.mean(E / C - 1))}  LB*/ER-1 {pct(np.mean(L / E - 1))}  ER>LB* {np.mean(E > L + 1e-9):.3f}  ER=LB* {np.mean(np.abs(E - L) < 1e-9):.3f}  "
              f"max(LB*,ER)/LB*-1 {pct(np.mean(np.maximum(L, E) / L - 1))}  ER cpu mean {np.mean([r['cpu'] for r in rr]):.3f}s")
    row(R, "all")
    for n1 in range(20, 81, 10): row([r for r in R if r["n1"] == n1], f"n1={n1}")
    for g in ("m2", "m4", "m10", "u24", "u210"): row([r for r in R if r["g"] == g], g)
    row([r for r in R if r["g"] == "m10" and r["n1"] == 20], "m10,n1=20")


def det(fa, fb=None):
    A = json.load(open(fa)); rows = A["rows"]; print(f"{A['host']} py {A['py']} n={len(rows)}")
    d = np.array([r["lb_det"] for r in rows]); c4 = np.array([r["lb_cpu4"] for r in rows])
    print(f"det (0.5 det-s, 50 steps) vs CPU-capped 4 s: mean det/cpu4-1 {pct(np.mean(d / c4 - 1))}; det > cpu4 {np.mean(d > c4):.2f}, < {np.mean(d < c4):.2f}; "
          f"exact det {np.mean([r['exact_det'] for r in rows]):.2f} vs cpu4 {np.mean([r['exact_cpu4'] for r in rows]):.2f}; cpu det mean {np.mean([r['cpu'] for r in rows]):.2f}s (max {max(r['cpu'] for r in rows):.1f}) vs cpu4 {np.mean([r['cpu4'] for r in rows]):.2f}s")
    for n1 in (60, 70, 80):
        rr = [r for r in rows if r["key"].split("_")[1] == str(n1)]
        print(f"  n1={n1}: exact det {np.mean([r['exact_det'] for r in rr]):.2f}, cpu4 {np.mean([r['exact_cpu4'] for r in rr]):.2f}; det/cpu4-1 {pct(np.mean([r['lb_det'] / r['lb_cpu4'] - 1 for r in rr]))}")
    if fb:
        B = json.load(open(fb)); mb = {r["key"]: r for r in B["rows"]}
        same = [r["lb_det"] == mb[r["key"]]["lb_det"] and r["steps"] == mb[r["key"]]["steps"] for r in rows if r["key"] in mb]
        print(f"cross-machine ({A['host']} vs {B['host']}): identical value and steps on {sum(same)}/{len(same)}")


if __name__ == "__main__":
    {"exact": lambda: exact(sys.argv[2], sys.argv[3]), "dock": lambda: dock(sys.argv[2]), "ub": lambda: ub(sys.argv[2]),
     "er": lambda: erana(sys.argv[2]), "det": lambda: det(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)}[sys.argv[1]]()
