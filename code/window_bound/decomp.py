"""Phase 1: headroom decomposition on the solved optima (read from exact_cpsat/fbils_census_exact_cpuc outputs, read-only).
Per instance: a ladder of VALID destructive bounds (each one adds an ingredient) and ORACLE bounds that use the
heads/tails of one optimal schedule S*. Every value is checked <= OPT (assert). usage: decomp.py OUT SHARD NSH"""
import sys; sys.dont_write_bytecode = True
import os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset

def opt_schedule(I, v, tl=240.0):
    if v["opt_sched"] is not None: return v["opt_sched"], "v3"
    sys.path.insert(0, f"{D}/snap"); sys.path.insert(0, f"{D}/../exact_cpsat"); import g6cp
    x = g6cp.exact_v3(I, v["opt"], v["ub_ils10"], (v["s1"], v["s2"]), tl)
    if x["sched"] is not None and int(x["hi"]) == int(v["opt"]): return [list(map(int, x["sched"][0])), list(map(int, x["sched"][1]))], "resolved"
    return None, "fail"

def port_refute(I, T, cfg, h, q, cp, det, st=None):
    for K, mode in ((8, "score"), (8, "mix"), (16, "uniform"), (12, "score")):
        if lbx.refute_nested(I, T, cfg, h, q, cuts=cp.pick(T, K, mode), det=det, stats=st): return True
    return False

def single_cuts(cp, T):
    return sorted(set(cp.pick(T, 48, "score")) | set(cp.pick(T, 16, "uniform")))

def run_small(I, v, h, q, lo, hi, budget=60.0):
    cp = lbx.CutPicker(I, h, q); out = {}
    for name, f in (("J_dff", lambda T: lbx.refute_single(I, T, lbx.Cfg(), h, q, cuts=single_cuts(cp, T))),
                    ("N8s", lambda T: lbx.refute_nested(I, T, lbx.Cfg(), h, q, cuts=cp.pick(T, 8, "score"), det=3.0))):
        lb, n, cpu = lbx.destructive(f, lo, hi, budget=budget); out[name] = dict(lb=lb, n=n, cpu=round(cpu, 2))
    return out

def run(I, v, h, q, lo, hi, budget=120.0, tag=""):
    cp = lbx.CutPicker(I, h, q); out = {}
    def rec(name, f, lo_=lo):
        t = time.process_time(); lb, n, cpu = lbx.destructive(f, lo_, hi, budget=budget); out[name] = dict(lb=lb, n=n, cpu=round(cpu, 2))
    rec("J_fluid", lambda T: lbx.refute_single(I, T, lbx.Cfg(s1dff=False, s2dff=False, partial=False), h, q, cuts=single_cuts(cp, T)))
    rec("J_pw", lambda T: lbx.refute_single(I, T, lbx.Cfg(s1dff=False, s2dff=False), h, q, cuts=single_cuts(cp, T)))
    rec("J_dff", lambda T: lbx.refute_single(I, T, lbx.Cfg(), h, q, cuts=single_cuts(cp, T)))
    rec("J_exact", lambda T: lbx.refute_single(I, T, lbx.Cfg(exact_s1=True, exact_s2=True), h, q, cuts=single_cuts(cp, T), det=2.0))
    rec("N8s", lambda T: lbx.refute_nested(I, T, lbx.Cfg(), h, q, cuts=cp.pick(T, 8, "score"), det=3.0))
    rec("N_port", lambda T: port_refute(I, T, lbx.Cfg(), h, q, cp, 3.0), lo_=max(lo, out["N8s"]["lb"]))
    rec("N8s_exact", lambda T: lbx.refute_nested(I, T, lbx.Cfg(exact_s1=True, exact_s2=True), h, q, cuts=cp.pick(T, 8, "score"), det=6.0))
    rec("N8s_nodff", lambda T: lbx.refute_nested(I, T, lbx.Cfg(s1dff=False, s2dff=False), h, q, cuts=cp.pick(T, 8, "score"), det=3.0))
    return out

def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    O = sset.load(); keys = sorted(k for k, v in O.items() if v["opt"] is not None)
    for qd, k in enumerate(keys):
        if qd % ns != sh: continue
        fo = f"{out}/{k}.json"
        if os.path.exists(fo): continue
        v = O[k]; I = lbx.parse(sset.path(v["g"], v["n1"], v["seed"])); opt = int(v["opt"]); t0 = time.process_time()
        h, q = lbx.heads_tails_bp(I); hb, qb = lbx.heads_tails_basic(I)
        triv = int(max(-(-I["p1"].sum() // I["m1"]), -(-I["p2"].sum() // I["m2"])))
        lo, hi = int(v["lbu"]), int(v["best_hi"])
        r = dict(key=k, g=v["g"], n1=v["n1"], n2=v["n2"], m1=I["m1"], m2=I["m2"], opt=opt, lbstar=lo, hi=hi,
                 heads_bp_gain=int((h > hb).sum()), tails_bp_gain=int((q > qb).sum()), py=platform.python_version())
        r["pre"] = lbx.destructive(lambda T: lbx.refute_pmtn(I, T, h, q), triv, hi)[0]
        r["s2np"] = lbx.stage_np_lb(I["p2"], h, I["m2"], tl=10)[0]; r["s1np"] = lbx.stage_np_lb(I["p1"], q, I["m1"], tl=10)[0]
        r["s2pmtn"] = lbx.stage_pmtn_lb(I["p2"], h, I["m2"]); r["s1pmtn"] = lbx.stage_pmtn_lb(I["p1"], q, I["m1"])
        r["valid"] = run(I, v, h, q, lo, hi)
        S, src = opt_schedule(I, v); r["sched_src"] = src
        if S is not None:
            s1, s2 = np.array(S[0]), np.array(S[1]); C1 = s1 + I["p1"]; P = I["prec"]
            assert abs(lbx_check(I, s1, s2) - opt) < 1e-6
            rs = np.where(P, C1[None, :], 0).max(axis=1).astype(np.int64)
            qs = np.array([opt - s2[P[:, i]].min() if P[:, i].any() else 0 for i in range(I["n1"])], dtype=np.int64)
            e2 = max(float(rs[j] + I["p2"][rs >= rs[j]].sum() / I["m2"]) for j in range(I["n2"]))
            r["oracle"] = dict(pre2=lbx.stage_pmtn_lb(I["p2"], rs, I["m2"]), pre1=lbx.stage_pmtn_lb(I["p1"], qs, I["m1"]), e2=e2,
                               np2=lbx.stage_np_lb(I["p2"], rs, I["m2"], tl=10)[0])
            r["oracle"]["nested"] = run(I, v, rs, qs, lo, hi, budget=90.0)
            r["oracle"]["heads_only"] = run_small(I, v, np.maximum(rs, h), q, lo, hi)
            r["oracle"]["tails_only"] = run_small(I, v, h, np.maximum(qs, q), lo, hi)
            for nm, x in list(r["oracle"]["nested"].items()) + list(r["oracle"]["heads_only"].items()) + list(r["oracle"]["tails_only"].items()):
                assert x["lb"] <= opt, ("ORACLE>OPT", k, nm, x["lb"], opt)
        r["cpu"] = round(time.process_time() - t0, 1)
        for name, x in list(r["valid"].items()): assert x["lb"] <= opt, ("VALIDITY", k, name, x["lb"], opt)
        for name in ("pre", "s2np", "s1np", "s2pmtn", "s1pmtn"): assert r[name] <= opt, ("VALIDITY", k, name, r[name], opt)
        json.dump(r, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo); print(k, r["cpu"], flush=True)
    print("done", flush=True)

def lbx_check(I, s1, s2):
    p1, p2 = I["p1"], I["p2"]
    for s, p, m in ((s1, p1, I["m1"]), (s2, p2, I["m2"])):
        ev = sorted([(t, 1) for t in s] + [(t, -1) for t in s + p], key=lambda e: (e[0], e[1])); c = 0
        for _, d in ev: c += d; assert c <= m
    e1 = s1 + p1
    for j in range(I["n2"]):
        pr = np.flatnonzero(I["prec"][j])
        if len(pr): assert s2[j] >= e1[pr].max()
    return float(max(e1.max(), (s2 + p2).max()))

if __name__ == "__main__":
    main()
