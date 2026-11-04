"""WIB experiments. usage: wib_run.py SET OUT SHARD NSH VERSIONS [BUDGET] [ONLY_KEYS_REGEX]
SET = solved (98 proven optima, lo = LB*, hi = best UB, assert LB <= OPT) | e105 (dev sample105, assert LB <= best UB).
VERSIONS = comma list of names in VERS. Per (instance, version): destructive search from LB* with a CPU budget.
One JSON per instance (merged across versions; existing versions are skipped)."""
import sys; sys.dont_write_bytecode = True
import os, json, time, re, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset, wib


def refute_v3(C, I, T, h, q, cp, budget, k_cuts=16):
    """V3: V2 windows, then the CP-SAT nested test (Proposition N) with the shaved windows (small det)."""
    t0 = time.process_time()
    ref, W = wib.refute_v2(C, T, h, q, budget=budget * 0.6, k_cuts=k_cuts)
    if ref: return True
    for K, mode in ((8, "score"), (12, "mix")):
        if time.process_time() - t0 > budget: return False
        if wib.nested_with_windows(I, T, W, cp.pick(T, K, mode), det=2.0): return True
    return False


def refute_v5(C, I, T, h, q, cp, budget, rounds=3, cfg=None):
    """V5: joint fixpoint of the polynomial windows (V2: P+F+E shaving + JLP probing) and the CP-SAT forced-J shaving
    (B10, lbx.shave_windows) on the current heads r2 / tails T - d1; then single-stage CP-SAT tests and the nested
    portfolio (Proposition N) with window forcing. Every refutation is a proof; timeouts = not refuted."""
    t0 = time.process_time(); dl = t0 + budget
    W = wib.init_win(I, T, h, q)
    def lp_extra(W_):
        return wib.lp_probe(C, T, W_, wib.lp_cuts(C, T, W_, 16), True, deadline=dl)
    for _ in range(rounds):
        k0 = W.key()
        W, ref = wib.shave(C, W, deadline=dl, extra=lp_extra)
        if ref: return True
        if time.process_time() > dl: return False
        h2, q2, ref = lbx.shave_windows(I, T, W.r2, (T - W.d1), cfg=cfg, det=0.1, deadline=min(dl, time.process_time() + budget * 0.25))
        if ref: return True
        W.r2 = np.maximum(W.r2, h2); W.d1 = np.minimum(W.d1, T - q2)
        if W.key() == k0: break
    if not wib.consistent(C, W): return True
    if time.process_time() > dl: return False
    if lbx.stage_np_lb(I["p2"], W.r2, I["m2"], tl=3.0, hi=T + 1)[0] > T: return True
    if lbx.stage_np_lb(I["p1"], (T - W.d1), I["m1"], tl=3.0, hi=T + 1)[0] > T: return True
    for K, mode in ((8, "score"), (8, "mix"), (16, "uniform"), (12, "score")):
        if time.process_time() > dl: return False
        if wib.nested_with_windows(I, T, W, cp.pick(T, K, mode), det=3.0, cfg=cfg): return True
    return False


def refute_v6(C, I, T, h, q, cp, budget):
    """V6 = V5 with door-start / door-end caps (Lemma W7 + door-start lemma) in every J / nested / LP test and in
    the CP-SAT forced-J shaving. C.prof must be set."""
    return refute_v5(C, I, T, h, q, cp, budget, cfg=lbx.Cfg(prof=C.prof))


def make(name, C, I, h, q, cp, per_T):
    if name == "v1": return lambda T: wib.refute_v1(C, T, h, q, budget=per_T)[0]
    if name == "v1d": return lambda T: wib.refute_v1(C, T, h, q, budget=per_T, mode="dich")[0]
    if name == "v1f": return lambda T: wib.refute_v1(C, T, h, q, budget=per_T, er=False)[0]
    if name == "v2": return lambda T: wib.refute_v2(C, T, h, q, budget=per_T, k_cuts=16)[0]
    if name == "v2nodff": return lambda T: wib.refute_v2(C, T, h, q, budget=per_T, k_cuts=16, dff=False)[0]
    if name == "v2k32": return lambda T: wib.refute_v2(C, T, h, q, budget=per_T, k_cuts=32)[0]
    if name == "v2a": return lambda T: wib.refute_v2(C, T, h, q, budget=per_T, k_cuts=16, avoid=True)[0]
    if name == "v4": return lambda T: wib.refute_v4(C, T, h, q, budget=per_T)[0]
    if name == "v6": return lambda T: refute_v6(C, I, T, h, q, cp, per_T)
    if name == "v2d": return lambda T: wib.refute_v2(C, T, h, q, budget=per_T, k_cuts=16)[0]
    if name == "v5": return lambda T: refute_v5(C, I, T, h, q, cp, per_T)
    if name == "v3": return lambda T: refute_v3(C, I, T, h, q, cp, per_T)
    raise KeyError(name)


def instances(which):
    if which == "solved":
        O = sset.load()
        for k in sorted(k for k, v in O.items() if v["opt"] is not None):
            v = O[k]; yield k, v["g"], v["n1"], v["seed"], int(v["lbu"]), int(v["best_hi"]), int(v["opt"])
    else:
        import eval105
        S = json.load(open(f"{D}/../lower_bounds/sample105.json")); U = eval105.ub105()
        for r in S:
            k = f"{r['g']}_{r['n1']}_{r['seed']}"; G = json.load(open(f"{D}/../lower_bounds/out_full/{k}.json"))
            lbs = int(np.ceil(max(G["lb_u1"], G["lb_rev"]) - 1e-9)); hi = int(min(U.get(k, 1e18), r["ub_hgr"]))
            yield k, r["g"], r["n1"], r["seed"], lbs, hi, None


def main():
    which, out, sh, ns, vers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5].split(",")
    budget = float(sys.argv[6]) if len(sys.argv) > 6 else 120.0
    rx = re.compile(sys.argv[7]) if len(sys.argv) > 7 else None
    os.makedirs(out, exist_ok=True)
    for qd, (k, g, n1, seed, lo, hi, opt) in enumerate(instances(which)):
        if qd % ns != sh or (rx and not rx.search(k)): continue
        fo = f"{out}/{k}.json"
        r = json.load(open(fo)) if os.path.exists(fo) else dict(key=k, g=g, n1=n1, lbstar=lo, hi=hi, opt=opt, host=platform.node(), py=platform.python_version())
        I = lbx.parse(sset.path(g, n1, seed)); h, q = lbx.heads_tails_bp(I); C = wib.Ctx(I); cp = None; prof = None
        r.update(n2=I["n2"], m1=I["m1"], m2=I["m2"])
        for name in vers:
            if name in r: continue
            if name in ("v3", "v5", "v6") and cp is None: cp = lbx.CutPicker(I, h, q)
            if name in ("v6", "v2d"):          # door caps ON only for these versions (never leak into others)
                if prof is None:
                    tp = time.process_time(); prof = wib.union_profiles2(C, det=1.0); r["prof_cpu"] = round(time.process_time() - tp, 2)
                C.prof = prof; wib.PROF[id(I)] = prof
            else:
                C.prof = None; wib.PROF.pop(id(I), None)
            # per-T budget: one T test may use at most half the instance budget (the gallop needs several tests)
            f = make(name, C, I, h, q, cp, per_T=budget / 2)
            lb, n, cpu = lbx.destructive(f, lo, hi, budget=budget)
            r[name] = dict(lb=int(lb), n=n, cpu=round(cpu, 2))
            ref = opt if opt is not None else hi
            assert lb <= ref, ("VALIDITY", k, name, lb, ref)
            json.dump(r, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
            print(k, name, lb, lo, opt or hi, round(cpu, 1), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
