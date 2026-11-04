"""Fresh-instance exact experiment (preregistered; see the paper): frozen exact method CP-UC on 300 fresh instances.
usage: exact_fresh.py gen                       -> data_fresh/*.dat + list_fresh.json (seed 20261193)
       exact_fresh.py run KEY ARMS OUTDIR TL     -> base (LB*, ILS-10 s warm start) if missing, then each arm once
ARMS (comma list, run in the given order; the queue rotates the order per instance):
  v3     (CP-UC) g6cp.exact_v3 (k = 12, forward + reversed union cuts, head/tail cuts)        [frozen, unchanged]
  v2lbu  (CP-LB) g5lib.exact_v2 started from LB*                                              [prior arm, unchanged]
  onone  (CP-plain) g5lib.cpsat_exact, optimisation mode, warm start, no LB, no cuts              [plain CP-SAT, unchanged]
  tim    (TIM-MIP) time-indexed MIP (Cota 2016) with HiGHS, 1 thread, CPU limit TL (tim_mip.py), horizon = warm-start UB
  v3nofwd / v3norev / v3noht   ablations: v3 without forward union cuts / reversed union cuts / head-tail cuts
Every arm gets the same LB* (where used), the same warm-start schedule and the same CPU limit TL (process time)."""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); R = f"{D}/../.."
for p in (D, f"{R}/fbils_census_exact_cpuc", f"{R}/exact_cpsat", f"{R}/confirmation_eacdc", f"{R}/lower_bounds",
          f"{R}/eacdc_public/code"):
    sys.path.insert(0, p)
import g5lib as L  # noqa: E402  (imports ortools first, then numpy)
import numpy as np  # noqa: E402
import g6cp, gen_new  # noqa: E402

SEED = 20261193; GROUPS = ("m2", "m4", "m10", "u24", "u210"); N1S = (20, 30); PER = 30


def gen():
    rng = np.random.default_rng(SEED); os.makedirs(f"{D}/data_fresh", exist_ok=True); out = []
    for n1 in N1S:
        for g in GROUPS:
            for k in range(PER):
                n1_, n2, m1, m2, p1, p2, P = gen_new.gen(rng, n1, g); fn = f"data_fresh/{g}_{n1}_{k}.dat"
                with open(f"{D}/{fn}", "w") as f:
                    f.write(f"{n1_}\n{n2}\n{m1}\n{m2}\n\n" + "\n".join(map(str, p1)) + "\n\n" + "\n".join(map(str, p2)) + "\n\n"
                            + "\n".join(" ".join(str(int(x)) for x in row) for row in P) + "\n")
                out.append(dict(key=f"{g}_{n1}_{k}", g=g, n1=n1, k=k, path=fn))
    json.dump(out, open(f"{D}/list_fresh.json", "w")); print(len(out))


def base(I, key, fo):
    """LB* exactly as in the fresh-instance census of the confirmation experiment (confirmation_eacdc lb_new.py: classical bounds, then the
    union walk forward from their max and on the reversed instance, tl 1 s / budget 4 s), then ILS 10 s, seed 1."""
    import cdc, g1, fonseca_port
    t = time.process_time()
    LBs, LBr, _, _ = fonseca_port.lbs_lbr(I["n1"], I["n2"], I["m1"], I["m2"], I["p1"].tolist(), I["p2"].tolist(),
                                          I["prec"].astype(int).tolist())
    kl = dict(lb_pch=cdc.lb_pch(I), lb_cs=max(LBs, LBr), lb_e2=cdc.lb_stage2_energy(I), lb_load=cdc.lb_load(I))
    lo = max(kl.values())
    u = g1.lb_union_exact(I, lo, tl=1.0, budget=4.0); rv = g1.lb_union_exact(g1.reverse(I), 0.0, tl=1.0, budget=4.0)
    lbu = float(np.ceil(max(u["lb"], rv["lb"], lo) - 1e-9)); lb_cpu = time.process_time() - t
    ub, sched, cpu, tr, constr = L.warm_ub(I, 10.0, seed=1)
    assert lbu <= ub + 1e-9, (key, lbu, ub)
    B = dict(key=key, m1=int(I["m1"]), m2=int(I["m2"]), n1=int(I["n1"]), n2=int(I["n2"]), lbu=lbu, lb_cpu=lb_cpu, **kl,
             lb_u1=u["lb"], lb_rev=rv["lb"], ub_ils10=float(ub), ils_cpu=cpu, constr=constr,
             s1=list(map(int, sched[0])), s2=list(map(int, sched[1])), py=platform.python_version(), host=platform.node())
    json.dump(B, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    return B


def run_arm(arm, I, B, tl):
    lo, ub, sched = B["lbu"], B["ub_ils10"], (B["s1"], B["s2"])
    if arm in ("v3", "v3nofwd", "v3norev", "v3noht"):
        kw = dict(v3nofwd=dict(fwd=False), v3norev=dict(rev=False)).get(arm, {})
        orig = g6cp.solve
        if arm == "v3noht":
            g6cp.solve = lambda *a, **k: orig(*a, ht_cuts=False, **k)
        try:
            x = g6cp.exact_v3(I, lo, ub, sched, tl, k=12, **kw)
        finally:
            g6cp.solve = orig
        return dict(opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"], t_opt=x["t_opt"], n_probe=x["n_probe"], log=x["log"]), x["sched"]
    if arm == "v2lbu":
        x = L.exact_v2(I, lo, ub, sched, tl)
        return dict(opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"], t_opt=x["t_opt"], n_probe=x["n_probe"], log=x["log"]), x["sched"]
    if arm == "onone":
        x = L.cpsat_exact(I, tl, hint=sched, horizon=int(ub))
        hi = int(x["ub"]) if x["ub"] is not None else int(ub)
        opt = hi if x["status"] == "OPTIMAL" else None
        lo_ = hi if opt is not None else int(np.ceil(x["lb"] - 1e-9))
        return dict(opt=opt, lo=lo_, hi=hi, cpu=x["cpu"], t_opt=x["t_opt"], status=x["status"]), x["sched"]
    if arm == "tim":     # separate interpreter (ortools/highspy symbol clash on Linux); CPU of the child is counted
        import subprocess, resource
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        o = subprocess.run([sys.executable, "-I", f"{D}/tim_mip.py", B["path"], str(int(ub)), str(tl), B["start_file"]],
                           capture_output=True, text=True, check=True).stdout
        r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        x = json.loads(next(l for l in o.splitlines() if l.startswith("TIMRESULT "))[10:])
        x["cpu_child_total"] = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
        return dict(opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"], wall=x["wall"], t_opt=x["cpu"] if x["opt"] is not None else None,
                    status=x["status"], ncol=x["ncol"], mip_nodes=x["mip_nodes"], cpu_child_total=x["cpu_child_total"]), None
    raise KeyError(arm)


def run(key, arms, out, tl):
    os.makedirs(out, exist_ok=True)
    r = next(x for x in json.load(open(f"{D}/list_fresh.json")) if x["key"] == key)
    I = L.cdc.parse(f"{D}/{r['path']}")
    fb = f"{out}/{key}_base.json"
    B = json.load(open(fb)) if os.path.exists(fb) else base(I, key, fb)
    B["path"] = f"{D}/{r['path']}"; B["start_file"] = fb       # for the tim child process (reads s1, s2 from the base)
    for arm in arms:
        fo = f"{out}/{key}_{arm}_tl{int(tl)}.json"
        if os.path.exists(fo): continue
        w0 = time.time(); res, sched = run_arm(arm, I, B, tl)
        if sched is not None and res["opt"] is not None:
            assert abs(L.cdc.check_schedule(I, *sched) - res["hi"]) < 1e-6, (key, arm)
        assert res["lo"] <= res["hi"] <= B["ub_ils10"] + 1e-9, (key, arm, res)
        res.update(key=key, arm=arm, tl=tl, wall=res.get("wall", time.time() - w0), host=platform.node(), py=platform.python_version())
        json.dump(res, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
        print(key, arm, res["opt"], res["lo"], res["hi"], round(res["cpu"], 1), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen": gen()
    else: run(sys.argv[2], [a for a in sys.argv[3].split(",") if a], sys.argv[4], float(sys.argv[5]))
