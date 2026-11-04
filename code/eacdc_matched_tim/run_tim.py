"""TIM-LP (Cota+2016) vs LB_U* (lb_u1) and known LBs.
usage: run_tim.py small OUTDIR            (generated n1<=8 instances, Cota generator; OPT by CP-SAT)
       run_tim.py real  OUTDIR SHARD NSH  (50 benchmark instances n1=20, T = best known UB = HgR)"""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, f"{D}/../benchmark_data"); sys.path.insert(0, f"{D}/../lower_bounds")
sys.path.insert(0, f"{D}/../eacdc_public/code")
from ortools.sat.python import cp_model  # noqa: load before highspy
import numpy as np
import cdc, g1, fonseca_port
from tim_lp import tim_lp

F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}


def known_lbs(I):
    n1, n2 = I["n1"], I["n2"]; S = I["prec"].astype(int).tolist()
    LBs, LBr, _, _ = fonseca_port.lbs_lbr(n1, n2, I["m1"], I["m2"], I["p1"].tolist(), I["p2"].tolist(), S)
    return dict(lb_pch=cdc.lb_pch(I), lb_e2=cdc.lb_stage2_energy(I), lb_load=cdc.lb_load(I), lb_cs=max(LBs, LBr))


def gen(rng, n1, mt):
    """Cota+2016 generator as described: p~U(10,100), precedence prob 0.5, n2 in [0.8n1,1.2n1], m in {2,4,U(2,4)}.
    Repair (as in the benchmark): every outbound has >=1 predecessor, every inbound >=1 successor."""
    n2 = int(rng.integers(int(np.ceil(0.8 * n1)), int(np.floor(1.2 * n1)) + 1))
    # benchmark u24 files draw m1 and m2 independently (m1 == m2 in only 833/2100); fixed 2026-11-02 (review m2).
    # The 300 small instances of out_tim_small were generated with m1 = m2 (old rule); the u24 third is therefore
    # the m1 = m2 sub-case, which does not affect the TIM-LP vs LB_U* comparison (both bounds take m1, m2 as input).
    m1, m2 = {'m2': (2, 2), 'm4': (4, 4)}.get(mt) or (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    P = rng.random((n2, n1)) < 0.5
    for j in range(n2):
        if not P[j].any(): P[j, rng.integers(n1)] = True
    for i in range(n1):
        if not P[:, i].any(): P[rng.integers(n2), i] = True
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=rng.integers(10, 101, n1), p2=rng.integers(10, 101, n2), prec=P)


def small(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261102)
    for n1 in (4, 5, 6, 7, 8):
        for mt in ('m2', 'm4', 'u24'):
            for k in range(20):
                I = gen(rng, n1, mt); fo = f"{out}/{mt}_{n1}_{k}.json"
                if os.path.exists(fo): continue
                kl = known_lbs(I); lo = max(kl.values())
                t = time.process_time(); u = g1.lb_union_exact(I, lo, tl=5, budget=60); cu = time.process_time() - t
                rv = g1.lb_union_exact(g1.reverse(I), 0.0, tl=5, budget=60)
                op = cdc.cpsat(I, time_limit=60)
                T = int(round(op["ub"]))
                a = tim_lp(I); b = tim_lp(I, T=T)
                json.dump(dict(mt=mt, n1=n1, k=k, n2=I["n2"], m=I["m1"], **kl, lb_u1=u["lb"], u1_exact=u["exact"], u1_cpu=cu,
                               lb_rev=rv["lb"], opt=op["ub"], opt_status=op["status"], tim_sumP=a["lb"], tim_sumP_cpu=a["cpu"],
                               tim=b["lb"], tim_cpu=b["cpu"], T=T, Tsum=a["T"]), open(fo, "w"))


def real(out, sh, ns):
    os.makedirs(out, exist_ok=True)
    A = json.load(open(f"{D}/../lower_bounds/allbounds.json"))
    rng = np.random.default_rng(7)
    pick = []
    for g in ('m2', 'm4', 'm10', 'u24', 'u210'):
        idx = [k for k, r in enumerate(A) if r['g'] == g and r['n1'] == 20]
        pick += sorted(rng.choice(idx, 10, replace=False).tolist())
    for q, k in enumerate(pick):
        if q % ns != sh: continue
        r = A[k]; fo = f"{out}/{r['g']}_20_{r['seed']}.json"
        if os.path.exists(fo): continue
        I = cdc.parse(f"{D}/../benchmark_data/data/Dados/{F[r['g']]}/20jobs/{r['seed']}.dat")
        kl = known_lbs(I); lo = max(kl.values())
        t = time.process_time(); u = g1.lb_union_exact(I, lo, tl=5, budget=60); cu = time.process_time() - t
        T = int(r['ub_hgr'])
        b = tim_lp(I, T=T, time_limit=600)
        json.dump(dict(r, **kl, lb_u1=u["lb"], u1_exact=u["exact"], u1_cpu=cu, tim=b["lb"], tim_status=b["status"],
                       tim_cpu=b["cpu"], T=T, ncol=b["ncol"], nrow=b["nrow"]), open(fo, "w"))


def large(out, sh, ns, tl):
    """Review D (2026-11-02): TIM-LP at n1 = 30/40/50, 3 instances each (one per group m2, m4, u210; rng 8),
    T = HgR UB, HiGHS time limit tl s. Non-optimal LPs give no valid bound and are reported as such."""
    os.makedirs(out, exist_ok=True)
    A = json.load(open(f"{D}/../lower_bounds/allbounds.json")); rng = np.random.default_rng(8); pick = []
    for n1 in (30, 40, 50):
        for g in ('m2', 'm4', 'u210'):
            idx = [k for k, r in enumerate(A) if r['g'] == g and r['n1'] == n1]; pick.append(int(rng.choice(idx)))
    for q, k in enumerate(pick):
        if q % ns != sh: continue
        r = A[k]; fo = f"{out}/{r['g']}_{r['n1']}_{r['seed']}.json"
        if os.path.exists(fo): continue
        I = cdc.parse(f"{D}/../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
        kl = known_lbs(I); lo = max(kl.values())
        t = time.process_time(); u = g1.lb_union_exact(I, lo, tl=1, budget=4); cu = time.process_time() - t
        T = int(r['ub_hgr']); b = tim_lp(I, T=T, time_limit=tl)
        json.dump(dict(r, **kl, lb_u1=u["lb"], u1_exact=u["exact"], u1_cpu=cu, tim=b["lb"], tim_status=b["status"],
                       tim_cpu=b["cpu"], T=T, ncol=b["ncol"], nrow=b["nrow"]), open(fo, "w"))


if __name__ == "__main__":
    if sys.argv[1] == "large": large(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])); sys.exit()
    if sys.argv[1] == "small": small(sys.argv[2])
    else: real(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
