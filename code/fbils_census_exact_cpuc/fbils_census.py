"""FB-ILS census (final method): ILS v2 (compiled ILS + forward-backward improvement + window moves; the frozen prereg
configuration of confirm_fbils.py: pfbi 0, stall_mult 40, rounds 3, wfrac 0.25) on the Cota instances, 10 CPU s incl. the
best step 2 constructive, seed 1, CPU-only stop (no eval budget). Same instance order as ils_all.py (rng 6 permutation, so a
prefix is a random sample of all cells). One JSONL line per instance in OUT/v2_s<seed>_<shard>.jsonl (resumable).
Keys listed in OUT/done_*.txt are skipped. Remote outputs are pulled into per-host dirs out_v2c_<host>/ (file
names collide across NSH values). usage: fbils_census.py OUT SHARD NSH [SEED] [BUDGET]"""
import sys, os, json, platform, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, f"{D}/../exact_cpsat")
import g5lib as L, srch  # noqa: E402
from ils_all import all_keys  # noqa: E402
g2 = L.g2

def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    B = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
    os.makedirs(out, exist_ok=True); fo = f"{out}/v2_s{seed}_{sh}.jsonl"; done = set()
    if os.path.exists(fo):
        for ln in open(fo):
            try: done.add(json.loads(ln)["key"])
            except Exception: pass
    import glob  # keys finished elsewhere (e.g. the local NSH=4 shards, copied as OUT/done_*.txt, one key per line)
    for fd in glob.glob(f"{out}/done_*.txt"): done |= set(open(fd).read().split())
    f = open(fo, "a")
    for q, (g, n1, sd) in enumerate(all_keys()):
        if q % ns != sh: continue
        key = f"{g}_{n1}_{sd}"
        if key in done: continue
        r = L.g1rec(g, n1, sd); I = L.cdc.parse(L.path(g, n1, sd))
        t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
        J = I if side == "f" else g2.reverse(I); tcon = time.process_time() - t0
        b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, max(0.0, B - tcon), 0, cap=B + 1, seed=100 + seed,
                                     nev0=len(C), pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
        assert g2.Dec(J)(bo) == b
        cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(L.cdc.check_schedule(I, s1, s2) - b) < 1e-6
        tr = [(c + tcon, e, v) for c, e, v in tr]; ub4 = min([v for c, e, v in tr if c <= 4.0 + 1e-9] or [C[kb][0]])
        rec = dict(key=key, g=g, n1=n1, seed=sd, n2=I["n2"], m1=I["m1"], m2=I["m2"], ils_seed=seed, method="v2", budget=B,
                   ub10=float(b), ub4=float(ub4), constr=kb, tcon=tcon, nev=ne, lbstar=L.lbu_star(r), lb_cs=r["lb_hgr"],
                   lb_pch=r["lb_pch"], ub_hgr=r["ub_hgr"], ub_h1=r["ub_h1"], ub_h2=r["ub_h2"],
                   s1=list(map(int, s1)), s2=list(map(int, s2)), py=platform.python_version(), host=platform.node())
        f.write(json.dumps(rec) + "\n"); f.flush()
    print("done", flush=True)

if __name__ == "__main__":
    main()
