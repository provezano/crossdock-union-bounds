"""FB-ILS census: ILS (frozen step 2/step 3 constants, best step 2 constructive, constructive CPU counted) on all 10,500 Cota
instances, 10 CPU s, seed 1 -- exactly g5lib.warm_ub(I, 10.0, seed=1), the protocol of the step 1 exact-run bases
(those 300 are reused, not rerun). Output: one JSONL line per instance in OUT/ils_<shard>.jsonl (resumable).
usage: ils_all.py OUT SHARD NSH [SEED]"""
import sys, os, json, platform, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../exact_cpsat")
import g5lib as L  # noqa: E402
import numpy as np  # noqa: E402

def all_keys():
    R = json.load(open(f"{D}/../lower_bounds/allbounds.json"))
    ks = sorted({(r['g'], r['n1'], r['seed']) for r in R})
    rng = np.random.default_rng(6)
    return [ks[i] for i in rng.permutation(len(ks))]   # mix cells across shards

def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    os.makedirs(out, exist_ok=True); fo = f"{out}/ils_s{seed}_{sh}.jsonl"
    done = set()
    if os.path.exists(fo):
        for ln in open(fo):
            try: done.add(json.loads(ln)["key"])
            except Exception: pass
    ex = f"{D}/../exact_cpsat/out_exact"
    f = open(fo, "a")
    for q, (g, n1, sd) in enumerate(all_keys()):
        if q % ns != sh: continue
        key = f"{g}_{n1}_{sd}"
        if key in done: continue
        r = L.g1rec(g, n1, sd); I = L.cdc.parse(L.path(g, n1, sd))
        bp = f"{ex}/{key}_base.json"
        if seed == 1 and os.path.exists(bp):
            B = json.load(open(bp)); ub, ub4, constr, cpu, s1, s2, src = B["ub_ils10"], B["ub_ils4"], B["constr"], None, B["s1"], B["s2"], "g5base"
        else:
            ub, sched, cpu, tr, constr = L.warm_ub(I, 10.0, seed=seed)
            ub4 = min(v for c, v in tr if c <= 4.0 + 1e-9); s1, s2 = sched; src = "run"
        mk = L.cdc.check_schedule(I, s1, s2); assert abs(mk - ub) < 1e-6
        rec = dict(key=key, g=g, n1=n1, seed=sd, n2=I["n2"], m1=I["m1"], m2=I["m2"], ils_seed=seed, ub10=ub, ub4=ub4,
                   constr=constr, cpu=cpu, src=src, lbstar=L.lbu_star(r), lb_cs=r["lb_hgr"], lb_pch=r["lb_pch"],
                   ub_hgr=r["ub_hgr"], ub_h1=r["ub_h1"], ub_h2=r["ub_h2"], s1=list(map(int, s1)), s2=list(map(int, s2)),
                   py=platform.python_version(), host=platform.node())
        f.write(json.dumps(rec) + "\n"); f.flush()
    print("done", flush=True)

if __name__ == "__main__":
    main()
