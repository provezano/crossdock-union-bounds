"""anytime profile experiment: IG tuning on DEV only (sample105, Cota files). Full factorial grid (irace-like, no elimination needed at
this cost): d in {1,2,3,4,6,8} x T in {0,0.1,0.2,0.4,0.8,1.6} x LS in {0,1} = 72 configurations, seed 7 (dev-only seed),
each run to 10 x E0 evaluations and read at 1 x E0 and 10 x E0 (prefix of one seeded trajectory).
Finalist round: the 5 best configurations (mean RPD over the two budgets) re-run to 100 x E0.
usage: ig_tune.py grid OUT SHARD NSH | ig_tune.py final OUT FINALISTS.json SHARD NSH"""
import sys, os, json, itertools, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import common as C  # noqa
GRID = [dict(d=d, T=T, ls=ls) for d, T, ls in itertools.product([1, 2, 3, 4, 6, 8], [0.0, 0.1, 0.2, 0.4, 0.8, 1.6], [0, 1])]
SEED = 7
def cname(c): return f"d{c['d']}_T{c['T']}_ls{c['ls']}"
def main():
    mode, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
    dev = json.load(open(f"{D}/list_dev105.json"))
    if mode == "grid": cfgs, mult, sh, ns = GRID, [1, 10], int(sys.argv[3]), int(sys.argv[4])
    else: cfgs, mult, sh, ns = json.load(open(sys.argv[3])), [1, 10, 100], int(sys.argv[4]), int(sys.argv[5])
    jobs = [(r, c) for r in dev for c in cfgs]
    env = dict(python=sys.version.split()[0], host=platform.node())
    for q, (r, c) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}__{cname(c)}.json"
        if os.path.exists(fo): continue
        I = C.cdc.parse(f"{D}/{r['path']}"); E0 = r["E0"]
        x = C.run_ig(I, max(mult) * E0, SEED, c["d"], c["T"], c["ls"])
        rec = dict(key=r["key"], n1=r["n1"], cfg=c, E0=E0, vals={str(k): C.at(x["trace"], k * E0) for k in mult},
                   cpu={str(k): C.cpu_at(x["trace"], k * E0) for k in mult}, nev=x["nev"], cpu_total=x["cpu"], capped=x["capped"], env=env)
        json.dump(rec, open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done", flush=True)
if __name__ == "__main__":
    main()
