"""anytime profile experiment: tuned IG on TEST, after anytime.py. Configuration = ig_tuned.json (selected on DEV by ig_select.py, frozen and
hashed before this script runs). Run once (seed 51) to the largest target of the instance's anytime.py record, read at
every target. usage: igt_run.py LIST ANYDIR OUT SHARD NSH [CFG.json, smoke tests only]"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import common as C  # noqa
def main():
    lst, adir, out, sh, ns = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); os.makedirs(out, exist_ok=True)
    cfg = json.load(open(sys.argv[6] if len(sys.argv) > 6 else f"{D}/ig_tuned.json"))["cfg"]; env = dict(python=sys.version.split()[0], host=platform.node())
    for q, r in enumerate(json.load(open(lst))):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}.json"
        if os.path.exists(fo): continue
        A = json.load(open(f"{adir}/{r['key']}.json")); T = A["targets"]; I = C.cdc.parse(f"{D}/{r['path']}")
        x = C.run_ig(I, max(T.values()), A["seed"], cfg["d"], cfg["T"], cfg["ls"])
        json.dump(dict(key=r["key"], cfg=cfg, best=x["best"], nev=x["nev"], cpu=x["cpu"], capped=x["capped"],
                       at={t: C.at(x["trace"], e) for t, e in T.items()}, cpu_at={t: C.cpu_at(x["trace"], e) for t, e in T.items()},
                       trace=x["trace"], env=env), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done", flush=True)
if __name__ == "__main__":
    main()
