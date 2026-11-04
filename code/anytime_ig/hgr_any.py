"""anytime profile experiment: HgR (our re-implementation of Fonseca, Nogueira & Ravetti 2019; frozen hgr_comparison/frozen/hgr.py,
sha256 8d88583b...) run once per TEST instance to its own stopping rule with a 120 CPU-s safety budget (deterministic
method). Also HgR-noLP (LP skipped; the variant most favourable to HgR in the HgR prereg). Records the UB, the number
of decodes (nev), CPU and stop reason; every schedule checked with cdc.check_schedule.
usage: hgr_any.py LIST OUT SHARD NSH   (interpreter: hgr_comparison venv; highspy 1.15.1)"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); FZ = f"{D}/../hgr_comparison/frozen"; sys.path.insert(0, FZ)
import hgr, cdc  # noqa
hgr.LP_OPTS = dict(solver="ipm")
B = 120.0
def main():
    lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
    env = dict(python=sys.version.split()[0], host=platform.node())
    for q, r in enumerate(json.load(open(lst))):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}.json"
        if os.path.exists(fo): continue
        path = f"{D}/{r['path']}"; I = cdc.parse(path); res = {}
        for v in (("hgr", "hgrnolp") if q % 2 == 0 else ("hgrnolp", "hgr")):
            P = hgr.load(path); t0 = time.process_time()
            x = hgr.hgr(P, mode="budget", budget=B, h2_bug=True, lp_frac=(0.0 if v == "hgrnolp" else 0.5),
                        lp_cap=(0.0 if v == "hgrnolp" else 360.0))
            cpu = time.process_time() - t0
            mk = cdc.check_schedule(I, [int(z) for z in x["s1"]], [int(z) for z in x["s2"]]); assert abs(mk - x["ub"]) < 1e-6
            res[v] = dict(ub=float(x["ub"]), nev=int(x["nev"]), cpu=cpu, stop=x["stop"], n_rounds=len(x["rounds"]),
                          lp_ok=[y["lp_ok"] for y in x["rounds"]])
            print(r["key"], v, x["ub"], x["nev"], round(cpu, 1), x["stop"], flush=True)
        json.dump(dict(key=r["key"], n1=r["n1"], g=r["g"], budget=B, env=env, **res), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done", flush=True)
if __name__ == "__main__":
    main()
