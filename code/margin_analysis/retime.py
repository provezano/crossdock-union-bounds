"""Task 1 (2026-11-02): same-machine re-timing of exact v3 vs v2lbu on the 55 instances both solved in the main runs
(v3 on machine-B, v2lbu on local). Here both arms run on ONE machine, same process per instance, order alternating, same
start (LB*) and warm start (step 1 base ILS schedule) as the main runs, time limit TL (60 s to stay within 1 CPU-h).
Records CPU, proof status and CP-SAT branch/conflict counts are NOT available from the library (not recorded).
usage: retime.py OUT TL SHARD NSH"""
import sys, os, json, platform
G6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fbils_census_exact_cpuc"); sys.path.insert(0, G6); sys.path.insert(0, f"{G6}/../exact_cpsat")
import g6cp, g5lib as L
out, tl, sh, ns = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); os.makedirs(out, exist_ok=True)
keys = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "both55.json")))
for q, key in enumerate(keys):
    if q % ns != sh: continue
    fo = f"{out}/{key}.json"
    if os.path.exists(fo): continue
    B = json.load(open(f"{G6}/../exact_cpsat/out_exact/{key}_base.json"))
    I = L.cdc.parse(L.path(B["g"], B["n1"], B["seed"])); sched = (B["s1"], B["s2"])
    arms = ["v2lbu", "v3"] if q % 2 == 0 else ["v3", "v2lbu"]; res = {}
    for a in arms:
        x = L.exact_v2(I, B["lbu"], B["ub_ils10"], sched, tl) if a == "v2lbu" else g6cp.exact_v3(I, B["lbu"], B["ub_ils10"], sched, tl, k=12)
        res[a] = dict(opt=x["opt"], lo=x["lo"], hi=x["hi"], cpu=x["cpu"])
    json.dump(dict(key=key, tl=tl, first=arms[0], res=res, host=platform.node(), py=platform.python_version()), open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print(key, res, flush=True)
print("done", flush=True)
