"""Gap 6a: energetic-reasoning LB (er.lb_er) on all 10,500 benchmark instances.
lo = max(published/cheap LBs: lb_pch, lb_e2, lb_load, lb_hgr) (valid), hi = HgR UB (valid).
usage: run_er.py OUT.jsonl SHARD NSH"""
import sys, os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g5lib as L, er

out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
done = set()
if os.path.exists(out):
    done = {json.loads(l)["key"] for l in open(out)}
F = sorted(glob.glob(f"{D}/../lower_bounds/out_full/*.json"))
with open(out, "a") as fo:
    for q, fj in enumerate(F):
        if q % ns != sh: continue
        r = json.load(open(fj)); key = f"{r['g']}_{r['n1']}_{r['seed']}"
        if key in done: continue
        I = L.cdc.parse(L.path(r['g'], r['n1'], r['seed']))
        lo = max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_hgr'])
        e, n, c = er.lb_er(I, lo, r['ub_hgr'])
        fo.write(json.dumps(dict(key=key, g=r['g'], n1=r['n1'], lb_er=e, n_chk=n, cpu=c, lo=lo, lb_cs=r['lb_hgr'],
                                 lb_u1=r['lb_u1'], lb_rev=r['lb_rev'], ub_hgr=r['ub_hgr'])) + "\n"); fo.flush()
print("done")
