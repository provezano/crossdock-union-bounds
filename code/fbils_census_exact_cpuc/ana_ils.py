"""FB-ILS census analysis: ILS-10 s census (all shards, dedup by key) -> per-cell new best-knowns vs the published HgR
UBs, certified ratios UB/LB*, counts of improved / tied / worse published best-knowns. Optional: exact-run optima
and proven LBs (step 1 + step 6 v3) tighten both sides on the 300-instance subset. usage: ana_ils.py [OUTTXT] [GLOB] [BKJSON]
(default GLOB = v1 census out_ils/ils_s1_*.jsonl; v2 census: 'out_v2c/v2_s1_*.jsonl', bk_table_v2.json)"""
import json, glob, os, sys, collections, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
recs = {}
for fp in sorted(glob.glob(f"{D}/" + (sys.argv[2] if len(sys.argv) > 2 else "out_ils/ils_s1_*.jsonl"))):
    for ln in open(fp):
        try: r = json.loads(ln)
        except Exception: continue
        recs.setdefault(r["key"], r)
ex = {}
try:
    import solved_set
    for k, v in solved_set.load().items(): ex[k] = dict(lo=v["best_lo"], hi=v["best_hi"], opt=v["opt"])
except Exception as e: print("no exact", e)
for fp in glob.glob(f"{D}/out_exact_v3/*_tl300.json"):
    a = json.load(open(fp)); e = ex.setdefault(a["key"], dict(lo=0, hi=1e18, opt=None))
    e["lo"] = max(e["lo"], a["lo"]); e["hi"] = min(e["hi"], a["hi"])
    if a["opt"] is not None: e["opt"] = a["opt"]
out = []; P = lambda *a: out.append(" ".join(str(x) for x in a))
R = list(recs.values()); P(f"instances: {len(R)} / 10500")
G = ["m2", "m4", "m10", "u24", "u210"]
def summ(rs, tag):
    if not rs: return
    d = np.array([(r["ub10"] - r["ub_hgr"]) / r["ub_hgr"] * 100 for r in rs])
    imp = sum(r["ub10"] < r["ub_hgr"] for r in rs); tie = sum(r["ub10"] == r["ub_hgr"] for r in rs)
    lbx = [max(r["lbstar"], ex.get(r["key"], {}).get("lo", 0)) for r in rs]
    ubx = [min(r["ub10"], r["ub_hgr"], ex.get(r["key"], {}).get("hi", 1e18)) for r in rs]
    cr = np.array([u / l for u, l in zip(ubx, lbx)])
    cert = sum(abs(u - l) < 1e-9 for u, l in zip(ubx, lbx))
    P(f"{tag:10s} n={len(rs):5d}  ILS-HgR {d.mean():+.2f}% (med {np.median(d):+.2f})  improved {imp} ({imp/len(rs)*100:.1f}%) tied {tie} worse {len(rs)-imp-tie}  "
      f"BK/LB mean {cr.mean():.4f} med {np.median(cr):.4f} p90 {np.quantile(cr,.9):.4f} max {cr.max():.4f}  <=1% {np.mean(cr<=1.01)*100:.1f}%  <=5% {np.mean(cr<=1.05)*100:.1f}%  certified-opt {cert}")
summ(R, "ALL")
for n1 in range(20, 90, 10): summ([r for r in R if r["n1"] == n1], f"n1={n1}")
for g in G: summ([r for r in R if r["g"] == g], f"g={g}")
for h in sorted({r["host"] for r in R}): summ([r for r in R if r["host"] == h], f"host={h[:14]}")
P("\nper cell (group x n1): ILS-HgR %, improved %, mean BK/LB")
for g in G:
    row = []
    for n1 in range(20, 90, 10):
        rs = [r for r in R if r["g"] == g and r["n1"] == n1]
        if rs:
            row.append(f"{n1}:{np.mean([(r['ub10']-r['ub_hgr'])/r['ub_hgr']*100 for r in rs]):+.2f}/{np.mean([r['ub10']<r['ub_hgr'] for r in rs])*100:.0f}%/"
                       f"{np.mean([min(r['ub10'],r['ub_hgr'])/r['lbstar'] for r in rs]):.3f}")
    P(g, " ".join(row))
cpu = [r.get("cpu") or r.get("budget") for r in R if r.get("cpu") or r.get("budget")]; P(f"\nCPU: {len(cpu)} fresh runs, {sum(cpu)/3600:.1f} CPU-h; hosts {collections.Counter(r['host'] for r in R)}")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
json.dump({k: dict(ub10=r["ub10"], ub4=r["ub4"], ub_hgr=r["ub_hgr"], lbstar=r["lbstar"], lb_proven=max(r["lbstar"], ex.get(k, {}).get("lo", 0)),
               bk=min(r["ub10"], r["ub_hgr"], ex.get(k, {}).get("hi", 1e18)), opt=ex.get(k, {}).get("opt")) for k, r in recs.items()},
          open(f"{D}/" + (sys.argv[3] if len(sys.argv) > 3 else "bk_table.json"), "w"))
