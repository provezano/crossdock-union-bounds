"""Analyse decomp2 (shave/combo/docks on solved set), census (n1 in {20,30}) and ubdec (decoder headroom). Read-only."""
import sys; sys.dont_write_bytecode = True
import json, glob, collections, numpy as np
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0: return "n/a"
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.2f} [{np.percentile(b,2.5):.2f}, {np.percentile(b,97.5):.2f}]"
D = {}
for f in glob.glob("out_decomp/*.json"):
    r = json.load(open(f)); D[r["key"]] = r
rows = [json.load(open(f)) for f in glob.glob("out_decomp2/*.json")]
print(f"### decomp2 (n = {len(rows)})")
viol = []
for arm in ("shave", "combo", "docks"):
    gc, gp, cpu = [], [], []
    for r in rows:
        if arm not in r: continue
        o, l0, lb = r["opt"], r["lbstar"], r[arm]["lb"]
        if lb > o: viol.append((r["key"], arm, lb, o))
        gc.append(100 * (lb - l0) / (o - l0) if o > l0 else 0.0); gp.append(100 * (o - lb) / o); cpu.append(r[arm]["cpu"])
    print(f"| {arm} | gap% {ci(gp)} | closed% {ci(gc)} | cpu mean {np.mean(cpu):.1f} max {np.max(cpu):.1f} |")
# best valid incl. decomp2
bv, bv2 = [], []
for r in rows:
    d = D.get(r["key"]); 
    if not d: continue
    o, l0 = r["opt"], r["lbstar"]
    b0 = max([l0] + [v["lb"] for k, v in d.get("valid", {}).items()] + [d.get(k, l0) for k in ("pre", "s2np", "s1np", "s2pmtn", "s1pmtn") if isinstance(d.get(k), (int, float))])
    b1 = max([b0] + [r[a]["lb"] for a in ("shave", "combo", "docks") if a in r])
    if o > l0: bv.append(100 * (b0 - l0) / (o - l0)); bv2.append(100 * (b1 - l0) / (o - l0))
print(f"BEST valid (decomp) on same instances: {ci(bv)}; + shave/combo/docks: {ci(bv2)}")
print("VALIDITY decomp2:", viol)
C = []
for f in glob.glob("out_census/*.jsonl"):
    for l in open(f): C.append(json.loads(l))
print(f"\n### census (n = {len(C)})")
bad = [c["key"] for c in C if c["lb"] > c["ub"]]
for n1 in (20, 30, None):
    S = [c for c in C if n1 is None or c["n1"] == n1]
    if not S: continue
    g = [100 * (c["lb"] - c["lbstar"]) / c["lbstar"] for c in S]
    sh = [100 * (c["lb"] - c["lbstar"]) / (c["ub"] - c["lbstar"]) for c in S if c["ub"] > c["lbstar"]]
    cl = sum(c["lb"] >= c["ub"] for c in S); cl0 = sum(c["lbstar"] >= c["ub"] for c in S)
    print(f"| n1={n1 or 'all'} | n={len(S)} | gain% {ci(g)} | share closed% {ci(sh)} | raised {sum(c['lb']>c['lbstar'] for c in S)} | LB=UB {cl0}->{cl} | cpu mean {np.mean([c['cpu'] for c in S]):.1f} |")
print("VALIDITY census:", bad)
U = [json.loads(l) for l in open("out_ubdec/dec_0.jsonl")]
U = [u for u in U if u["dec_gain"] is not None]; g = [100 * u["dec_gain"] for u in U]
print(f"\n### ubdec n = {len(U)}; stage2 opt {sum(u['stage2_opt'] for u in U)}; gain% {ci(g)}; max {max(g):.2f}; improved {sum(x>0 for x in g)}; cpu {sum(u['cpu'] for u in U)/3600:.2f} h")
