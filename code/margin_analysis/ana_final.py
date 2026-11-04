"""Tables for the paper and the note: margins by headroom band (pooled, per set), share of EACDC's certified gap closed,
partial correlation of union gain given headroom, true headroom on solved structured n1 = 30 instances (exact_out).
usage: ana_final.py"""
import json, glob, numpy as np, collections
rows = json.load(open("rows.json")); g = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); b = x[g.integers(0, len(x), (10000, len(x)))].mean(1); return f"{x.mean():+.2f} [{np.percentile(b,2.5):+.2f}, {np.percentile(b,97.5):+.2f}]"
FB = collections.defaultdict(list)
for f in glob.glob("fb_out/*.json"):
    r = json.load(open(f)); FB[r["key"]].append((r["v2_ev"] - r["eacdc_best"]) / r["eacdc_best"] * 100)
for r in rows: r["fb_ev"] = float(np.mean(FB[r["key"]])) if r["key"] in FB else None
bands = [(0, 3), (3, 6), (6, 10), (10, 100)]
print("band h_E | set | n | base m_cpu | base m_ev | FB m_ev(struct) | mean h_E | share closed (cpu)")
for lo, hi in bands:
    for src in ("bench", "struct", "all"):
        q = [r for r in rows if lo <= r["h_E"] < hi and (src == "all" or r["src"] == src)]
        if not q: continue
        fb = [r["fb_ev"] for r in q if r["fb_ev"] is not None]
        print(f"[{lo},{hi}) {src:6s} {len(q):3d}  {ci([r['m_cpu'] for r in q])}  {ci([r['m_ev'] for r in q])}  {np.mean(fb) if fb else float('nan'):+.2f}  {np.mean([r['h_E'] for r in q]):5.2f}  {np.mean([-r['m_cpu'] for r in q])/np.mean([r['h_E'] for r in q]):.3f}")
for src in ("bench", "struct"):
    q = [r for r in rows if r["src"] == src]
    print(f"{src}: share closed = -mean(m_cpu)/mean(h_E) = {np.mean([-r['m_cpu'] for r in q])/np.mean([r['h_E'] for r in q]):.3f}; mean h_E {np.mean([r['h_E'] for r in q]):.2f}")
# slope of m on h_E, per set, and test of equal slope
for src in ("bench", "struct", "all"):
    q = [r for r in rows if src == "all" or r["src"] == src]; x = np.array([r["h_E"] for r in q]); y = np.array([r["m_cpu"] for r in q])
    A = np.column_stack([np.ones_like(x), x]); c = np.linalg.lstsq(A, y, rcond=None)[0]
    bs = []
    for _ in range(4000):
        i = g.integers(0, len(x), len(x)); bs.append(np.linalg.lstsq(A[i], y[i], rcond=None)[0][1])
    print(f"OLS m_cpu = {c[0]:+.3f} {c[1]:+.4f} h_E  ({src}, n={len(x)}; slope CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}])")
# pooled: is the set difference explained by headroom? regress m on h_E + struct dummy
x = np.array([r["h_E"] for r in rows]); d = np.array([r["src"] == "struct" for r in rows], float); y = np.array([r["m_cpu"] for r in rows])
A = np.column_stack([np.ones_like(x), x, d]); c = np.linalg.lstsq(A, y, rcond=None)[0]
bs = []
for _ in range(4000):
    i = g.integers(0, len(x), len(x)); bs.append(np.linalg.lstsq(A[i], y[i], rcond=None)[0][2])
print(f"pooled m_cpu = {c[0]:+.3f} {c[1]:+.4f} h_E {c[2]:+.3f} struct  (struct dummy CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}])")
# partial Spearman of m on gainU given h_E (residualise ranks)
rk = lambda v: np.argsort(np.argsort(v)).astype(float)
for src in ("bench", "struct"):
    q = [r for r in rows if r["src"] == src]; a = rk([r["m_cpu"] for r in q]); b = rk([r["gainU"] for r in q]); h = rk([r["h_E"] for r in q])
    res = lambda u: u - np.polyval(np.polyfit(h, u, 1), h)
    print(f"{src}: partial rank corr(m_cpu, gainU | h_E) = {np.corrcoef(res(a), res(b))[0,1]:+.2f}; corr(gainU, h_E) = {np.corrcoef(b, h)[0,1]:+.2f}")
# true headroom (exact)
E = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob("exact_out/*.json")}
R = {r["key"]: r for r in rows if r["src"] == "struct"}
sol = [k for k in E if E[k]["opt"] is not None]
print(f"\nexact v3 on struct n1=30: {len(sol)}/{len(E)} solved (300 s)")
for k in sorted(E):
    r = R[k]; o = E[k]["opt"]
    tag = f"OPT {o}: EACDC {(r['ue']-o)/o*100:+.2f}% above, best-of-runs {(r['ub_best']-o)/o*100:+.2f}%, LB* {(o-r['lbstar'])/o*100:.2f}% below" if o else f"open [{E[k]['lo']}, {E[k]['hi']}]"
    print(f"  {k:12s} m_cpu {r['m_cpu']:+.2f}  h_E {r['h_E']:.2f}%  {tag}")
if sol:
    tE = np.array([(R[k]["ue"] - E[k]["opt"]) / E[k]["opt"] * 100 for k in sol]); lbs = np.array([(E[k]["opt"] - R[k]["lbstar"]) / E[k]["opt"] * 100 for k in sol])
    m = np.array([R[k]["m_cpu"] for k in sol])
    print(f"  solved: mean true EACDC headroom {tE.mean():.2f}%, LB* slack {lbs.mean():.2f}%, base ILS m_cpu {m.mean():+.2f}%; corr(m, true headroom) {np.corrcoef(rk(m), rk(tE))[0,1]:+.2f}")
    print(f"  base ILS closes {np.mean(-m)/tE.mean():.2f} of EACDC's distance to OPT on solved instances")
