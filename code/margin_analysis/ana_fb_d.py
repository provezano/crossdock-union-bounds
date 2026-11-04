"""Exploratory: FB-ILS at matched evaluations on the structured set (fb_out) and the H(d) diagnosis (d_out).
usage: ana_fb_d.py"""
import json, glob, collections, numpy as np
rows = {r["key"]: r for r in json.load(open("rows.json")) if r["src"] == "struct"}
FB = collections.defaultdict(list)
for f in glob.glob("fb_out/*.json"):
    r = json.load(open(f)); FB[r["key"]].append((r["v2_ev"] - r["eacdc_best"]) / r["eacdc_best"] * 100)
x = np.array([np.mean(v) for k, v in sorted(FB.items())]); g = np.random.default_rng(0)
b = x[g.integers(0, len(x), (10000, len(x)))].mean(1)
w = ((x < -1e-9).sum(), (abs(x) <= 1e-9).sum(), (x > 1e-9).sum())
print(f"FB-ILS - EACDC at matched evaluations, struct n={len(x)}: {x.mean():+.3f}% [{np.percentile(b,2.5):+.3f}, {np.percentile(b,97.5):+.3f}] W/T/L {w}")
base = np.array([rows[k]["m_ev"] for k in sorted(FB)]); print(f"  base ILS same instances m_ev {base.mean():+.3f}%; FB - base {np.mean(x-base):+.3f}%")
hE = np.array([rows[k]["h_E"] for k in sorted(FB)]); print(f"  share of EACDC's certified gap closed: base {np.mean(-base/np.maximum(hE,1e-9)*(hE>0.5)) :.3f}  (instances with h_E>0.5%)")
rk = lambda v: np.argsort(np.argsort(v)); print(f"  spearman FB m_ev ~ h_E: {np.corrcoef(rk(x), rk(hE))[0,1]:+.2f}")
by = collections.defaultdict(list)
for k, v in FB.items(): by[(rows[k]["cell"], rows[k]["n1"])].append(np.mean(v))
print("  by cell: " + " ".join(f"{c}/{n}:{np.mean(v):+.2f}" for (c, n), v in sorted(by.items())))
fam = collections.defaultdict(list)
for k, v in FB.items(): fam[rows[k]["fam"]].append(np.mean(v))
print("  by family: " + " ".join(f"{f}:{np.mean(v):+.2f}(n={len(v)})" for f, v in sorted(fam.items())))
# H(d)
D = [json.load(open(f)) for f in glob.glob("d_out/*.json")]
print(f"\nH(d) runs: {len(D)}; decode of EACDC's induced inbound order by our decoder equals EACDC's value: {sum(abs(d['f_ours_on_eacdc_order']-d['eacdc_best'])<1e-9 for d in D)}/{len(D)}; rerun EACDC equals matched-run EACDC: {sum(abs(d['eacdc_best']-d['eacdc_match_run'])<1e-9 for d in D)}/{len(D)}")
cell = lambda k: k.rsplit("_", 1)[0]
G = collections.defaultdict(list)
for d in D: G[cell(d["key"])].append(d)
rel = lambda a, e: (a - e) / e * 100
print("cell        n  ILS-own  ILS-from-EACDC  ILS-outbound-rep   (all % vs EACDC, same evaluation budget)   plateau(=,worse,better): ILS-own | outrep | EACDC-inb | EACDC-outrep")
for c in sorted(G) + ["ALL", "ALL-but-U2x10"]:
    q = D if c == "ALL" else ([d for d in D if cell(d["key"]) != "U2x10_30"] if c == "ALL-but-U2x10" else G[c])
    m = lambda f: np.mean([rel(d[f], d["eacdc_best"]) for d in q])
    pl = lambda f: np.round(np.mean([d[f] for d in q], 0), 2).tolist()
    print(f"{c:14s}{len(q):3d}  {m('ils_own'):+.2f}    {m('ils_from_eacdc'):+.2f}           {m('ils_outbound_rep'):+.2f}        {pl('plateau_ils_own')} | {pl('plateau_ils_outrep')} | {pl('plateau_eacdc_inb')} | {pl('plateau_eacdc_outrep')}")
u = G["U2x10_30"]
for d in sorted(u, key=lambda d: (d["key"], d["seed"])): print("  U2x10_30", d["key"], d["seed"], "EACDC", d["eacdc_best"], "ILS-own", d["ils_own"], "from-EACDC", d["ils_from_eacdc"], "outrep", d["ils_outbound_rep"], "side", d["side"])
