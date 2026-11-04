"""Analyse WIB runs on the solved set (gap closed vs LB*, per n1 and dock group, instance bootstrap CI95) and on
sample105 (gain over LB* per n1/group). Combined rows take the max with the section-2 bounds (decomp + decomp2).
usage: ana_wib.py SOLVED_DIR [E105_DIR] [TXT]"""
import sys; sys.dont_write_bytecode = True
import json, glob, os, numpy as np
rng = np.random.default_rng(0)
def ci(x, f="%.1f"):
    x = np.asarray(x, float)
    if len(x) == 0: return "n/a"
    if len(x) == 1: return f % x.mean()
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return (f + " [" + f + ", " + f + "]") % (x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5))
out = []; P = out.append
S = {}
for f in glob.glob(f"{sys.argv[1]}/*.json"):
    r = json.load(open(f)); S[r["key"]] = r
D0 = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob("out_decomp/*.json")}
D2 = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob("out_decomp2/*.json")}
def old_best(k, l0):
    d = D0.get(k, {}); e = D2.get(k, {})
    v = [l0] + [x["lb"] for x in d.get("valid", {}).values()] + [d[a] for a in ("pre", "s2np", "s1np", "s2pmtn", "s1pmtn") if isinstance(d.get(a), (int, float))]
    v += [e[a]["lb"] for a in ("shave", "combo", "docks") if a in e]
    return max(v)
vers = sorted({k for r in S.values() for k, v in r.items() if isinstance(v, dict) and "lb" in v})
viol = [(k, v, r[v]["lb"], r["opt"]) for k, r in S.items() for v in vers if v in r and r[v]["lb"] > r["opt"]]
P(f"solved set: {len(S)} instances; versions {vers}; VALIDITY violations (LB > OPT): {viol}")
P("| version | gap closed % [CI95] all | n1=20 | n1=30 | m2 | m4 | m10 | u24 | u210 | = OPT | CPU s mean / max |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
rows = [(v, (lambda v: lambda r: r[v]["lb"])(v)) for v in vers] + [("old BEST valid (§2, incl. combo)", lambda r: old_best(r["key"], r["lbstar"]))] + \
       [(f"max({v}, old BEST)", (lambda v: lambda r: max(r[v]["lb"], old_best(r["key"], r["lbstar"])))(v)) for v in vers]
for name, f in rows:
    base = [r for r in S.values() if all(v in r for v in vers if v in name or name.startswith("old"))]
    base = [r for r in base if (name.startswith("old") or all(v in r for v in vers if v == name or f"({v}," in name))]
    def cl(rs): return [100 * (f(r) - r["lbstar"]) / (r["opt"] - r["lbstar"]) for r in rs if r["opt"] > r["lbstar"]]
    cells = [ci(cl(base))] + [ci(cl([r for r in base if r["n1"] == n])) for n in (20, 30)] + [ci(cl([r for r in base if r["g"] == g])) for g in ("m2", "m4", "m10", "u24", "u210")]
    vn = name if name in vers else None
    cpu = f"{np.mean([r[vn]['cpu'] for r in base]):.1f} / {np.max([r[vn]['cpu'] for r in base]):.1f}" if vn else "—"
    P(f"| {name} (n={len(base)}) | " + " | ".join(cells) + f" | {sum(f(r) >= r['opt'] for r in base)}/{len(base)} | {cpu} |")
if len(sys.argv) > 2 and os.path.isdir(sys.argv[2]):
    E = [json.load(open(f)) for f in glob.glob(f"{sys.argv[2]}/*.json")]
    ev = sorted({k for r in E for k, v in r.items() if isinstance(v, dict) and "lb" in v})
    bad = [(r["key"], v) for r in E for v in ev if v in r and r[v]["lb"] > r["hi"]]
    P(f"\nsample105: {len(E)} instances; versions {ev}; VALIDITY (LB > best UB): {bad}")
    for v in ev:
        P(f"\n**{v}**: gain over LB* % [CI95] by n1 / group; share of (UB − LB*) closed")
        P("| subset | n | gain % [CI95] | share closed % | raised | CPU s mean / max |"); P("|---|---|---|---|---|---|")
        subs = [(f"n1={n}", [r for r in E if r["n1"] == n and v in r]) for n in sorted({r["n1"] for r in E})] + \
               [(g, [r for r in E if r["g"] == g and v in r]) for g in ("m2", "m4", "m10", "u24", "u210")] + [("all", [r for r in E if v in r])]
        for nm, rs in subs:
            if not rs: continue
            g = [100 * (r[v]["lb"] - r["lbstar"]) / r["lbstar"] for r in rs]
            sh = [100 * (r[v]["lb"] - r["lbstar"]) / (r["hi"] - r["lbstar"]) for r in rs if r["hi"] > r["lbstar"]]
            P(f"| {nm} | {len(rs)} | {ci(g, '%.2f')} | {ci(sh)} | {sum(r[v]['lb'] > r['lbstar'] for r in rs)}/{len(rs)} | {np.mean([r[v]['cpu'] for r in rs]):.1f} / {np.max([r[v]['cpu'] for r in rs]):.1f} |")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 3: open(sys.argv[3], "w").write(txt + "\n")
