"""Phase-1 table: mean gap to OPT (% of OPT) and share of the LB*->OPT gap closed, per (group, n1) and overall,
with instance-bootstrap CI95. usage: ana_decomp.py [TXT]"""
import json, glob, sys, collections, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob("out_decomp/*.json"))]
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2: return f"{x.mean():.1f}"
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.1f} [{np.percentile(b, 2.5):.1f}, {np.percentile(b, 97.5):.1f}]"
cols = [("LB*", lambda r: r["lbstar"]), ("PRE (pmtn windows)", lambda r: max(r["lbstar"], r["pre"])),
        ("S2 non-pmtn P|h|C", lambda r: max(r["lbstar"], r["s2np"])),
        ("J fluid", lambda r: r["valid"]["J_fluid"]["lb"]), ("J +partial", lambda r: r["valid"]["J_pw"]["lb"]),
        ("J +DFF", lambda r: r["valid"]["J_dff"]["lb"]), ("J exact pack", lambda r: r["valid"]["J_exact"]["lb"]),
        ("N8 no-DFF", lambda r: r["valid"]["N8s_nodff"]["lb"]), ("N8 +DFF", lambda r: r["valid"]["N8s"]["lb"]),
        ("N8 exact pack", lambda r: r["valid"]["N8s_exact"]["lb"]), ("N portfolio", lambda r: r["valid"]["N_port"]["lb"]),
        ("BEST valid", lambda r: max([r["lbstar"], r["pre"], r["s2np"], r["s1np"]] + [v["lb"] for v in r["valid"].values()]))]
orc = [("O: pmtn stage 2 w/ S* heads", lambda r: r["oracle"]["pre2"]), ("O: fluid e2 w/ S* heads", lambda r: np.ceil(r["oracle"]["e2"] - 1e-9)),
       ("O: pmtn stage 1 w/ S* tails", lambda r: r["oracle"]["pre1"]), ("O: N8 w/ S* heads only", lambda r: r["oracle"]["heads_only"]["N8s"]["lb"]),
       ("O: N8 w/ S* tails only", lambda r: r["oracle"]["tails_only"]["N8s"]["lb"]), ("O: N8 w/ S* heads+tails", lambda r: r["oracle"]["nested"]["N8s"]["lb"]),
       ("O: J exact w/ S* heads+tails", lambda r: r["oracle"]["nested"]["J_exact"]["lb"])]
out = []; P = out.append
groups = [("all", R)] + [(f"n1={n}", [r for r in R if r["n1"] == n]) for n in (20, 30)] + \
         [(f"{g}", [r for r in R if r["g"] == g]) for g in ("m2", "m4", "m10", "u24", "u210")]
P(f"instances: {len(R)}; with oracle schedule: {sum('oracle' in r for r in R)}")
for gname, rs in groups:
    if not rs: continue
    P(f"\n### {gname} (n = {len(rs)})")
    P("| Bound | (OPT − LB)/OPT % [CI95] | gap closed vs LB* % [CI95] | = OPT |")
    P("|---|---|---|---|")
    for name, f in cols + orc:
        rr = [r for r in rs if (not name.startswith("O:")) or "oracle" in r]
        if not rr: continue
        g = [(r["opt"] - f(r)) / r["opt"] * 100 for r in rr]
        cl = [(f(r) - r["lbstar"]) / (r["opt"] - r["lbstar"]) * 100 for r in rr if r["opt"] > r["lbstar"]]
        P(f"| {name} | {ci(g)} | {ci(cl)} | {sum(f(r) >= r['opt'] for r in rr)}/{len(rr)} |")
viol = [(r["key"], n) for r in R for n, f in cols if f(r) > r["opt"]]
P(f"\nVALIDITY: bounds above OPT: {viol}")
cp = collections.defaultdict(list)
for r in R:
    for n, v in r["valid"].items(): cp[n].append(v["cpu"])
P("CPU s per bound (mean / max): " + "; ".join(f"{n} {np.mean(v):.1f}/{np.max(v):.1f}" for n, v in cp.items()))
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
