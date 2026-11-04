"""Analysis of the fresh-instance exact experiment (frozen before any timed run; run once after all cells).
usage: ana_fresh.py OUTDIR TL > ana_fresh.txt"""
import sys, os, json, glob, math, collections
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
OUT, TL = sys.argv[1], int(float(sys.argv[2]))
LST = json.load(open(f"{D}/list_fresh.json")); KEYS = [r["key"] for r in LST]; N1 = {r["key"]: r["n1"] for r in LST}
G = {r["key"]: r["g"] for r in LST}
A = collections.defaultdict(dict)
for f in glob.glob(f"{OUT}/*_tl{TL}.json"):
    r = json.load(open(f)); A[r["key"]][r["arm"]] = r
B = {k: json.load(open(f"{OUT}/{k}_base.json")) for k in KEYS if os.path.exists(f"{OUT}/{k}_base.json")}


def binom_two_sided(b, c):
    """Exact McNemar: two-sided binomial test on the discordant pairs (b = only A solved, c = only B solved)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def boot(d, B_=10000, seed=0):
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(B_, d.size), replace=True).mean(axis=1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), min(1.0, max(p, 1.0 / B_))


def gap(r): return 100.0 * (r["hi"] - r["lo"]) / r["hi"]


print(f"instances with base: {len(B)} / {len(KEYS)}; arms present:",
      dict(collections.Counter(a for k in A for a in A[k])))
# integrity: proven optima agree across arms; every arm's lo <= every arm's hi
bad = []
for k in A:
    opts = {r["opt"] for r in A[k].values() if r.get("opt") is not None}
    if len(opts) > 1: bad.append((k, "opt disagreement", opts))
    if A[k]:
        mlo = max(r["lo"] for r in A[k].values()); mhi = min(r["hi"] for r in A[k].values())
        if mlo > mhi: bad.append((k, "lo > hi across arms", mlo, mhi))
    if k in B and any(r["lo"] < 0 for r in A[k].values()): bad.append((k, "neg lo"))
print("integrity violations:", bad)

CONF = [("v3", "v2lbu", [k for k in KEYS]), ("v3", "onone", [k for k in KEYS]), ("v3", "tim", [k for k in KEYS if N1[k] == 20])]
print("\n== Confirmatory family S (solved within the limit; exact McNemar, Holm over 3) ==")
rows, ps = [], []
for a, b, ks in CONF:
    ks = [k for k in ks if a in A[k] and b in A[k]]
    sa = [A[k][a]["opt"] is not None for k in ks]; sb = [A[k][b]["opt"] is not None for k in ks]
    bb = sum(x and not y for x, y in zip(sa, sb)); cc = sum(y and not x for x, y in zip(sa, sb))
    p = binom_two_sided(bb, cc); ps.append(p); rows.append((a, b, len(ks), sum(sa), sum(sb), bb, cc, p))
for (a, b, n, x, y, bb, cc, p), ph in zip(rows, holm(ps)):
    print(f"{a} vs {b}: n={n} solved {x} vs {y}; only {a} {bb}, only {b} {cc}; p={p:.3g}; Holm p={ph:.3g}")

print("\n== Confirmatory family G (final gap (hi-lo)/hi %, paired bootstrap 10,000, Holm over 3) ==")
rows, ps = [], []
for a, b, ks in CONF:
    ks = [k for k in ks if a in A[k] and b in A[k]]
    d = [gap(A[k][a]) - gap(A[k][b]) for k in ks]
    m, lo, hi, p = boot(d); ps.append(p)
    rows.append((a, b, len(ks), np.mean([gap(A[k][a]) for k in ks]), np.mean([gap(A[k][b]) for k in ks]), m, lo, hi, p,
                 sum(x < -1e-12 for x in d), sum(abs(x) <= 1e-12 for x in d), sum(x > 1e-12 for x in d)))
for (a, b, n, ga, gb, m, lo, hi, p, w, t, l), ph in zip(rows, holm(ps)):
    print(f"{a} vs {b}: n={n} mean gap {ga:.3f}% vs {gb:.3f}%; diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}] pp; lower/tie/higher {w}/{t}/{l}; p={p:.3g}; Holm p={ph:.3g}")

print("\n== Descriptive ==")
for arm in ("v3", "v2lbu", "onone", "tim", "v3nofwd", "v3norev", "v3noht"):
    for n1 in (20, 30):
        ks = [k for k in KEYS if N1[k] == n1 and arm in A[k]]
        if not ks: continue
        s = [k for k in ks if A[k][arm]["opt"] is not None]
        byg = {g: sum(1 for k in s if G[k] == g) for g in ("m2", "m4", "m10", "u24", "u210")}
        cpu = [A[k][arm]["cpu"] for k in ks]
        print(f"{arm:8s} n1={n1}: solved {len(s)}/{len(ks)} {byg}; mean gap {np.mean([gap(A[k][arm]) for k in ks]):.3f}%; "
              f"CPU mean/max {np.mean(cpu):.1f}/{np.max(cpu):.1f}")
both = [k for k in KEYS if all(a in A[k] and A[k][a]["opt"] is not None for a in ("v3", "v2lbu"))]
if both:
    print(f"solved by v3 and v2lbu: {len(both)}; median CPU to proof {np.median([A[k]['v3']['t_opt'] for k in both]):.1f} vs "
          f"{np.median([A[k]['v2lbu']['t_opt'] for k in both]):.1f} s")
ks = [k for k in KEYS if "v3" in A[k]]
print(f"v3 solved share: fresh {sum(A[k]['v3']['opt'] is not None for k in ks)}/{len(ks)} (development set, other host: 161/300)")
tim = [A[k]["tim"] for k in KEYS if "tim" in A[k]]
if tim:
    print("tim: statuses", dict(collections.Counter(r["status"] for r in tim)), "; CPU max", round(max(r["cpu"] for r in tim), 1),
          "; wall max", round(max(r["wall"] for r in tim), 1))
mx = {arm: max((A[k][arm]["cpu"] for k in KEYS if arm in A[k]), default=0) for arm in ("v3", "v2lbu", "onone", "tim")}
print("max CPU per arm (limit", TL, "):", {a: round(v, 1) for a, v in mx.items()})

print("\n== Exploratory ablation (unadjusted; same instances as each ablation arm) ==")
for ab in ("v3norev", "v3nofwd", "v3noht"):
    ks = [k for k in KEYS if ab in A[k] and "v3" in A[k]]
    if not ks: print(ab, "not run"); continue
    sa = [A[k]["v3"]["opt"] is not None for k in ks]; sb = [A[k][ab]["opt"] is not None for k in ks]
    bb = sum(x and not y for x, y in zip(sa, sb)); cc = sum(y and not x for x, y in zip(sa, sb))
    d = [gap(A[k]["v3"]) - gap(A[k][ab]) for k in ks]; m, lo, hi, p = boot(d)
    print(f"v3 vs {ab}: n={len(ks)} solved {sum(sa)} vs {sum(sb)} (only v3 {bb}, only {ab} {cc}, McNemar p={binom_two_sided(bb, cc):.3g}); "
          f"gap diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}] pp")

# ---- Analysis rule R1 (host change, written before the analysis): host stratum ----
print("\n== R1: host stratum (paired tests per host; level endpoints per host only) ==")
HOST = {k: next(iter({r["host"] for r in A[k].values()}), None) for k in A}
mixed = [k for k in A if len({r["host"] for r in A[k].values()}) > 1]
print("instances with arms on >1 host (should be 0):", mixed)
for h in sorted({v for v in HOST.values() if v}):
    print(f"-- host {h}: {sum(1 for v in HOST.values() if v == h)} instances")
    for a, b, ks in CONF:
        ks = [k for k in ks if HOST.get(k) == h and a in A[k] and b in A[k]]
        if not ks: continue
        sa = [A[k][a]["opt"] is not None for k in ks]; sb = [A[k][b]["opt"] is not None for k in ks]
        bb = sum(x and not y for x, y in zip(sa, sb)); cc = sum(y and not x for x, y in zip(sa, sb))
        m, lo, hi, p = boot([gap(A[k][a]) - gap(A[k][b]) for k in ks])
        print(f"   {a} vs {b}: n={len(ks)} solved {sum(sa)} vs {sum(sb)} (only {a} {bb}, only {b} {cc}; McNemar p={binom_two_sided(bb, cc):.3g}); "
              f"gap diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}] pp")
    for n1 in (20, 30):
        ks = [k for k in KEYS if HOST.get(k) == h and N1[k] == n1 and "v3" in A[k]]
        if ks: print(f"   v3 solved share n1={n1}: {sum(A[k]['v3']['opt'] is not None for k in ks)}/{len(ks)}")
