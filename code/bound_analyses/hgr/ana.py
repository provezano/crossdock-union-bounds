"""Analysis of the HgR fidelity and matched runs -> out/ana.txt (and stdout).
Relative difference d = (A - B) / B per instance; mean with 95% percentile bootstrap CI (10,000 resamples over
instances, rng 0); W/T/L counts A<B / A=B / A>B."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); O = f"{H}/out"
G = ["m2", "m4", "m10", "u24", "u210"]
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); lines.append(s); print(s)

def load(fn):
    R = {}
    if not os.path.exists(f"{O}/{fn}"): return R
    for ln in open(f"{O}/{fn}"):
        d = json.loads(ln); R[(d["key"], d["method"])] = d
    return R

def boot(d, B=10000):
    d = np.asarray(d, float); rng = np.random.default_rng(0)
    m = rng.choice(d, size=(B, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def summ(label, a, b, keys):
    a = np.array(a, float); b = np.array(b, float); d = 100 * (a - b) / b
    mu, lo, hi = boot(d)
    P(f"{label:38s} n={len(d):3d}  mean {mu:+6.2f}% [{lo:+.2f}, {hi:+.2f}]  W/T/L {int((a<b).sum())}/{int((a==b).sum())}/{int((a>b).sum())}")
    return d

def table(title, rows, fa, fb):
    P(f"\n== {title}")
    summ("overall", [fa(r) for r in rows], [fb(r) for r in rows], None)
    for g in G:
        s = [r for r in rows if r["g"] == g]
        if s: summ(f"  group {g}", [fa(r) for r in s], [fb(r) for r in s], None)
    for n1 in range(20, 90, 10):
        s = [r for r in rows if r["n1"] == n1]
        if s: summ(f"  n1={n1}", [fa(r) for r in s], [fb(r) for r in s], None)

# ------------------------------------------------------------------ fidelity
F35 = load("fid_code35.jsonl"); F105 = load("fid_code105_nolp.jsonl")
for name, R, meth in (("code mode, LP (35 inst., cap min(pub time,120 s))", F35, "hgr"),
                      ("code mode, no LP (105 inst., cap min(pub time,120 s))", F105, "hgrnolp")):
    rows = [d for (k, m), d in R.items() if m == meth]
    if not rows: continue
    P(f"\n######## FIDELITY: {name}")
    table("ours min(H1,H2) vs published min(H1,H2)", rows, lambda r: r["ub"], lambda r: min(r["pub_h1"], r["pub_h2"]))
    P(""); summ("ours H1 (stage-2 UB) vs published H1", [r["ub1"] for r in rows], [r["pub_h1"] for r in rows], None)
    summ("ours H2 (stage-2 UB) vs published H2", [r["ub2"] for r in rows], [r["pub_h2"] for r in rows], None)
    cpu = np.array([r["cpu"] for r in rows]); pt = np.array([r["pub_time"] for r in rows])
    P(f"CPU ours median {np.median(cpu):.1f} s (max {cpu.max():.1f}); published Tempo_MD median {np.median(pt):.1f} s; "
      f"ours/published median ratio {np.median(cpu/pt):.2f}")
    st = {}
    for r in rows: st[r["stop"]] = st.get(r["stop"], 0) + 1
    P("stop reasons:", st)
    lpst = {}
    for r in rows:
        for x in r.get("lp", []): lpst[x["status"]] = lpst.get(x["status"], 0) + 1
    P("LP statuses:", lpst)

# ------------------------------------------------------------------ matched
for fn, B in (("match10.jsonl", 10), ("match60.jsonl", 60)):
    R = load(fn)
    if not R: continue
    keys = sorted({k for k, m in R})
    def get(m): return {k: R[(k, m)] for k in keys if (k, m) in R}
    hg, hn, i1, i2 = get("hgr"), get("hgrnolp"), get("ils1"), get("ils2")
    P(f"\n######## MATCHED {B} s CPU per instance ({fn}); d = (ILS - HgR)/HgR")
    for lab, A, Bm in (("ILS v1 vs HgR", i1, hg), ("ILS v2 vs HgR", i2, hg), ("ILS v1 vs HgR-noLP", i1, hn),
                       ("ILS v1 vs best(HgR, HgR-noLP)", i1, None), ("ILS v2 vs best(HgR, HgR-noLP)", i2, None)):
        ks = [k for k in keys if k in A and (k in Bm if Bm is not None else (k in hg and k in hn))]
        if not ks: continue
        rows = [dict(g=A[k]["g"], n1=A[k]["n1"], a=A[k]["ub"],
                     b=(Bm[k]["ub"] if Bm is not None else min(hg[k]["ub"], hn[k]["ub"]))) for k in ks]
        table(lab, rows, lambda r: r["a"], lambda r: r["b"])
    for lab, A in (("HgR", hg), ("HgR-noLP", hn)):
        if not A: continue
        rows = [dict(g=d["g"], n1=d["n1"], a=d["ub"], b=min(d["pub_h1"], d["pub_h2"])) for d in A.values()]
        P(""); summ(f"{lab} ({B} s, this machine) vs published HgR", [r["a"] for r in rows], [r["b"] for r in rows], None)
        cpu = np.array([d["cpu"] for d in A.values()]); st = {}
        for d in A.values(): st[d["stop"]] = st.get(d["stop"], 0) + 1
        P(f"  {lab}: CPU used median {np.median(cpu):.1f} s, stop reasons {st}")
    for lab, A in (("ILS v1", i1), ("ILS v2", i2)):
        if not A: continue
        rows = list(A.values())
        summ(f"{lab} ({B} s) vs published HgR", [d["ub"] for d in rows], [min(d["pub_h1"], d["pub_h2"]) for d in rows], None)
    allck = all(d["checked"] for d in R.values())
    P(f"all {len(R)} schedules passed cdc.check_schedule: {allck}")

open(f"{O}/ana.txt", "w").write("\n".join(lines) + "\n")
