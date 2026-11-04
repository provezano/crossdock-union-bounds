"""Exploratory analysis (see the paper, Section 8.9): per-instance margins of base ILS vs
static EACDC (g4_match protocol) on the structured matched subset (58) and the first confirmation set (210), related to
headroom, binding term of LB*, Theorem t9 residual, density and raw features. Writes per-instance table + text.
usage: ana_margin.py OUTJSON OUTTXT"""
import sys, os, json, glob, math, collections, numpy as np
F = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, f"{F}/confirmation_eacdc")
from ana_lib import at_cpu, at_eval
def parse(p):
    t = np.array(open(p).read().split(), dtype=np.int64); n1, n2, m1, m2 = map(int, t[:4])
    p1 = t[4:4+n1]; p2 = t[4+n1:4+n1+n2]; P = t[4+n1+n2:4+n1+n2+n1*n2].reshape(n2, n1).astype(bool)
    return n1, n2, m1, m2, p1, p2, P
rows = []
def add(src, key, fam, cell, path, L, lbstar, runs):
    n1, n2, m1, m2, p1, p2, P = parse(path)
    mc, me, ue, ui, cpe, cpi, eve = [], [], [], [], [], [], []
    for r in runs:
        e, i = r["eacdc"], r["ils"]
        if e["cpu"] > 600: continue
        a, b = at_cpu(i["trace"], e["cpu"]), at_eval(i["trace"], e["n_evalpop"])
        mc.append((a - e["best"]) / e["best"] * 100); me.append((b - e["best"]) / e["best"] * 100)
        ue.append(e["best"]); ui.append(min(a, b)); cpe.append(e["cpu"] / (e["n_init"] + e["n_evalpop"])); cpi.append(i["cpu"] / i["nev"])
        eve.append(e["n_init"] + e["n_evalpop"])
    if len(mc) < 3: return
    a_ = p1.sum() / m1; b_ = p2.sum() / m2
    cls = max(L["lb_cs"], L["lb_pch"], L["lb_e2"], L["lb_load"], p1.max(), p2.max())
    ccls = math.ceil(cls - 1e-9)
    bind = "union" if lbstar > ccls + 1e-9 else ("cs" if math.ceil(L["lb_cs"] - 1e-9) >= ccls else "other")
    R = (a_ + b_ + (1 - 1 / m1) * p1.max() + (1 - 1 / m2) * p2.max()) / lbstar - 1
    rows.append(dict(src=src, key=key, fam=fam, cell=cell, n1=n1, n2=n2, m1=m1, m2=m2, dens=float(P.mean()),
        m_cpu=float(np.mean(mc)), m_ev=float(np.mean(me)), ue=float(np.mean(ue)), ub_best=float(min(min(ue), min(ui))),
        lbstar=float(lbstar), h_E=(np.mean(ue) - lbstar) / lbstar * 100, h_best=(min(min(ue), min(ui)) - lbstar) / lbstar * 100,
        gainU=(lbstar - math.ceil(max(L["lb_pch"], L["lb_cs"]) - 1e-9)) / lbstar * 100, bind=bind, R=float(R),
        rho1=a_ / lbstar, rho2=b_ / lbstar, cv1=float(p1.std() / p1.mean()), cv2=float(p2.std() / p2.mean()),
        lmr=math.log(m1 / m2), cpu_per_ev_E=float(np.mean(cpe)), cpu_per_ev_I=float(np.mean(cpi)), ev_E=float(np.mean(eve)),
        host=runs[0]["env"]["host"]))
G6 = f"{F}/fbils_census_exact_cpuc"; S = {json.load(open(f))["key"]: json.load(open(f)) for f in glob.glob(f"{G6}/out_struct/*.json")}
M = collections.defaultdict(list)
for f in glob.glob(f"{G6}/out_match_struct/*.json"): r = json.load(open(f)); M[r["key"]].append(r)
for k, rs in M.items():
    s = S[k]; add("struct", k, s["fam"], s["cell"], s["path"], s, s["lbstar"], rs)
G4 = f"{F}/confirmation_eacdc"; M = collections.defaultdict(list)
for f in glob.glob(f"{G4}/out_new/*.json"): r = json.load(open(f)); M[r["key"]].append(r)
for k, rs in M.items():
    L = json.load(open(f"{G4}/out_lbnew/{k}.json")); add("bench", k, "bench", L["g"], f"{G4}/{L['path']}", L, max(L["lb_u1"], L["lb_rev"]), rs)
json.dump(rows, open(sys.argv[1], "w"))
out = []; P_ = out.append
def boot(x):
    x = np.asarray(x); r = np.random.default_rng(0); b = x[r.integers(0, len(x), (5000, len(x)))].mean(1)
    return f"{x.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}] (n={len(x)})"
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    a, b = rk(x), rk(y); r = np.corrcoef(a, b)[0, 1]; g = np.random.default_rng(1)
    nul = np.array([np.corrcoef(a, g.permutation(b))[0, 1] for _ in range(4000)]); p = (np.sum(np.abs(nul) >= abs(r)) + 1) / 4001
    return f"rho={r:+.2f} p={p:.1e}"
for src in ("bench", "struct"):
    R = [r for r in rows if r["src"] == src]
    P_(f"== {src}: m_cpu {boot([r['m_cpu'] for r in R])}  m_ev {boot([r['m_ev'] for r in R])}")
    P_(f"   h_E {np.mean([r['h_E'] for r in R]):.2f}%  h_best {np.mean([r['h_best'] for r in R]):.2f}%  gainU {np.mean([r['gainU'] for r in R]):.2f}%  dens {np.mean([r['dens'] for r in R]):.3f}  R {np.mean([r['R'] for r in R]):.3f}")
    P_(f"   binding: {dict(collections.Counter(r['bind'] for r in R))}")
    for b in ("union", "cs", "other"):
        q = [r for r in R if r["bind"] == b]
        if q: P_(f"     bind={b:5s}: m_cpu {boot([r['m_cpu'] for r in q])} m_ev {np.mean([r['m_ev'] for r in q]):+.3f}  h_E {np.mean([r['h_E'] for r in q]):.2f}%")
    for v in ("h_E", "h_best", "gainU", "R", "dens", "rho1", "rho2"):
        P_(f"   spearman m_cpu~{v}: {spear([r['m_cpu'] for r in R], [r[v] for r in R])};  m_ev~{v}: {spear([r['m_ev'] for r in R], [r[v] for r in R])}")
P_("== struct by cell (m_cpu, m_ev, h_E, h_best, gainU, dens, bind)")
by = collections.defaultdict(list)
for r in rows:
    if r["src"] == "struct": by[(r["cell"], r["n1"])].append(r)
for (c, n), q in sorted(by.items()):
    P_(f"   {c:6s} {n:3d}: {np.mean([r['m_cpu'] for r in q]):+.2f} {np.mean([r['m_ev'] for r in q]):+.2f}  h_E {np.mean([r['h_E'] for r in q]):5.2f} h_best {np.mean([r['h_best'] for r in q]):5.2f} gainU {np.mean([r['gainU'] for r in q]):5.2f} dens {np.mean([r['dens'] for r in q]):.2f} {[r['bind'] for r in q]} m1/m2 {[(r['m1'], r['m2']) for r in q]}")
P_("== bench by group,n1 (m_cpu, h_E)")
by = collections.defaultdict(list)
for r in rows:
    if r["src"] == "bench": by[(r["cell"], r["n1"])].append(r)
for (c, n), q in sorted(by.items()): P_(f"   {c:5s} {n:3d}: {np.mean([r['m_cpu'] for r in q]):+.2f} {np.mean([r['m_ev'] for r in q]):+.2f} h_E {np.mean([r['h_E'] for r in q]):5.2f} gainU {np.mean([r['gainU'] for r in q]):5.2f}")
# LOO regressions
def loo(X, y):
    X = np.column_stack([np.ones(len(y)), X]); n = len(y); H = X @ np.linalg.pinv(X.T @ X) @ X.T
    e = y - H @ y; e_loo = e / (1 - np.diag(H)); return 1 - (e_loo ** 2).sum() / ((y - y.mean()) ** 2).sum()
def feats(R, block):
    if block == "raw":
        return np.array([[r["dens"], r["lmr"], math.log(r["m1"]), math.log(r["m2"]), math.log(r["n1"]), r["cv1"], r["cv2"]] for r in R])
    if block == "theory":
        return np.array([[r["h_E"], r["gainU"], r["R"], r["rho1"], r["rho2"], r["bind"] == "union", r["bind"] == "cs"] for r in R], float)
    if block == "headroom": return np.array([[r["h_E"]] for r in R])
    if block == "theory-noh":
        return np.array([[r["gainU"], r["R"], r["rho1"], r["rho2"], r["bind"] == "union", r["bind"] == "cs"] for r in R], float)
for nm, R in (("pooled", rows), ("bench", [r for r in rows if r["src"] == "bench"]), ("struct", [r for r in rows if r["src"] == "struct"])):
    for yv in ("m_cpu", "m_ev"):
        y = np.array([r[yv] for r in R]); res = {}
        for b in ("raw", "theory", "headroom", "theory-noh"): res[b] = loo(feats(R, b), y)
        res["both"] = loo(np.column_stack([feats(R, "raw"), feats(R, "theory")]), y)
        P_(f"LOO R2 {nm:6s} {yv}: " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()) + f"  | partial: theory|raw={res['both']-res['raw']:+.3f} raw|theory={res['both']-res['theory']:+.3f}")
# density addendum (coordinator): per-decode CPU, struct only, single host machine-D
R = [r for r in rows if r["src"] == "struct"]
P_("== density addendum (struct, host " + str(collections.Counter(r['host'] for r in R)) + ")")
for lo, hi in ((0, 0.12), (0.12, 0.25), (0.25, 0.4), (0.4, 1)):
    q = [r for r in R if lo <= r["dens"] < hi]
    if q: P_(f"   dens [{lo},{hi}): n={len(q)} m_cpu {np.mean([r['m_cpu'] for r in q]):+.2f} m_ev {np.mean([r['m_ev'] for r in q]):+.2f} gap(cpu-ev) {np.mean([r['m_cpu']-r['m_ev'] for r in q]):+.2f}  CPU/eval EACDC {np.mean([r['cpu_per_ev_E'] for r in q])*1e6:.0f} us  ILS {np.mean([r['cpu_per_ev_I'] for r in q])*1e6:.0f} us  ratio {np.mean([r['cpu_per_ev_E']/r['cpu_per_ev_I'] for r in q]):.2f}")
for v in ("m_cpu", "m_ev"): P_(f"   spearman {v}~dens: {spear([r[v] for r in R], [r['dens'] for r in R])}")
P_(f"   spearman (m_cpu-m_ev)~dens: {spear([r['m_cpu']-r['m_ev'] for r in R], [r['dens'] for r in R])}; ratio~dens: {spear([r['cpu_per_ev_E']/r['cpu_per_ev_I'] for r in R], [r['dens'] for r in R])}")
for n in (30, 60, 100, 150):
    q = [r for r in R if r["n1"] == n]
    P_(f"   n1={n}: ratio CPU/eval EACDC/ILS by density: " + " ".join(f"{r['dens']:.2f}:{r['cpu_per_ev_E']/r['cpu_per_ev_I']:.2f}" for r in sorted(q, key=lambda r: r['dens'])))
txt = "\n".join(out); print(txt); open(sys.argv[2], "w").write(txt + "\n")
