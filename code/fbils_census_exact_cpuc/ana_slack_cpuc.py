"""Item 2 error analysis on the ENLARGED solved set (step 1 + exact v3 optima; n1 = 20 only, FF subset DP needs n1 <= 26):
FF (fully fluid DP, both directions) vs LB* vs OPT, by group. Helpers copied from ana_slack.py (importing it would rerun it). usage: ana_slack_cpuc.py [TXT]"""
import json, glob, os, sys, tempfile, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import solved_set
import subprocess
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
def path(g, n1, s): return f"{D}/../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{s}.dat"
def ff(fp): return float(subprocess.run([f"{D}/c/ff", fp], capture_output=True, text=True).stdout.split()[0])
def write_rev(fp, out):
    t = open(fp).read().split(); n1, n2, m1, m2 = map(int, t[:4]); p1 = t[4:4+n1]; p2 = t[4+n1:4+n1+n2]
    P = np.array(t[4+n1+n2:4+n1+n2+n1*n2], int).reshape(n2, n1)
    open(out, "w").write(f"{n2} {n1} {m2} {m1}\n" + " ".join(p2) + "\n" + " ".join(p1) + "\n" + "\n".join(" ".join(map(str, r)) for r in P.T) + "\n")
O = solved_set.load(); opt = {k: v["opt"] for k, v in O.items() if v["opt"] is not None}
for fp in glob.glob(f"{D}/out_exact_v3/*_tl300.json"):
    a = json.load(open(fp))
    if a["opt"] is not None: opt[a["key"]] = a["opt"]
tmp = tempfile.mkdtemp(); rows = []
for k, o in sorted(opt.items()):
    v = O[k]
    if v["n1"] > 26: continue
    fp = path(v["g"], v["n1"], v["seed"]); a = ff(fp); rv = f"{tmp}/r.dat"; write_rev(fp, rv); b = ff(rv)
    rows.append(dict(key=k, g=v["g"], opt=o, lbs=v["lbu"], ffc=float(np.ceil(max(a, b) - 1e-9))))
out = []
for name, R in [("all", rows)] + [(gg, [r for r in rows if r["g"] == gg]) for gg in F]:
    if not R: continue
    m = lambda f: np.mean([f(r) for r in R])
    out.append(f"{name:5s} n={len(R):2d}  OPT-LB* {m(lambda r:(r['opt']-r['lbs'])/r['opt']*100):.2f}%  OPT-FF {m(lambda r:(r['opt']-r['ffc'])/r['opt']*100):.2f}%  "
               f"FF>LB* {m(lambda r: r['ffc'] > r['lbs'])*100:.0f}%  OPT-max(FF,LB*) {m(lambda r:(r['opt']-max(r['ffc'], r['lbs']))/r['opt']*100):.2f}%  max=OPT on {sum(max(r['ffc'], r['lbs']) >= r['opt'] for r in R)}")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
