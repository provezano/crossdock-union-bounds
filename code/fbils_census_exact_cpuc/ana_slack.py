"""Error analysis of LB* on the solved step 1 instances: FF (fully fluid DP) vs LB* vs OPT, by group and features."""
import json, os, subprocess, tempfile, sys, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); 
import solved_set
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
def path(g, n1, s): return f"{D}/../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{s}.dat"
def ff(fp): return float(subprocess.run([f"{D}/c/ff", fp], capture_output=True, text=True).stdout.split()[0])
def write_rev(fp, out):
    t = open(fp).read().split(); n1, n2, m1, m2 = map(int, t[:4]); p1 = t[4:4+n1]; p2 = t[4+n1:4+n1+n2]
    P = np.array(t[4+n1+n2:4+n1+n2+n1*n2], int).reshape(n2, n1)
    open(out, "w").write(f"{n2} {n1} {m2} {m1}\n" + " ".join(p2) + "\n" + " ".join(p1) + "\n" + "\n".join(" ".join(map(str, r)) for r in P.T) + "\n")
O = solved_set.load(); rows = []
tmp = tempfile.mkdtemp()
for k, v in sorted(O.items()):
    if v["opt"] is None: continue
    fp = path(v["g"], v["n1"], v["seed"]); a = ff(fp); rv = f"{tmp}/r.dat"; write_rev(fp, rv); b = ff(rv)
    rows.append(dict(key=k, g=v["g"], m1=v["m1"], m2=v["m2"], n2=v["n2"], opt=v["opt"], lbs=v["lbu"], u1=v["lb_u1"], rev=v["lb_rev"],
                     ff=a, ffr=b, ffc=float(np.ceil(max(a, b) - 1e-9))))
json.dump(rows, open(f"{D}/slack_solved.json", "w"))
g = lambda x, y: np.mean([(r[x] - r[y]) / r["opt"] * 100 for r in R])
for name, R in [("all", rows)] + [(gg, [r for r in rows if r["g"] == gg]) for gg in F]:
    if not R: continue
    print(f"{name:5s} n={len(R):2d}  OPT-LB* {g('opt','lbs'):.2f}%  OPT-FF {np.mean([(r['opt']-r['ffc'])/r['opt']*100 for r in R]):.2f}%  "
          f"FF>LB* in {np.mean([r['ffc']>r['lbs'] for r in R])*100:.0f}%  max(FF,LB*) gap {np.mean([(r['opt']-max(r['ffc'],r['lbs']))/r['opt']*100 for r in R]):.2f}%  "
          f"|FF-FFrev| max {max(abs(r['ff']-r['ffr']) for r in R):.3f}")
