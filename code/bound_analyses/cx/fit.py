"""Log-log slope fits (OLS, 95% CI from t quantile) of measured CPU / memory against n1 and against the dominant
size term n1*n2 + n2^2 (benchmark: |E| ~ n1 n2 / 2, n2 ~ n1). Reads cx.json (+ cx2.json, LP/det census if present)."""
import json, os, glob, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
def fit(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float)); n = len(x)
    b, a = np.polyfit(x, y, 1); res = y - (a + b * x); se = np.sqrt(res @ res / (n - 2) / ((x - x.mean()) @ (x - x.mean())))
    h = 2.0 * se  # t_{.975, n-2} ~ 2.0 for n ~ 40
    return b, b - h, b + h
R = json.load(open(f"{D}/cx.json"))
n1 = [r['n1'] for r in R]; S = [r['n1'] * r['n2'] + r['n2'] ** 2 for r in R]
print("n =", len(R), "instances (m4, m10; n1 = 20..80; 3 each)")
for k in ("dec_py", "ev_fast", "ev_orig", "constr", "ug1", "ga_var", "lp_hull", "lcs", "mem_dec_py", "mem_ev_fast", "mem_ev_orig", "mem_constr"):
    a = fit(n1, [r[k] for r in R]); b = fit(S, [r[k] for r in R])
    print(f"{k:12s} slope vs n1 {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]   vs n1n2+n2^2 {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]   at n1=80 mean {np.mean([r[k] for r in R if r['n1']==80]):.3g}")
for nn in (20, 50, 80):
    X = [r for r in R if r['n1'] == nn]
    print(f"n1={nn}: ev_fast/dec_py {np.mean([r['ev_fast']/r['dec_py'] for r in X]):.2f}  ev_orig/dec_py {np.mean([r['ev_orig']/r['dec_py'] for r in X]):.1f}  "
          f"constr in ev_fast units {np.mean([r['constr']/r['ev_fast'] for r in X]):.0f}  GA non-eval share of a generation {np.mean([r['ga_var']/(r['ga_var']+100*r['ev_fast']) for r in X]):.2f}")
if os.path.exists(f"{D}/cx2.json"):
    C = json.load(open(f"{D}/cx2.json")); m = [r['n1'] for r in C]; S2 = [r['n1'] * r['n2'] + r['n2'] ** 2 for r in C]
    for k in ("c_ils", "c_v2"):
        a = fit(m, [r[k] for r in C]); b = fit(S2, [r[k] for r in C])
        print(f"{k:12s} slope vs n1 {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]   vs n1n2+n2^2 {b[0]:.2f} [{b[1]:.2f}, {b[2]:.2f}]  at n1=80 {np.mean([r[k] for r in C if r['n1']==80])*1e6:.2f} us")
    K = {(r['g'], r['n1'], r['n2']): r for r in C}
    for nn in (20, 50, 80):
        X = [(r, K[(r['g'], r['n1'], r['n2'])]) for r in R if r['n1'] == nn and (r['g'], r['n1'], r['n2']) in K]
        print(f"n1={nn}: ev_fast / C-v2-decode {np.mean([a['ev_fast']/b['c_v2'] for a, b in X]):.0f}x  ev_orig / C-v2 {np.mean([a['ev_orig']/b['c_v2'] for a, b in X]):.0f}x")
