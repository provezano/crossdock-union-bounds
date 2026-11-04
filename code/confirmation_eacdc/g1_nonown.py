"""E: recompute the step 1 LB gain vs the best NON-OWN published LB = max(lb_hgr (Chen-Song as reported by HgR), lb_pch),
and vs lb_hgr alone; raw and with the baseline ceiled (lb_u1 is ceiled). Distribution, not bootstrap (census)."""
import json, glob, math, numpy as np
R = [json.load(open(f)) for f in glob.glob(f"{__import__('os').path.dirname(__file__) or '.'}/../lower_bounds/out_full/*.json")]
L = np.array([max(r['lb_u1'], r['lb_rev']) for r in R]); U1 = np.array([r['lb_u1'] for r in R])
def rep(name, base, lb=U1):
    g = (lb - base) / base * 100
    q = np.percentile(g, [0, 10, 50, 90, 100])
    print(f"{name:52s} n={len(g)} mean {g.mean():.2f}%  q0/10/50/90/100 {' / '.join(f'{x:.2f}' for x in q)}  frac<0 {np.mean(g < -1e-9):.4f}")
for lab, lb in (("LB_U*", U1), ("max(LB_U*, reversed)", L)):
    print("==", lab)
    no = np.array([max(r['lb_hgr'], r['lb_pch']) for r in R]); hg = np.array([r['lb_hgr'] for r in R]); lo = np.array([r['lo_cheap'] for r in R])
    rep("vs best non-own max(lb_hgr, lb_pch) raw", no, lb); rep("vs ceil(best non-own)", np.ceil(no - 1e-9), lb)
    rep("vs lb_hgr only raw", hg, lb); rep("vs ceil(lb_hgr)", np.ceil(hg - 1e-9), lb); rep("vs old baseline lo_cheap (incl. own e2/load)", lo, lb)
print("lb_pch > lb_hgr in", np.mean([r['lb_pch'] > r['lb_hgr'] + 1e-9 for r in R]))
