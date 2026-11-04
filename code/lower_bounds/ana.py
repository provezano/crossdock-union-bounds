import json,glob,sys,numpy as np,collections
d=sys.argv[1]; L=[json.load(open(f)) for f in glob.glob(d+'/*.json')]
def row(L,tag):
    g0=np.array([r['lb_union'] for r in L]); u=np.array([r['lb_u1'] for r in L]); ub=np.array([r['ub_hgr'] for r in L]); lh=np.array([r['lb_hgr'] for r in L])
    assert (u<=ub+1e-9).all(), 'LB>UB!'
    print(f"{tag:8s} n={len(L):4d} gain%={100*np.mean((u-g0)/g0):6.2f} (max {100*np.max((u-g0)/g0):5.2f}, worse {np.sum(u<g0-1e-6)}) gapHgR_vsG0={100*np.mean((ub-g0)/g0):5.1f} vsNew={100*np.mean((ub-u)/u):5.1f} pub={100*np.mean((ub-lh)/lh):5.1f} exact={np.mean([r['u1_exact'] for r in L]):.2f} cpu={np.mean([r['u1_cpu'] for r in L]):.1f}/{np.max([r['u1_cpu'] for r in L]):.0f}")
row(L,'all')
for k in ('n1','g'):
    G=collections.defaultdict(list)
    for r in L: G[r[k]].append(r)
    for g in sorted(G): row(G[g],f"{k}={g}")
