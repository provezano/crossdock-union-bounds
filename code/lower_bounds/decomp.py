"""Decomposition: does the union bound beat max(known bounds)? Marginal gain of reversal. usage: decomp.py DIR"""
import json,glob,sys,numpy as np,collections
L=[json.load(open(f)) for f in glob.glob(sys.argv[1]+'/*.json')]
def rep(L,tag):
    known=np.array([np.ceil(max(r['lb_pch'],r['lb_e2'],r['lb_load'],r['lb_hgr'])-1e-9) for r in L])
    kno=np.array([np.ceil(max(r['lb_pch'],r['lb_e2'],r['lb_load'])-1e-9) for r in L])  # without Lagrangian
    fw=np.array([np.ceil(r['u1_raw']-1e-9) for r in L]); rv=np.array([np.ceil(r.get('rev_raw',0)-1e-9) for r in L])
    ub=np.array([r['ub_hgr'] for r in L]); lh=np.array([r['lb_hgr'] for r in L])
    both=np.maximum(fw,rv); new=np.maximum(known,both)
    assert (new<=ub+1e-9).all()
    s=lambda a,b: f"{100*np.mean(a>b):5.1f}% +{100*np.mean((np.maximum(a,b)-b)/b):5.2f}%"
    print(f"{tag:8s} n={len(L):5d} U>known {s(fw,known)} | U>known(noLagr) {s(fw,kno)} | rev adds {s(both,np.maximum(fw,known))} | gapHgR pub(own LB,/LB) {100*np.mean((ub-lh)/lh):5.1f} vs new {100*np.mean((ub-new)/new):5.1f} | exact {np.mean([r['u1_exact'] for r in L]):.2f}")
    return new
rep(L,'all')
for k in ('n1','g'):
    G=collections.defaultdict(list)
    for r in L: G[r[k]].append(r)
    for g in sorted(G): rep(G[g],f"{k}={g}")
