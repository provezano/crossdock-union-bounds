"""Census gain over ceil(best non-own = max(L_CS, PCH)) with the forward bound = CPU-capped (paper) vs deterministic."""
import json, glob, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
R = {json.load(open(f))['key']: json.load(open(f)) for f in glob.glob(f"{D}/out/*.json")}
G = {f"{r['g']}_{r['n1']}_{r['seed']}": r for r in (json.load(open(f)) for f in glob.glob(f"{D}/../../lower_bounds/out_full/*.json"))}
ks = sorted(R); no = np.ceil(np.array([max(G[k]['lb_hgr'], G[k]['lb_pch']) for k in ks]) - 1e-9)
cap = np.array([G[k]['lb_u1'] for k in ks]); det = np.array([R[k]['lb_det'] for k in ks]); n1 = np.array([G[k]['n1'] for k in ks])
for lab, v in (("capped (paper)", cap), ("deterministic", det)):
    g = (v - no) / no * 100
    print(f"{lab:16s} forward LB_U* gain over ceil(max(L_CS, PCH)): all {g.mean():.2f}%  n1>=60 {g[n1>=60].mean():.2f}%  " + " ".join(f"n1={k}:{g[n1==k].mean():.2f}" for k in range(20, 81, 10)))
