"""Exploratory: FB-ILS (frozen v2 config of confirm_fbils.py) on the 58 structured matched instances, seeds 11-13, read at
the evaluation count (n_init + n_evalpop) of the EXISTING EACDC run of the same instance and seed (out_match_struct).
Evaluation matching is machine-independent, so the EACDC runs need not be repeated; CPU matching is NOT done here.
usage: fb_struct.py OUT"""
import sys, os, json, glob, time
G6 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fbils_census_exact_cpuc"); sys.path.insert(0, G6); sys.path.insert(0, f"{G6}/../exact_cpsat")
import srch, g5lib as L
g2 = L.g2; out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def at(tr, idx, lim):
    v = [x[-1] for x in tr if x[idx] <= lim + 1e-9]; return min(v) if v else tr[0][-1]
for f in sorted(glob.glob(f"{G6}/out_match_struct/*.json")):
    r = json.load(open(f)); fo = f"{out}/{os.path.basename(f)}"
    if os.path.exists(fo): continue
    e = r["eacdc"]; ev_b = e["n_init"] + e["n_evalpop"]; s = r["seed"]
    I = L.cdc.parse(f"{G6}/data_struct/{r['key']}.dat")
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I)
    b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, 0.0, ev_b, cap=600, seed=100 + s, nev0=len(C), pfbi=0.0,
                                 stall_mult=40, rounds=3, wfrac=0.25)
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(L.cdc.check_schedule(I, s1, s2) - b) < 1e-6
    json.dump(dict(key=r["key"], seed=s, eacdc_best=e["best"], ev_b=ev_b, v2_ev=at(tr, 1, ev_b), v2_best=b, nev=ne,
                   cpu=time.process_time() - t0), open(fo, "w"))
print("done")
