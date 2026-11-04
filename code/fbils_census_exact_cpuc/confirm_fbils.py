"""Preregistered run: per (instance, seed) in ONE process: EACDC full run first, then v1 (Python
ILS), v2 (ILS-FBI, C) and IG (C) in rotated order; every search runs until it used EACDC's CPU AND its evaluation count
(init + evalPop) and is read at both. usage: confirm_fbils.py LIST OUT SEEDS SHARD NSH"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, f"{D}/../confirmation_eacdc"); sys.path.insert(0, f"{D}/../exact_cpsat")
import g4_match, srch, g5lib as L  # noqa
g2 = L.g2
def at(tr, idx, lim):
    v = [x[-1] for x in tr if x[idx] <= lim + 1e-9]; return min(v) if v else tr[0][-1]
def c_arm(I, kind, cpu_b, ev_b, seed):
    t0 = time.process_time(); C = g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); side, order = C[kb][1], C[kb][2]
    J = I if side == "f" else g2.reverse(I); tcon = time.process_time() - t0
    if kind == "v2":
        b, bo, tr, ne = srch.run_fbi(J, g2.reverse(J), order, max(0.0, cpu_b - tcon), ev_b, cap=3 * cpu_b + 2, seed=100 + seed,
                                     nev0=len(C), pfbi=0.0, stall_mult=40, rounds=3, wfrac=0.25)
    else:
        b, bo, tr, ne = srch.run(J, order, "ig", max(0.0, cpu_b - tcon), ev_b, cap=3 * cpu_b + 2, seed=100 + seed, nev0=len(C), ip1=4, fp1=0.4, fp2=1)
    assert g2.Dec(J)(bo) == b
    cm, s1, s2 = g2.schedule_from(I, side, bo); assert abs(L.cdc.check_schedule(I, s1, s2) - b) < 1e-6
    tr = [(c + tcon, e, v) for c, e, v in tr]
    return dict(best=float(b), v_cpu=at(tr, 0, cpu_b), v_ev=at(tr, 1, ev_b), nev=ne, tcon=tcon)
def main():
    lst, out, seeds, sh, ns = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4]), int(sys.argv[5])
    os.makedirs(out, exist_ok=True); Lst = json.load(open(lst)); jobs = [(r, s) for s in seeds for r in Lst]
    env = dict(python=sys.version.split()[0], host=platform.node())
    for q, (r, s) in enumerate(jobs):
        if q % ns != sh: continue
        fo = f"{out}/{r['key']}_s{s}.json"
        if os.path.exists(fo): continue
        fp = r["path"] if os.path.isabs(r["path"]) else f"{D}/{r['path']}"; I = L.cdc.parse(fp)
        e = g4_match.run_eacdc(fp, s); cpu_b = e["cpu"]; ev_b = e["n_init"] + e["n_evalpop"]
        res = {}; arms = ["v1", "v2", "ig"]; k = q % 3; arms = arms[k:] + arms[:k]
        for a in arms:
            if a == "v1":
                i = g4_match.run_ils(I, cpu_b, ev_b, s)
                res[a] = dict(best=i["best"], v_cpu=at([(c, n, v) for c, n, v in i["trace"]], 0, cpu_b),
                              v_ev=at([(c, n, v) for c, n, v in i["trace"]], 1, ev_b), nev=i["nev"])
            else: res[a] = c_arm(I, a, cpu_b, ev_b, s)
        json.dump(dict(key=r["key"], seed=s, order=arms, eacdc=dict(best=e["best"], cpu=cpu_b, ev=ev_b), **res, env=env),
                  open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
    print("done", flush=True)
if __name__ == "__main__":
    main()
