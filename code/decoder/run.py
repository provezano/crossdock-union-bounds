"""Stage A: per instance constructive + ILS10 + LNS10 (warm from best constructive) + CP-SAT plain 10s.
Stage B (arg B): ILS60 and ILS30->LNS30 on the 60-subset. Args: stage shard nshards. Cache JSON per instance."""
import sys, os, json, time, resource, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import cdc, g2
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
stage, shard, nsh = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
S = json.load(open(f"{D}/../lower_bounds/sample105.json"))
if stage == "B":  # 60-subset: first 2 of the 3 per (group,n1) minus 10 dropped by fixed rng
    from collections import defaultdict
    gg = defaultdict(list)
    for r in S: gg[(r['g'], r['n1'])].append(r)
    S = [x for k in sorted(gg) for x in gg[k][:2]]
    rng = np.random.default_rng(2026110102); S = [S[k] for k in sorted(rng.choice(len(S), 60, replace=False))]
od = f"{D}/out_{stage}"; os.makedirs(od, exist_ok=True)
def chk(I, side, order, val):
    cm, s1, s2 = g2.schedule_from(I, side, order); assert abs(cdc.check_schedule(I, s1, s2) - val) < 1e-6; return s1, s2
for k, r in enumerate(S):
    if k % nsh != shard: continue
    fo = f"{od}/{r['g']}_{r['n1']}_{r['seed']}.json"
    if os.path.exists(fo): continue
    I = cdc.parse(f"{D}/../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
    out = dict(g=r['g'], n1=r['n1'], seed=r['seed'], n2=r['n2'], m1=r['m1'], m2=r['m2'], ub_hgr=r['ub_hgr'], ub_h1=r['ub_h1'])
    t = time.process_time(); C = g2.all_constructive(I); out['constr_cpu'] = time.process_time() - t
    out['constr'] = {k: v[0] for k, v in C.items()}
    kb = min(C, key=lambda k: C[k][0]); side, o0 = C[kb][1], C[kb][2]; out['constr_best'] = kb
    if stage == "A":
        b, bo, tr = g2.ils(I, side, o0, 10.0); s1, s2 = chk(I, side, bo, b); out['ils10'] = b; out['ils10_tr'] = tr
        cm, s1, s2 = g2.schedule_from(I, side, o0)
        b, s1, s2, tr = g2.lns(I, s1, s2, 10.0); assert abs(cdc.check_schedule(I, s1, s2) - b) < 1e-6; out['lns10'] = b; out['lns10_tr'] = tr
        t = time.process_time(); res = cdc.cpsat(I, 10.0); out['cp10'] = res['ub']; out['cp10_lb'] = res['lb']; out['cp10_cpu'] = time.process_time() - t
        if res['sched']: assert abs(cdc.check_schedule(I, *res['sched']) - res['ub']) < 1e-6
    else:
        b, bo, tr = g2.ils(I, side, o0, 60.0, seed=1); chk(I, side, bo, b); out['ils60'] = b; out['ils60_tr'] = tr
        pass
    out['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    json.dump(out, open(fo, "w")); print(fo, out.get('ils10', out.get('ils60')), r['ub_hgr'], flush=True)
