"""Equal-budget Optuna tuning of ILS vs EACDC on a TRAIN split disjoint from sample105; then test eval.
usage: tune.py METHOD NTRIALS  -> out_tune/METHOD_trials.json ; tune.py test METHOD WHICH SHARD NSH
All runs: 4 s CPU (EACDC: anytime truncation + restarts), objective = mean UB/UB_HgR over train."""
import sys, os, json, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, g3, cdc
B = 4.0
A = json.load(open(f"{D}/../lower_bounds/allbounds.json"))
S105 = {(r['g'], r['n1'], r['seed']) for r in json.load(open(f"{D}/../lower_bounds/sample105.json"))}
rng = np.random.default_rng(11); TR = []
for g in ('m2', 'm4', 'm10', 'u24', 'u210'):
    for n1 in (20, 40, 50, 60, 80):
        c = [r for r in A if r['g'] == g and r['n1'] == n1 and (g, n1, r['seed']) not in S105]
        TR.append(c[int(rng.integers(len(c)))])
OD = f"{D}/out_tune"; os.makedirs(OD, exist_ok=True)

def run(method, prm, r, seed):
    if method == "ils":
        I = cdc.parse(g3.path(r['g'], r['n1'], r['seed'])); t0 = time.process_time()
        C = g3.g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); tc = time.process_time() - t0
        b, _, _ = g3.ils_p(I, C[kb][1], C[kb][2], B - tc, seed=100 + seed, **prm); return b
    p = (prm['ts'], prm['ngen'], prm['mu'], prm['lam'], prm['cx'], prm['mut'])
    return g3.eacdc(r['g'], r['n1'], r['seed'], seed, budget=B, params=p)["best"]

DEF = {"ils": dict(stall_mult=40, n_pert=2, seg_rev=True),
       "eacdc": dict(ts=3, ngen=150, mu=75, lam=100, cx=0.95, mut=0.4)}

def space(method, t):
    if method == "ils":
        return dict(stall_mult=t.suggest_int("stall_mult", 5, 200, log=True), n_pert=t.suggest_int("n_pert", 0, 6),
                    seg_rev=t.suggest_categorical("seg_rev", [True, False]))
    return dict(ts=t.suggest_int("ts", 2, 10), ngen=t.suggest_int("ngen", 80, 200), mu=t.suggest_int("mu", 25, 200),
                lam=t.suggest_int("lam", 35, 200), cx=t.suggest_float("cx", 0.1, 0.95), mut=t.suggest_float("mut", 0.1, 0.9))

if sys.argv[1] != "test":
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    method, nt = sys.argv[1], int(sys.argv[2]); rec = []
    def obj(t):
        prm = space(method, t); t0 = time.process_time()
        v = float(np.mean([run(method, prm, r, t.number) / r['ub_hgr'] for r in TR]))
        rec.append(dict(n=t.number, prm=prm, v=v, cpu=time.process_time() - t0))
        json.dump(rec, open(f"{OD}/{method}_trials.json", "w")); return v
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=12))
    st.enqueue_trial({k: v for k, v in DEF[method].items()})
    st.optimize(obj, n_trials=nt)
    json.dump(dict(best=st.best_params, best_v=st.best_value, train=[(r['g'], r['n1'], r['seed']) for r in TR]),
              open(f"{OD}/{method}_best.json", "w"))
    print("done")
else:
    method, which, sh, ns = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    prm = DEF[method] if which == "default" else json.load(open(f"{OD}/{method}_best.json"))["best"]
    L = json.load(open(f"{D}/../lower_bounds/sample105.json")); od = f"{OD}/test_{method}_{which}"; os.makedirs(od, exist_ok=True)
    jobs = [(r, s) for s in (1, 2) for r in L]
    for k, (r, s) in enumerate(jobs):
        if k % ns != sh: continue
        fo = f"{od}/{r['g']}_{r['n1']}_{r['seed']}_s{s}.json"
        if os.path.exists(fo): continue
        t0 = time.process_time(); b = run(method, prm, r, s)
        json.dump(dict(best=float(b), cpu=time.process_time() - t0, seed=s), open(fo, "w"))
    print("done")
