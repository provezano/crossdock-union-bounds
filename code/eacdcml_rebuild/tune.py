"""Per-instance Optuna calibration on the 35 training instances (paper Sect. 4.7; budget = SPEC B1).
Search space (paper): NGEN int [50,200]; CXPB real [0,1]; mu int [50,200]; lambda int [50,200]; ts int [1,20].
Trials per instance = max(5, round(n1/2)) (public Optuna.py), each one EA run with the fixed seed 169 (paper:
fixed seed; code: random.seed(169)); objective = best makespan. Paper cap: 3,600 s per instance (CPU here).
Sampler: GP Bayesian optimisation (Optuna GPSampler), the nearest available equivalent of the code's
optuna.integration.SkoptSampler (GP, 10 random start-up points).
usage: tune.py LIST.json OUTDIR SHARD NSH [TRIAL_MULT]"""
import sys, os, json, time, platform
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import torch; torch.set_num_threads(1)
import optuna
import eacdcml as E, cdc

optuna.logging.set_verbosity(optuna.logging.WARNING)
CAP = 3600.0


def tune_one(r, out, mult=1):
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): return
    I = cdc.parse(f"{D}/{r['path']}"); n_trials = max(5, round(I["n1"] / 2)) * mult
    t0 = time.process_time(); rows = []

    def objective(trial):
        p = dict(NGEN=trial.suggest_int("NGEN", 50, 200), CXPB=trial.suggest_float("CXPB", 0.0, 1.0),
                 mu=trial.suggest_int("mu", 50, 200), lam=trial.suggest_int("lam", 50, 200),
                 ts=trial.suggest_int("ts", 1, 20))
        res = E.run(I, p, seed=169)
        rows.append(dict(params=p, best=res["best"], cpu=res["cpu"], evals=res["evals"], gens=res["gens"],
                         stop=res["stop"]))
        return res["best"]

    def cap_cb(study, trial):
        if time.process_time() - t0 >= CAP: study.stop()

    sampler = optuna.samplers.GPSampler(seed=169)
    st = optuna.create_study(direction="minimize", sampler=sampler)
    st.optimize(objective, n_trials=n_trials, callbacks=[cap_cb])
    bp = st.best_params
    json.dump(dict(key=r["key"], features=E.features(I), feat_names=E.FEATURES, n_trials=n_trials,
                   n_done=len(rows), best_value=st.best_value,
                   best_params=dict(NGEN=bp["NGEN"], CXPB=bp["CXPB"], mu=bp["mu"], lam=bp["lam"], ts=bp["ts"]),
                   trials=rows, cpu_total=time.process_time() - t0,
                   env=dict(python=sys.version.split()[0], optuna=optuna.__version__, host=platform.node())),
              open(fo + ".tmp", "w"))
    os.replace(fo + ".tmp", fo)


def main():
    lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    mult = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    os.makedirs(out, exist_ok=True)
    L = json.load(open(lst))
    # largest instances first, round-robin over shards, to balance shard CPU
    L = sorted(L, key=lambda r: -r["n1"])
    for q, r in enumerate(L):
        if q % ns == sh: tune_one(r, out, mult)
    print("done")


if __name__ == "__main__":
    main()
