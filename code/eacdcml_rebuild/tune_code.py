"""Per-instance Optuna calibration of EACDCml-code (GA.py) on the 35 training instances: same space, trial count,
cap, sampler and seed as tune.py. usage (cwd = third/EA-ML, python -I): tune_code.py LIST OUTDIR SHARD NSH [MULT]"""
import sys, os, json, time, platform
D = "/".join(os.path.abspath(sys.argv[0]).split("/")[:-1]); sys.path.insert(0, D)
import torch; torch.set_num_threads(1)
import optuna
import eacdcml as E, cdc
from code_arm import run_code
optuna.logging.set_verbosity(optuna.logging.WARNING)
CAP = 3600.0
lst, out, sh, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); mult = int(sys.argv[5]) if len(sys.argv) > 5 else 1
os.makedirs(out, exist_ok=True)
for q, r in enumerate(sorted(json.load(open(lst)), key=lambda r: -r["n1"])):
    if q % ns != sh: continue
    fo = f"{out}/{r['key']}.json"
    if os.path.exists(fo): continue
    fp = f"{D}/{r['path']}"; I = cdc.parse(fp); n_trials = max(5, round(I["n1"] / 2)) * mult; rows = []; t0 = time.process_time()
    def objective(trial):
        p = dict(NGEN=trial.suggest_int("NGEN", 50, 200), CXPB=trial.suggest_float("CXPB", 0.0, 1.0),
                 mu=trial.suggest_int("mu", 50, 200), lam=trial.suggest_int("lam", 50, 200), ts=trial.suggest_int("ts", 1, 20))
        x = run_code(fp, p, 169); rows.append(dict(params=p, best=x["best"], cpu=x["cpu"], evals=x["evals"])); return x["best"]
    def cap_cb(study, trial):
        if time.process_time() - t0 >= CAP: study.stop()
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.GPSampler(seed=169))
    st.optimize(objective, n_trials=n_trials, callbacks=[cap_cb]); bp = st.best_params
    json.dump(dict(key=r["key"], features=E.features(I), feat_names=E.FEATURES, n_trials=n_trials, n_done=len(rows),
                   best_value=st.best_value, best_params={k: bp[k] for k in ("NGEN", "CXPB", "mu", "lam", "ts")}, trials=rows,
                   cpu_total=time.process_time() - t0, engine="GA.py", env=dict(python=sys.version.split()[0], host=platform.node())),
              open(fo + ".tmp", "w")); os.replace(fo + ".tmp", fo)
print("done")
