"""EACDCml-code: the authors' public GA.py (unmodified file, fast identical evaluator/loader via g3.eacdc, seed
redirect only) driven by per-instance parameters. Paper outputs (NGEN, CXPB, mu, lambda, ts) map to the GA.py
tuple (ts, NGEN, mu, lambda, CXPB, MUTPB = 1 - CXPB) (paper: crossover OR mutation, binomial with p = CXPB).
Run from the third-party dir with python -I (drivers insert their own paths)."""
import sys, os, time
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{D}/../eacdc_matched_tim")
import g3


def ga_tuple(p): return (int(p["ts"]), int(p["NGEN"]), int(p["mu"]), int(p["lam"]), float(p["CXPB"]), 1.0 - float(p["CXPB"]))


def run_code(fpath, p, seed=169):
    e = g3.eacdc(None, None, None, seed, budget=None, params=ga_tuple(p), fpath=fpath)
    return dict(best=float(e["best"]), cpu=e["cpu"], evals=e["n_init"] + e["n_evalpop"], n_init=e["n_init"],
                n_evalpop=e["n_evalpop"], trace_e=e["trace_e"], trace=e["trace"], params=dict(p))
