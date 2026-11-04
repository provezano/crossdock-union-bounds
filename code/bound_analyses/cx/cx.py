"""Complexity check (one machine, process CPU time, nice 15, 1 thread). For n1 in {20,...,80}, groups m4 and m10,
3 benchmark instances each (seeds fixed), measure per-call CPU of:
  dec_c    : ILS v2 C decoder (fbils_census_exact_cpuc c/srch.c dec via srch.decode)             [ours, v2]
  dec_py   : ILS v1 Python decoder (decoder g2.Dec)                              [ours, v1]
  ev_fast  : EACDC fast evaluator (or_sota_cd code/fast_eval.py; used in all matched runs)
  ev_orig  : EACDC original GA.evalPop (public GA.py, unmodified)
  constr   : the 14 constructive orders (g2.all_constructive), counted as 14 evaluations in the matched runs
  ga_var   : one GA generation without evaluations (DEAP varOr lambda=100, cxOrdered/mutShuffleIndexes, CXPB .95,
             MUTPB .4 renormalised by GA.py to .704/.296, + selTournament(mu=75, k=3) on pop+offspring)
  lp_hull  : LP hull (parametric min cut, lp/lblp.py) and number of vertices V
  lcs      : Chen-Song bound (authors' port fonseca_port.lbs_lbr, both directions)
plus tracemalloc peak (Python-side memory) for dec_py, ev_fast, ev_orig, constr. Writes cx/cx.json."""
import sys, os, json, time, random, tracemalloc, importlib.util, warnings, io, contextlib
warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); F = f"{D}/../.."
for p in ("benchmark_data", "lower_bounds", "decoder", "fbils_census_exact_cpuc", "eacdc_public/code"):
    sys.path.insert(0, f"{F}/{p}")
sys.path.insert(0, f"{D}/../lp")
import numpy as np
import cdc, g2, srch, lblp, fonseca_port, fast_eval
spec = importlib.util.spec_from_file_location("GA", f"{F}/eacdc_public/third/EA-ML/GA.py")
GA = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(GA)
from deap import base, creator, tools, algorithms
FD = {'m4': '4maquinas', 'm10': '10maquinas'}

def cpu_per(f, reps):
    t0 = time.process_time()
    for _ in range(reps): f()
    return (time.process_time() - t0) / reps

def peak(f):
    tracemalloc.start(); f(); pk = tracemalloc.get_traced_memory()[1]; tracemalloc.stop(); return pk

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
rows = []
for g in FD:
    for n1 in range(20, 81, 10):
        for sd in (17, 18, 19):
            I = cdc.parse(f"{F}/benchmark_data/data/Dados/{FD[g]}/{n1}jobs/{sd}.dat")
            n2, E = I["n2"], int(I["prec"].sum()); rng = np.random.default_rng(sd)
            perms1 = [rng.permutation(n1).tolist() for _ in range(64)]; perms2 = [rng.permutation(n2).tolist() for _ in range(64)]
            A = srch.arrays(I); k = [0]
            def dc(): srch.decode(I, perms1[k[0] % 64], A); k[0] += 1
            D2 = g2.Dec(I)
            def dp(): D2(perms1[k[0] % 64]); k[0] += 1
            GA.n1, GA.n2, GA.maq1, GA.maq2 = n1, n2, I["m1"], I["m2"]
            GA.p1, GA.p2, GA.Prec = I["p1"].tolist(), I["p2"].tolist(), I["prec"].astype(int)
            fe = fast_eval.make(GA)
            def ef(): fe(perms2[k[0] % 64]); k[0] += 1
            def eo(): GA.evalPop(perms2[k[0] % 64]); k[0] += 1
            r = dict(g=g, n1=n1, n2=n2, m1=I["m1"], m2=I["m2"], E=E)
            r["dec_c"] = cpu_per(dc, 20000); r["dec_py"] = cpu_per(dp, 400); r["ev_fast"] = cpu_per(ef, 400); r["ev_orig"] = cpu_per(eo, 60)
            r["mem_dec_py"] = peak(dp); r["mem_ev_fast"] = peak(ef); r["mem_ev_orig"] = peak(eo)
            r["constr"] = cpu_per(lambda: g2.all_constructive(I), 2); r["mem_constr"] = peak(lambda: g2.all_constructive(I))
            r["ug1"] = cpu_per(lambda: g2.union_greedy(I, 1.0), 3)
            tb = base.Toolbox(); tb.register("mate", tools.cxOrdered); tb.register("mutate", tools.mutShuffleIndexes, indpb=1.0 / n2)
            pop = [creator.Individual(random.sample(range(n2), n2)) for _ in range(75)]
            for ind in pop: ind.fitness.values = (1.0,)
            def gen():
                off = algorithms.varOr(pop, tb, 100, round(0.95 / 1.35, 3), round(0.4 / 1.35, 3))  # GA.py renormalises CXPB+MUTPB>1
                for o in off: o.fitness.values = (1.0,)
                tools.selTournament(pop + off, 75, tournsize=3)
            r["ga_var"] = cpu_per(gen, 20)
            t0 = time.process_time(); V = lblp.hull(I); r["lp_hull"] = time.process_time() - t0; r["V"] = len(V)
            n, m, m1, m2, p1, p2, S = fonseca_port.parse(f"{F}/benchmark_data/data/Dados/{FD[g]}/{n1}jobs/{sd}.dat")
            r["lcs"] = cpu_per(lambda: fonseca_port.lbs_lbr(n, m, m1, m2, p1, p2, S), 20)
            rows.append(r); print(g, n1, sd, {kk: (round(v * 1e6, 1) if isinstance(v, float) else v) for kk, v in r.items() if kk not in ('g',)}, flush=True)
json.dump(rows, open(f"{D}/cx.json", "w"), indent=0)
