"""EACDC (public GA.py, verified fast bit-identical evaluator/loader from or_sota_cd) with varied seeds:
GA.py hardcodes random.seed(169); we redirect that call to our seed (only harness change). Args: seed shard nsh"""
import sys, os, json, time, io, contextlib, random
D = os.path.dirname(os.path.abspath(__file__)); S = f"{D}/../eacdc_public/code"
sys.argv = [sys.argv[0]] + sys.argv[1:]; seed, shard, nsh = map(int, sys.argv[1:4])
sys.path.insert(0, S)
import importlib.util
spec = importlib.util.spec_from_file_location("run_eaml", f"{S}/run_eaml.py"); RE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RE)  # __main__ guard: only defines GA, SerialPool, leitura
GA = RE.GA
import fast_eval, fast_load
GA.evalPop = fast_eval.make(GA); GA.load_individuals = fast_load.make(GA)
_rs = random.seed
GA.random.seed = lambda x=None, *a, **k: _rs(seed if x == 169 else x)
params = (3, 150, 75, 100, 0.95, 0.4)
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
L = json.load(open(f"{D}/../lower_bounds/sample105.json"))
od = f"{D}/out_eacdc"; os.makedirs(od, exist_ok=True)
for k, r in enumerate(L):
    if k % nsh != shard: continue
    fo = f"{od}/{r['g']}_{r['n1']}_{r['seed']}_s{seed}.json"
    if os.path.exists(fo): continue
    I = RE.leitura(f"{D}/../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat")
    t = time.process_time()
    with contextlib.redirect_stdout(io.StringIO()): res = GA.GA(*I, *params)
    json.dump(dict(best=float(res[0]), it_best=res[1], cpu=time.process_time() - t, seed=seed), open(fo, "w"))
