"""Run the public EA-ML GA.py (yotitinogs/EA-ML, unmodified file) on Cota+2016 instances.
Only harness changes: multiprocessing.Pool replaced by a serial in-process map (fitness is
deterministic, so results are identical; keeps us at 1 process), optional fast evaluator swap.
Usage: run_eaml.py OUT.json [--fast] inst1.dat inst2.dat ..."""
import sys, os, json, time, resource, importlib.util, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
EAML = os.path.join(HERE, "..", "third", "EA-ML")
spec = importlib.util.spec_from_file_location("GA", os.path.join(EAML, "GA.py"))
GA = importlib.util.module_from_spec(spec); spec.loader.exec_module(GA)
sys.path.insert(0, HERE)

class SerialPool:
    def map(self, f, it): return list(map(f, it))
    def close(self): pass
GA.multiprocessing.Pool = lambda *a, **k: SerialPool()
GA.multiprocessing.cpu_count = lambda: 1

def leitura(path):  # identical to Alg.Leitura (lists for p1/p2, int ndarray Prec)
    import numpy as np
    d = [int(t) for t in open(path).read().split()]
    n1, n2, m1, m2 = d[:4]; p1 = d[4:4+n1]; p2 = d[4+n1:4+n1+n2]
    Prec = np.array(d[4+n1+n2:4+n1+n2+n1*n2], dtype=int).reshape(n2, n1)
    return n1, n2, m1, m2, p1, p2, Prec

if __name__ == "__main__":
    out, args = sys.argv[1], sys.argv[2:]
    fast = "--fast" in args; args = [a for a in args if a != "--fast"]
    if args and args[0].startswith("@"): args = open(args[0][1:]).read().split("\n")
    params = (3, 150, 75, 100, 0.95, 0.4)  # defaults in GA.py comments
    if fast:
        import fast_eval
        GA.evalPop = fast_eval.make(GA)
        import fast_load
        GA.load_individuals = fast_load.make(GA)
    res = []
    import io, contextlib
    for p in args:
        I = leitura(p); t = time.process_time()
        with contextlib.redirect_stdout(io.StringIO()):
            r = GA.GA(*I, *params)
        res.append(dict(inst=p, best=float(r[0]), it_best=r[1], cpu=time.process_time()-t, wall=r[2],
                        maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
        print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(out, "w"))
