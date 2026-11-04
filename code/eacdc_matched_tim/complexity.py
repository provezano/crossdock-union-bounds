"""One (method, instance) per process: CPU, wall, peak RSS, #evaluations, decoder cost. usage: complexity.py METHOD g n1 seed"""
import sys, os, time, json, resource
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import g3, cdc
m, g, n1, sd = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
w0 = time.time(); c0 = time.process_time(); ne = 0
if m == "ils":
    I = cdc.parse(g3.path(g, n1, sd)); Dd = g3.g2.Dec
    cnt = [0]; orig = Dd.__call__
    def wrap(self, *a, **k): cnt[0] += 1; return orig(self, *a, **k)
    Dd.__call__ = wrap
    C = g3.g2.all_constructive(I); kb = min(C, key=lambda k: C[k][0]); b, _, _ = g3.ils_p(I, C[kb][1], C[kb][2], 4.0, seed=101); ne = cnt[0]
else:
    RE, GA, fe = g3._load_ga(); cnt = [0]
    import fast_eval
    orig = g3._load_ga
    r = g3.eacdc(g, n1, sd, 1); b = r["best"]; ne = None
print(json.dumps(dict(m=m, n1=n1, best=b, cpu=time.process_time() - c0, wall=time.time() - w0, evals=ne,
                      rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)))
