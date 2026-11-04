"""ctypes wrapper for c/libsrch.dylib (FB-ILS development). run(I, side, order, algo, ...) mirrors g3.ils_p's interface."""
import ctypes as C, os, time, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); import sys as _s; LIB = C.CDLL(f"{D}/c/libsrch." + ("dylib" if _s.platform == "darwin" else "so"))  # platform-specific build of the same srch.c
ip = np.ctypeslib.ndpointer(np.int32, flags="C"); i64 = np.ctypeslib.ndpointer(np.int64, flags="C"); dp = np.ctypeslib.ndpointer(np.float64, flags="C")
LIB.search.restype = C.c_int
LIB.search.argtypes = [C.c_int] + [C.c_int] * 4 + [ip] * 5 + [ip, C.c_double, C.c_longlong, C.c_double, C.c_ulonglong,
                       C.c_double, C.c_double, C.c_int, C.c_int, dp, i64, ip, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_longlong), C.c_longlong]
LIB.decode_once.restype = C.c_int
LIB.decode_once.argtypes = [C.c_int] * 4 + [ip] * 5 + [ip]

def arrays(J):
    P = J["prec"]; pp = np.zeros(J["n2"] + 1, np.int32); pi = []
    for j in range(J["n2"]):
        q = np.flatnonzero(P[j]); pi += q.tolist(); pp[j + 1] = len(pi)
    lpt = np.array(sorted(range(J["n2"]), key=lambda j: (-int(J["p2"][j]), j)), np.int32)
    return (int(J["n1"]), int(J["n2"]), int(J["m1"]), int(J["m2"]), np.ascontiguousarray(J["p1"], np.int32),
            np.ascontiguousarray(J["p2"], np.int32), pp, np.array(pi, np.int32), lpt)

def decode(J, order, A=None):
    A = A or arrays(J); return LIB.decode_once(*A, np.ascontiguousarray(order, np.int32))

ALGOS = {"ils": 0, "ig": 1, "igb": 2, "lahc": 3, "crit": 4, "win": 5}
def run(J, order, algo, cpu_budget, eval_budget=0, cap=None, seed=0, fp1=0.0, fp2=0.0, ip1=40, ip2=0, nev0=0, A=None):
    """J: instance on the searched side. Returns (best, order, trace[(cpu, evals, best)], nevals)."""
    A = A or arrays(J); o = np.ascontiguousarray(order, np.int32).copy(); mt = 100000
    tc = np.zeros(mt); te = np.zeros(mt, np.int64); tv = np.zeros(mt, np.int32); nt = C.c_int(0); ne = C.c_longlong(0)
    cap = cap if cap is not None else max(cpu_budget, 1e9 if eval_budget == 0 else cpu_budget * 50 + 60)
    b = LIB.search(ALGOS[algo], *A, o, cpu_budget, eval_budget, cap, seed, fp1, fp2, ip1, ip2, tc, te, tv, mt, C.byref(nt), C.byref(ne), nev0)
    k = nt.value
    return b, o.tolist(), list(zip(tc[:k].tolist(), te[:k].tolist(), tv[:k].tolist())), ne.value

LIB.search_fbi.restype = C.c_int
LIB.search_fbi.argtypes = [C.c_int] * 4 + [ip] * 5 + [ip] * 3 + [ip, C.c_double, C.c_longlong, C.c_double, C.c_ulonglong,
                           C.c_double, C.c_int, C.c_int, C.c_double, dp, i64, ip, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_longlong), C.c_longlong]
def run_fbi(J, Jr, order, cpu_budget, eval_budget=0, cap=None, seed=0, pfbi=0.0, stall_mult=40, rounds=3, nev0=0, wfrac=0.0):
    """ILS + forward-backward improvement (algo 6). J: searched side, Jr: its reversed instance."""
    A = arrays(J); B = arrays(Jr); o = np.ascontiguousarray(order, np.int32).copy(); mt = 100000
    tc = np.zeros(mt); te = np.zeros(mt, np.int64); tv = np.zeros(mt, np.int32); nt = C.c_int(0); ne = C.c_longlong(0)
    cap = cap if cap is not None else max(cpu_budget, 1e9 if eval_budget == 0 else cpu_budget * 50 + 60)
    b = LIB.search_fbi(*A, B[6], B[7], B[8], o, cpu_budget, eval_budget, cap, seed, pfbi, stall_mult, rounds, wfrac, tc, te, tv, mt,
                       C.byref(nt), C.byref(ne), nev0)
    k = nt.value
    return b, o.tolist(), list(zip(tc[:k].tolist(), te[:k].tolist(), tv[:k].tolist())), ne.value
