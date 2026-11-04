"""Time-indexed MIP baseline (Cota et al. 2016, eqs. 10-15) solved with HiGHS, 1 thread, CPU-time limit.
Reuses the frozen LP builder eacdc_matched_tim/tim_lp.py UNCHANGED: a shim replaces highspy.Highs inside that module so
that (i) the x columns become integer, (ii) the incumbent schedule is passed as a start solution, (iii) the run is
interrupted by a MIP callback once the process CPU time since the start of the build exceeds the limit.
Horizon T = makespan of the incumbent (a valid UB). Cmax and r2 stay continuous (Cmax is integral at integral x)."""
import os, sys, time, types
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../../eacdc_matched_tim")
import numpy as np
import highspy as HP
import tim_lp as TL

_CTX = {}


class _MipHighs(HP.Highs):
    def passModel(self, lp):
        c = _CTX; n2 = c["n2"]; ncol = lp.num_col_; nx = ncol - n2 - 1
        lp.integrality_ = [HP.HighsVarType.kInteger] * nx + [HP.HighsVarType.kContinuous] * (n2 + 1)
        r = super().passModel(lp); c["h"] = self; c["nx"] = nx; c["ncol"] = ncol
        return r

    def run(self):
        c = _CTX
        self.setOptionValue("solver", "choose")            # tim_lp may set ipm for big LPs; MIP needs "choose"
        self.setOptionValue("threads", 1); self.setOptionValue("mip_rel_gap", 0.0)
        self.setOptionValue("mip_abs_gap", 0.999)            # objective is integral at integral x
        # presolve is not interruptible by callbacks: HiGHS time_limit (wall) = remaining CPU budget; callbacks enforce CPU
        self.setOptionValue("time_limit", max(1.0, float(c["limit"]) - (time.process_time() - c["t0"])))
        if c.get("start") is not None:
            self.setSolution(HP.HighsSolution() if False else _sol(c["start"], c["ncol"]))
        t0 = c["t0"]; lim = c["limit"]
        def _intr(e):
            if time.process_time() - t0 > lim: e.interrupt()
        self.cbMipInterrupt.subscribe(_intr); self.cbSimplexInterrupt.subscribe(_intr); self.cbIpmInterrupt.subscribe(_intr)
        return super().run()


def _sol(x, ncol):
    s = HP.HighsSolution(); s.col_value = list(map(float, x)); s.value_valid = True
    return s


def start_vector(I, T, s1, s2):
    p1 = np.asarray(I["p1"], int); p2 = np.asarray(I["p2"], int); P = np.asarray(I["prec"], bool); n2 = len(p2)
    off1 = np.concatenate([[0], np.cumsum(T - p1 + 1)]); N1 = off1[-1]
    off2 = N1 + np.concatenate([[0], np.cumsum(T - p2 + 1)]); N2 = off2[-1]
    x = np.zeros(N2 + n2 + 1)
    for e, s in enumerate(s1): assert 0 <= s <= T - p1[e]; x[off1[e] + s] = 1
    for e, s in enumerate(s2): assert 0 <= s <= T - p2[e]; x[off2[e] + s] = 1
    c1 = np.asarray(s1) + p1
    for e in range(n2): x[N2 + e] = float(c1[P[e]].max()) if P[e].any() else 0.0
    x[N2 + n2] = float(max(c1.max(), (np.asarray(s2) + p2).max()))
    return x


def solve(I, T, limit, sched=None):
    """Returns dict(opt or None, lo (ceil dual bound), hi (best integer objective, <= T), cpu, wall, status, ncol)."""
    t0 = time.process_time(); w0 = time.time()
    _CTX.clear(); _CTX.update(n2=int(I["n2"]), limit=float(limit), t0=t0,
                              start=start_vector(I, int(T), *sched) if sched is not None else None)
    shim = types.SimpleNamespace(**{k: getattr(HP, k) for k in dir(HP) if not k.startswith("__")}); shim.Highs = _MipHighs
    old = TL.highspy; TL.highspy = shim
    try:
        TL.tim_lp(I, T=int(T))
    finally:
        TL.highspy = old
    h = _CTX["h"]; info = h.getInfo(); st = h.getModelStatus()
    cpu = time.process_time() - t0
    prim = float(info.objective_function_value) if info.primal_solution_status == 2 else None
    db = float(info.mip_dual_bound)
    lo = int(np.ceil(db - 1e-6)) if np.isfinite(db) else 0
    hi = int(round(prim)) if prim is not None else int(T)
    hi = min(hi, int(T))
    opt = hi if (st == HP.HighsModelStatus.kOptimal or lo >= hi) else None
    if opt is not None: lo = hi
    return dict(opt=opt, lo=lo, hi=hi, cpu=cpu, wall=time.time() - w0, status=str(st), ncol=int(_CTX["ncol"]),
                mip_nodes=int(info.mip_node_count))


def parse(path):
    """Benchmark text format (same as cdc.parse): n1, n2, m1, m2, p1 (n1), p2 (n2), precedence matrix n2 x n1."""
    t = open(path).read().split(); n1, n2, m1, m2 = map(int, t[:4]); o = 4
    p1 = np.array(t[o:o + n1], dtype=np.int64); o += n1; p2 = np.array(t[o:o + n2], dtype=np.int64); o += n2
    P = np.array(t[o:o + n1 * n2], dtype=np.int64).reshape(n2, n1).astype(bool)
    return dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=p1, p2=p2, prec=P)


if __name__ == "__main__":
    # Child-process entry (Linux: ortools and highspy wheels clash on shared HiGHS symbols in one process, so the
    # caller runs this arm in a separate interpreter that never imports ortools). usage: tim_mip.py INST T LIMIT START.json
    import json
    I = parse(sys.argv[1]); st = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
    r = solve(I, int(sys.argv[2]), float(sys.argv[3]), sched=(st["s1"], st["s2"]) if st else None)
    print("TIMRESULT " + json.dumps(r), flush=True)
