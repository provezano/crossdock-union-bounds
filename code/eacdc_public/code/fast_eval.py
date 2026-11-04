"""Output-preserving faster replacement for EA-ML GA.evalPop (same tie-breaking):
inbound order = lexsort(first position of a successor outbound in the permutation, -p1, index);
inbound list scheduling on argmin dock (first index); outbound ERD+LPT (first index on ties),
fallback = min release, LPT among ties. Precomputes per-instance arrays once."""
import numpy as np

def make(GA):
    cache = {}
    def setup():
        key = id(GA.Prec)
        if key not in cache:
            P = np.asarray(GA.Prec, dtype=bool)
            cache.clear()
            cache[key] = (P, np.asarray(GA.p1, dtype=np.int64), np.asarray(GA.p2, dtype=np.int64),
                          int(GA.maq1), int(GA.maq2), int(GA.n1), int(GA.n2), P.any(axis=0))
        return cache[key]
    def evalPop(individual):
        P, p1, p2, m1, m2, n1, n2, has = setup()
        perm = np.frombuffer(individual, dtype=np.int32) if hasattr(individual, "typecode") else np.asarray(individual)
        pos = np.empty(n2, dtype=np.int64); pos[perm] = np.arange(n2)
        # first position (in perm) of an outbound requiring inbound j
        first = np.where(P, pos[:, None], n2 + 1).min(axis=0)
        idx = np.flatnonzero(has)
        order = idx[np.lexsort((idx, -p1[idx], first[idx]))]
        load = [0] * m1; C = np.zeros(n1, dtype=np.int64); p1l = p1.tolist()
        for j in order.tolist():
            k = load.index(min(load)); load[k] += p1l[j]; C[j] = load[k]
        rc = np.where(P, C[None, :], 0).max(axis=1)
        rcl = rc.tolist(); p2l = p2.tolist()
        done = [False] * n2; free = [0] * m2; cmax = 0
        rem = list(range(n2))
        for _ in range(n2):
            t = min(free); k = free.index(t)
            best = -1; ind = -1
            for j in rem:
                if rcl[j] <= t and p2l[j] > best: best = p2l[j]; ind = j
            if ind == -1:
                rmin = min(rcl[j] for j in rem)
                for j in rem:
                    if rcl[j] == rmin and p2l[j] > best: best = p2l[j]; ind = j
            free[k] = max(t, rcl[ind]) + p2l[ind]; cmax = max(cmax, free[k]); rem.remove(ind)
        return cmax,
    return evalPop
