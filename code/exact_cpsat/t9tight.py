"""step 3a (optional): adversarial search for the worst list-schedule ratio on bipartite depth-2 instances.
Objective: max over ALL inbound orders pi of Dec(pi) / LB_U* (project decoder and LB), n1 <= 6, n2 <= 5,
(m1, m2) fixed per restart in {(2,2),(2,3),(3,2),(3,3)}. Random-restart hill climbing on lengths (1..K) and arcs.
Reports the best ratio found per (m1, m2) vs T9' bound 3 - 1/max(m1, m2) and vs Graham 2 - 1/M.
usage: t9tight.py OUT.json CPU_SECONDS"""
import sys, os, json, time, itertools
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np
import t9prime as T


def worst(I):
    Dd = T.g2.Dec(I); lb = T.lbstar(I)["lb"]
    return max(Dd(list(p)) for p in itertools.permutations(range(I["n1"]))) / lb


def fix(I):
    P = I["prec"]
    for j in range(I["n2"]):
        if not P[j].any(): P[j, 0] = True
    for i in range(I["n1"]):
        if not P[:, i].any(): P[0, i] = True
    return I


def main():
    out, cap = sys.argv[1], float(sys.argv[2]); rng = np.random.default_rng(11); t0 = time.process_time()
    best = {}; it = 0
    while time.process_time() - t0 < cap:
        m1, m2 = [(2, 2), (2, 3), (3, 2), (3, 3)][it % 4]; it += 1
        n1, n2 = int(rng.integers(2, 7)), int(rng.integers(1, 6)); K = 20
        I = fix(dict(n1=n1, n2=n2, m1=m1, m2=m2, p1=rng.integers(1, K + 1, n1), p2=rng.integers(1, K + 1, n2),
                     prec=rng.random((n2, n1)) < 0.5))
        f = worst(I)
        for _ in range(150):
            J = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in I.items()}
            u = rng.random()
            if u < 0.4: J["p1"][rng.integers(n1)] = rng.integers(1, K + 1)
            elif u < 0.7: J["p2"][rng.integers(n2)] = rng.integers(1, K + 1)
            else: a, b = rng.integers(n2), rng.integers(n1); J["prec"][a, b] = ~J["prec"][a, b]
            J = fix(J); g = worst(J)
            if g >= f: I, f = J, g
        k = f"{m1},{m2}"
        if f > best.get(k, (0,))[0]:
            best[k] = (f, {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in I.items()})
    res = {k: dict(ratio=v[0], t9p=3 - 1 / max(map(int, k.split(","))), graham=2 - 1 / max(map(int, k.split(","))),
                   inst=v[1]) for k, v in best.items()}
    json.dump(dict(restarts=it, cpu=time.process_time() - t0, best=res), open(out, "w"), indent=1)
    print({k: round(v["ratio"], 4) for k, v in res.items()})


if __name__ == "__main__":
    main()
