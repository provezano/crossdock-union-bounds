"""structured-instance experiment generator (preregistered; see the paper). Documented extension of the Cota
benchmark rules (confirmation_eacdc/gen_new.py): only the precedence law and dock counts change.
usage: gen_struct.py [SEED [OUTDIR]]   (default 20261104, data_struct/) -> OUTDIR/<key>.dat + list file"""
import os, sys, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__))

def prec(rng, n1, n2, law, arg):
    if law == "S":
        P = rng.random((n2, n1)) < arg; b1 = b2 = None
    else:  # block: K destination groups, 0.6 inside, 0.02 across
        K = arg; b1 = rng.permutation(np.arange(n1) % K); b2 = rng.permutation(np.arange(n2) % K)
        same = b2[:, None] == b1[None, :]
        P = rng.random((n2, n1)) < np.where(same, 0.6, 0.02)
    for j in range(n2):
        if not P[j].any():
            c = np.arange(n1) if b1 is None else np.flatnonzero(b1 == b2[j]); P[j, rng.choice(c)] = True
    for i in range(n1):
        if not P[:, i].any():
            c = np.arange(n2) if b2 is None else np.flatnonzero(b2 == b1[i]); P[rng.choice(c), i] = True
    return P

def gen(rng, n1, law, arg, docks):
    n2 = int(rng.integers(int(np.ceil(0.8 * n1)), int(np.floor(1.2 * n1)) + 1))
    m1, m2 = docks if docks else (int(rng.integers(2, 11)), int(rng.integers(2, 11)))
    P = prec(rng, n1, n2, law, arg)
    return n1, n2, m1, m2, rng.integers(10, 101, n1), rng.integers(10, 101, n2), P

CELLS = ([("S", d, None, n1, 10) for d in (0.1, 0.2, 0.3) for n1 in (30, 60, 100)]
         + [("B", K, None, n1, 10) for K in (3, 6) for n1 in (30, 60, 100)]
         + [("U", 0.5, dk, n1, 5) for dk in ((2, 10), (10, 2), (2, 6), (6, 2)) for n1 in (30, 60, 100)]
         + [("L", 0.2, None, n1, 5) for n1 in (150, 200, 300)] + [("LB", 6, None, n1, 5) for n1 in (150, 200, 300)])

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261104
    out = sys.argv[2] if len(sys.argv) > 2 else f"{D}/data_struct"
    rng = np.random.default_rng(seed); os.makedirs(out, exist_ok=True); L = []
    for fam, arg, dk, n1, k in CELLS:
        law = "B" if fam in ("B", "LB") else "S"
        for q in range(k):
            n1_, n2, m1, m2, p1, p2, P = gen(rng, n1, law, arg, dk)
            tag = {"S": f"S{int(arg*10)}", "B": f"B{arg}", "U": f"U{dk[0]}x{dk[1]}" if dk else "", "L": "LS2", "LB": "LB6"}[fam]
            key = f"{tag}_{n1}_{q}"; fn = f"{out}/{key}.dat"
            with open(fn, "w") as f:
                f.write(f"{n1_}\n{n2}\n{m1}\n{m2}\n\n" + "\n".join(map(str, p1)) + "\n\n" + "\n".join(map(str, p2)) + "\n\n"
                        + "\n".join(" ".join(str(int(x)) for x in row) for row in P) + "\n")
            L.append(dict(key=key, fam=fam if fam != "LB" else "L", cell=tag, n1=n1, q=q, path=fn, density=float(P.mean()), m1=m1, m2=m2))
    json.dump(L, open(f"{out}/list.json", "w"), indent=0); print(len(L))
