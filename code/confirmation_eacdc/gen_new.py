"""Fresh Cota-rule instances (preregistered; see the paper): 35 cells x 6, seed 20261103.
Writes data_new/<g>_<n1>_<k>.dat in the benchmark text format and list_new.json."""
import os, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
def gen(rng, n1, g):
    n2 = int(rng.integers(int(np.ceil(0.8 * n1)), int(np.floor(1.2 * n1)) + 1))
    if g in ("m2", "m4", "m10"): m1 = m2 = int(g[1:])
    else:
        hi = 4 if g == "u24" else 10; m1 = int(rng.integers(2, hi + 1)); m2 = int(rng.integers(2, hi + 1))
    P = rng.random((n2, n1)) < 0.5
    for j in range(n2):
        if not P[j].any(): P[j, rng.integers(n1)] = True
    for i in range(n1):
        if not P[:, i].any(): P[rng.integers(n2), i] = True
    return n1, n2, m1, m2, rng.integers(10, 101, n1), rng.integers(10, 101, n2), P
if __name__ == "__main__":
    rng = np.random.default_rng(20261103); os.makedirs(f"{D}/data_new", exist_ok=True); L = []
    for g in ("m2", "m4", "m10", "u24", "u210"):
        for n1 in range(20, 90, 10):
            for k in range(6):
                n1_, n2, m1, m2, p1, p2, P = gen(rng, n1, g); fn = f"data_new/{g}_{n1}_{k}.dat"
                with open(f"{D}/{fn}", "w") as f:
                    f.write(f"{n1_}\n{n2}\n{m1}\n{m2}\n\n" + "\n".join(map(str, p1)) + "\n\n" + "\n".join(map(str, p2)) + "\n\n"
                            + "\n".join(" ".join(str(int(x)) for x in row) for row in P) + "\n")
                L.append(dict(key=f"{g}_{n1}_{k}", g=g, n1=n1, path=fn))
    json.dump(L, open(f"{D}/list_new.json", "w")); print(len(L))
