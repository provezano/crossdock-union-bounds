"""Fresh instances for the ILS v2 prereg: confirmation_eacdc/gen_new.py logic (benchmark rules), seed 20261105."""
import os, sys, json, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
import gen_new
rng = np.random.default_rng(20261105); os.makedirs(f"{D}/data_fbils_confirm", exist_ok=True); L = []
for g in ("m2", "m4", "m10", "u24", "u210"):
    for n1 in range(20, 90, 10):
        for k in range(6):
            n1_, n2, m1, m2, p1, p2, P = gen_new.gen(rng, n1, g); fn = f"data_fbils_confirm/{g}_{n1}_{k}.dat"
            with open(f"{D}/{fn}", "w") as f:
                f.write(f"{n1_}\n{n2}\n{m1}\n{m2}\n\n" + "\n".join(map(str, p1)) + "\n\n" + "\n".join(map(str, p2)) + "\n\n"
                        + "\n".join(" ".join(str(int(x)) for x in row) for row in P) + "\n")
            L.append(dict(key=f"{g}_{n1}_{k}", g=g, n1=n1, path=fn))
json.dump(L, open(f"{D}/list_fbils_confirm.json", "w")); print(len(L))
