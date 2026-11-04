"""Training set (35 = one random instance per <dock group, n1> class, paper Sect. 4.7) and a stratified
reproduction sample (6 per class), both disjoint from sample105 (our dev set). The 210-instance prereg
confirmation set is freshly generated (confirmation_eacdc/data_new), so it cannot overlap any Cota file.
The authors' 35 are not published (Instances.txt / Dados.csv missing), so their rule is replicated."""
import os, json, re
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
GROUPS = ["m2", "m4", "m10", "u24", "u210"]; N1 = list(range(20, 90, 10))
PER_CELL_REPRO = 6
SEED = 20261102


def rel(g, n1, s): return f"../benchmark_data/data/Dados/{F[g]}/{n1}jobs/{s}.dat"


def main():
    dev = set()
    for r in json.load(open(f"{D}/../confirmation_eacdc/list105.json")):
        g, n1, s = r["key"].split("_"); dev.add((g, int(n1), int(s)))
    rng = np.random.default_rng(SEED); train, repro = [], []
    for g in GROUPS:
        for n1 in N1:
            cand = [s for s in range(17, 317) if (g, n1, s) not in dev]
            pick = rng.choice(cand, size=1 + PER_CELL_REPRO, replace=False).tolist()
            train.append(dict(key=f"{g}_{n1}_{pick[0]}", g=g, n1=n1, seed=pick[0], path=rel(g, n1, pick[0])))
            for s in pick[1:]:
                repro.append(dict(key=f"{g}_{n1}_{s}", g=g, n1=n1, seed=s, path=rel(g, n1, s)))
    for r in train + repro: assert os.path.exists(f"{D}/{r['path']}"), r
    kt = {r["key"] for r in train}; kr = {r["key"] for r in repro}
    kd = {f"{g}_{n}_{s}" for g, n, s in dev}
    assert not (kt & kr) and not (kt & kd) and not (kr & kd)
    json.dump(train, open(f"{D}/train35.json", "w"), indent=0)
    json.dump(repro, open(f"{D}/repro210.json", "w"), indent=0)
    print(len(train), len(repro), "overlap with sample105:", len(kt & kd), len(kr & kd))


if __name__ == "__main__":
    main()
