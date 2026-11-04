"""anytime profile experiment (2026-11-03): instance lists. TEST = first two instances (k = 0, 1) of every cell of the first confirmation
set (confirmation protocol, generator seed 20261103; confirmation_eacdc/list_new.json): 35 cells x 2 = 70.
DEV = the 105 benchmark instances of sample105 (lower_bounds), with E0 = all evaluations (init + evalPop) of the
recorded seed-1 static EACDC run (confirmation_eacdc/out_m105/*_s1.json). DEV and TEST are disjoint by construction
(generated vs Cota files; content hashes checked here)."""
import json, os, hashlib
D = os.path.dirname(os.path.abspath(__file__)); G4 = f"{D}/../confirmation_eacdc"; G0 = f"{D}/../benchmark_data"
F = {'m2': '2maquinas', 'm4': '4maquinas', 'm10': '10maquinas', 'u24': '[2-4]maquinas', 'u210': '[2-10]'}
def h(p): return hashlib.sha256(" ".join(open(p).read().split()).encode()).hexdigest()
L = json.load(open(f"{G4}/list_new.json"))
test = [dict(key=r["key"], g=r["g"], n1=r["n1"], path=f"../confirmation_eacdc/{r['path']}") for r in L if r["key"].endswith(("_0", "_1"))]
assert len(test) == 70
dev = []
for r in json.load(open(f"{D}/../lower_bounds/sample105.json")):
    key = f"{r['g']}_{r['n1']}_{r['seed']}"; e = json.load(open(f"{G4}/out_m105/{key}_s1.json"))["eacdc"]
    dev.append(dict(key=key, g=r["g"], n1=r["n1"], path=f"../benchmark_data/data/Dados/{F[r['g']]}/{r['n1']}jobs/{r['seed']}.dat",
                    E0=int(e["n_init"] + e["n_evalpop"])))
ht = {h(f"{D}/{r['path']}") for r in test}; hd = {h(f"{D}/{r['path']}") for r in dev}
assert len(ht) == 70 and len(hd) == 105 and not (ht & hd)
json.dump(test, open(f"{D}/list_test70.json", "w"), indent=0); json.dump(dev, open(f"{D}/list_dev105.json", "w"), indent=0)
print("test", len(test), "dev", len(dev), "disjoint OK")
