"""Log-log fit of LP (round-1, both stages) build+solve CPU vs n1 and vs columns, from out/fid_code35.jsonl (Optimal only)."""
import json, numpy as np
R = [json.loads(l) for l in open("out/fid_code35.jsonl")]
rng = np.random.default_rng(0)
def fit(x, y, lab):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float)); A = np.vstack([x, np.ones_like(x)]).T
    s = np.linalg.lstsq(A, y, rcond=None)[0][0]; b = []
    for _ in range(2000):
        i = rng.integers(len(x), size=len(x)); b.append(np.linalg.lstsq(A[i], y[i], rcond=None)[0][0])
    print(f"{lab:40s} n={len(x):2d} slope {s:.2f} [{np.percentile(b,2.5):.2f}, {np.percentile(b,97.5):.2f}]")
for st in (0, 1):
    rows = [(d["n1"], d["lp"][st]["cols"], d["lp"][st]["nnz"], d["lp"][st]["cpu"], d["g"]) for d in R if len(d["lp"]) > st and d["lp"][st]["status"] == "Optimal"]
    fit([r[0] for r in rows], [r[3] for r in rows], f"stage{st+1} LP cpu vs n1 (all groups)")
    fit([r[1] for r in rows], [r[3] for r in rows], f"stage{st+1} LP cpu vs cols")
    fit([r[0] for r in rows], [r[1] for r in rows], f"stage{st+1} LP cols vs n1")
    for g in ("m10", "u210", "m4"):
        s = [r for r in rows if r[4] == g]
        if len(s) >= 4: fit([r[0] for r in s], [r[3] for r in s], f"  {g} stage{st+1} LP cpu vs n1")
    print("  max cols/nnz/cpu:", max(r[1] for r in rows), max(r[2] for r in rows), round(max(r[3] for r in rows), 1))
big = [(d["key"], d["lp"][0]["cols"], d["lp"][0]["nnz"], d["lp"][0]["status"], round(d["lp"][0]["cpu"])) for d in R if d["lp"] and d["lp"][0]["status"] != "Optimal"]
print("round-1 stage-1 LPs not solved to optimality within the cap:", big)
