import json, glob, sys, collections, numpy as np
R = [json.load(open(f)) for f in glob.glob("out_e105/*.json")]; rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.2f} [{np.percentile(b,2.5):.2f}, {np.percentile(b,97.5):.2f}]"
out = [f"sample105: {len(R)} instances; validity LB <= best UB: {all(max(r['J_dff']['lb'], r['N_port']['lb']) <= r['ub'] for r in R)}",
       "| n1 | n | LB gain over LB* % [CI95] | share of (UB − LB*) closed % [CI95] | raised | CPU s (mean) |", "|---|---|---|---|---|---|"]
for n1 in sorted({r["n1"] for r in R}) + ["all"]:
    rs = [r for r in R if n1 == "all" or r["n1"] == n1]
    best = lambda r: max(r["lbstar"], r["J_dff"]["lb"], r["N_port"]["lb"])
    g = [(best(r) - r["lbstar"]) / r["lbstar"] * 100 for r in rs]
    c = [(best(r) - r["lbstar"]) / (r["ub"] - r["lbstar"]) * 100 for r in rs if r["ub"] > r["lbstar"]]
    out.append(f"| {n1} | {len(rs)} | {ci(g)} | {ci(c)} | {sum(best(r) > r['lbstar'] for r in rs)}/{len(rs)} | {np.mean([r['cpu'] for r in rs]):.0f} |")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
