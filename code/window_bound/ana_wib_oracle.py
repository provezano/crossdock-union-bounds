"""Oracle-swap table for WIB (toolkit d). usage: ana_wib_oracle.py [TXT]"""
import sys; sys.dont_write_bytecode = True
import json, glob, numpy as np
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return f"{x.mean():.1f} [{np.percentile(b, 2.5):.1f}, {np.percentile(b, 97.5):.1f}]"
R = [json.load(open(f)) for f in glob.glob("out_wib_oracle/*.json")]; R = [r for r in R if r["opt"] > r["lbstar"]]
rows = [k for k in R[0] if isinstance(R[0][k], dict)]
out = [f"oracle subset: {len(R)} solved instances with stored S* (gap > 0); violations: {[(r['key'], k) for r in R for k in rows if r[k]['lb'] > r['opt']]}",
       "| row | gap closed % [CI95] all | m2 | m10 | u24 | u210 | CPU s mean |", "|---|---|---|---|---|---|---|"]
for k in rows:
    f = lambda rs: [100 * (r[k]["lb"] - r["lbstar"]) / (r["opt"] - r["lbstar"]) for r in rs]
    cells = [ci(f(R))] + [(ci(f([r for r in R if r["g"] == g])) if sum(r["g"] == g for r in R) > 1 else "n/a") + f" (n={sum(r['g'] == g for r in R)})" for g in ("m2", "m10", "u24", "u210")]
    out.append(f"| {k} | " + " | ".join(cells) + f" | {np.mean([r[k]['cpu'] for r in R]):.1f} |")
out.append(f"head deficit at T=OPT (sum (S* head - inferred r2)+ / sum S* heads): {ci([100*r['head_deficit_rel'] for r in R])} %; tail deficit: {ci([100*r['tail_deficit_rel'] for r in R])} %")
for g in ("m2", "m10", "u24", "u210"):
    rs = [r for r in R if r["g"] == g]
    if rs: out.append(f"  {g}: head deficit {np.mean([100*r['head_deficit_rel'] for r in rs]):.1f}%, tail deficit {np.mean([100*r['tail_deficit_rel'] for r in rs]):.1f}%")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
