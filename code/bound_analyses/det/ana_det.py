"""#3a analysis: LB_U*-det vs CPU-capped census (forward), per n1; reproducibility vs exact_cpsat det_local/det_machineB (150)."""
import json, glob, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__))
R = [json.load(open(f)) for f in glob.glob(f"{D}/out/*.json")]
print("instances", len(R), "hosts", sorted({r['host'] for r in R}), "py", sorted({r['py'] for r in R}))
def rep(S, name):
    d = np.array([r['lb_det'] for r in S]); c = np.array([r['lb_cpu4'] for r in S]); rel = (d - c) / c * 100
    ex_d = np.mean([r['exact_det'] for r in S]); ex_c = np.mean([r['exact_cpu4'] for r in S])
    both = [r for r in S if r['exact_det'] and r['exact_cpu4']]
    eq_both = np.mean([r['lb_det'] == r['lb_cpu4'] for r in both]) if both else float('nan')
    cpu = np.array([r['cpu'] for r in S])
    print(f"{name:8s} n={len(S):5d} equal {np.mean(d == c):.3f} det>cpu4 {np.mean(d > c):.3f} det<cpu4 {np.mean(d < c):.4f} "
          f"mean rel {rel.mean():+.3f}% max rel {rel.max():+.3f}% min rel {rel.min():+.3f}% max |abs| {np.abs(d - c).max():.0f} | "
          f"walk finished det {ex_d:.3f} cpu4 {ex_c:.3f} | both finished n={len(both)} equal {eq_both:.4f} | det cpu mean {cpu.mean():.2f}s max {cpu.max():.1f}s")
for n1 in sorted({r['n1'] for r in R}):
    rep([r for r in R if r['n1'] == n1], f"n1={n1}")
rep([r for r in R if r['n1'] >= 60], "n1>=60"); rep(R, "all")
for g in sorted({r['g'] for r in R}):
    rep([r for r in R if r['g'] == g and r['n1'] >= 60], f"{g}>=60")
# reproducibility against the paper's 150 (both machines)
G5 = f"{D}/../../exact_cpsat"; K = {r['key']: r for r in R}
for fn in ("det_local.json", "det_machineB.json"):
    X = json.load(open(f"{G5}/{fn}")); rows = [x for x in X['rows'] if x['key'] in K]
    same = sum(x['lb_det'] == K[x['key']]['lb_det'] and x['steps'] == K[x['key']]['steps'] for x in rows)
    print(f"reproduce {fn} ({X['host']}, py {X['py']}): identical value+steps {same}/{len(rows)}")
