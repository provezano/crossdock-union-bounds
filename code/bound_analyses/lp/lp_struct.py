"""#3b side check: does LB_U^LP ever beat the classical bounds on the structured set (sparse, block, unbalanced,
large; fbils_census_exact_cpuc data_struct, read only)? L0 here = max(PCH, e2, load, Chen-Song fwd/rev via fonseca_port), both directions."""
import sys, os, glob, json, re
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D); sys.path.insert(0, f"{D}/../../eacdc_public/code")
import lblp, cdc, g1, fonseca_port
from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0.0])
for f in sorted(glob.glob(f"{D}/../../fbils_census_exact_cpuc/data_struct/*.dat")):
    fam = re.sub(r"_\d+_\d+\.dat$", "", os.path.basename(f)); I = cdc.parse(f); R = g1.reverse(I)
    n, m, m1, m2, p1, p2, S = fonseca_port.parse(f); LBs, LBr, _, _ = fonseca_port.lbs_lbr(n, m, m1, m2, p1, p2, S)
    L0 = max(LBs, LBr, *(fn(X) for X in (I, R) for fn in (cdc.lb_pch, cdc.lb_stage2_energy, cdc.lb_load)))
    lp = max(float(lblp.lb_lp(I)[0]), float(lblp.lb_lp(R)[0]))
    a = agg[fam]; a[0] += 1; a[1] += lp > L0 + 1e-9; a[2] = max(a[2], (lp - L0) / L0)
for k, (n, w, mx) in sorted(agg.items()):
    print(f"{k:6s} n={n:3d}  LB_U^LP > L0 on {w}  max (LP-L0)/L0 {100*mx:+.2f}%")
