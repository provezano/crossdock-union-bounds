"""Theory check: union-greedy (alpha = 1, 2) list schedule vs a good order on the family
X (one outbound, length L, preds S = m inbound of length 2L/m) + m^2 outbound Y_k of length L/m, each with a private
inbound pred of length L/m; m1 = m2 = m. Prints Cmax(ug)/Cmax(S-first) -> 2 as m grows."""
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, "../decoder"); sys.path.insert(0, "../lower_bounds"); sys.path.insert(0, "../benchmark_data")
import numpy as np, g2
for m in (2, 4, 6, 8, 12, 16):
    L = 12 * m * m; c = 2 * L // m; ny = m * m
    n1 = m + ny; n2 = 1 + ny
    p1 = np.array([c] * m + [L // m] * ny); p2 = np.array([L] + [L // m] * ny)
    P = np.zeros((n2, n1), bool); P[0, :m] = True
    for k in range(ny): P[1 + k, m + k] = True
    I = dict(n1=n1, n2=n2, m1=m, m2=m, p1=p1, p2=p2, prec=P)
    D = g2.Dec(I); good = D(list(range(n1)))
    r = {a: D(g2.union_greedy(I, a)) for a in (0.5, 1.0, 2.0)}
    lb = max(c + L, p1.sum() / m, p2.sum() / m)
    print(f"m={m:2d} L={L}: S-first {good} (LB {lb:.0f}); ug1 {r[1.0]} ratio {r[1.0]/good:.3f}; ug2 {r[2.0]/good:.3f}; ug0.5 {r[0.5]/good:.3f}")
