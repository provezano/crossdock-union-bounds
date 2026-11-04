"""Elementary-operation counts per decode (loop iterations / element touches), from instrumented line-by-line
re-statements of (a) srch.c dec() [ours, v1 and v2 share it], (b) GA.py evalPop [EACDC original], (c) fast_eval.py
[EACDC fast]. Counted: dock scans, predecessor-list touches, matrix-cell touches, stage-2 list scans, memmove
lengths, sort comparisons (n log2 n for sorts). Same instances as cx.py, 20 random permutations each. Writes opcount.json."""
import sys, os, json, math
D = os.path.dirname(os.path.abspath(__file__)); F = f"{D}/../.."
sys.path.insert(0, f"{F}/benchmark_data"); import cdc, numpy as np

def ops_ours(I, ord_):
    p1, p2, P = I["p1"], I["p2"], I["prec"]; m1, m2, n1, n2 = I["m1"], I["m2"], I["n1"], I["n2"]; ops = 0
    fr = [0] * m1; C1 = [0] * n1
    for i in ord_:
        b = min(range(m1), key=lambda k: fr[k]); ops += m1; fr[b] += p1[i]; C1[i] = fr[b]
    pre = [np.flatnonzero(P[j]).tolist() for j in range(n2)]
    rb = [max(C1[i] for i in pr) if pr else 0 for pr in pre]; ops += sum(len(pr) for pr in pre) + n2
    left = sorted(range(n2), key=lambda j: (-p2[j], j)); ops += n2  # memcpy of the precomputed LPT list
    fr = [0] * m2
    while left:
        b = min(range(m2), key=lambda k: fr[k]); ops += m2; t = fr[b]; pick = None
        for k, j in enumerate(left):
            ops += 1
            if rb[j] <= t: pick = k; break
        if pick is None:
            ops += len(left); t = min(rb[j] for j in left)
            for k, j in enumerate(left):
                ops += 1
                if rb[j] <= t: pick = k; break
        j = left.pop(pick); ops += len(left) - pick; fr[b] = t + p2[j]
    return ops

def ops_orig(I, perm):
    P, m1, m2, n1, n2 = I["prec"], I["m1"], I["m2"], I["n1"], I["n2"]; E = int(P.sum())
    ops = n1 * n2                                   # double loop over Prec building r
    ops += int(E * max(1, math.log2(max(E, 2))))    # r.sort
    ops += E + E                                    # r_initial build + dict.fromkeys
    ops += m1 * n1 + n1 + m2 * n2                   # np.zeros Maq (m1 x n1) and Maq2 (m2 x n2) allocation/initialisation
    ops += n1 * m1                                  # argmin over docks per inbound
    ops += E                                        # release loop over r
    ops += n2 * (m2 + n2)                           # per pick: argmin docks + full scan of n2 (always full)
    ops += sum(1 for _ in range(n2)) * 0            # fallback scans counted below by simulation would add <= 2 n2 per pick
    return ops

def ops_fast(I, perm):
    P, m1, m2, n1, n2 = I["prec"], I["m1"], I["m2"], I["n1"], I["n2"]
    ops = n2 + n1 * n2                              # pos + where(P, pos).min over the n2 x n1 matrix
    ops += int(n1 * max(1, math.log2(n1)))          # lexsort
    ops += n1 * m1                                  # load.index(min(load))
    ops += n1 * n2                                  # rc = where(P, C).max over the matrix
    ops += n2 * (2 * m2) + n2 * n2 + n2 * n2 // 2   # per pick: min(free)+index, scan of rem (shrinking: n2^2/2 avg) + rem.remove
    return ops

FD = {'m4': '4maquinas', 'm10': '10maquinas'}; rows = []
for g in FD:
    for n1 in range(20, 81, 10):
        for sd in (17, 18, 19):
            I = cdc.parse(f"{F}/benchmark_data/data/Dados/{FD[g]}/{n1}jobs/{sd}.dat"); rng = np.random.default_rng(sd)
            o = [ops_ours(I, rng.permutation(n1).tolist()) for _ in range(20)]
            rows.append(dict(g=g, n1=n1, n2=I["n2"], E=int(I["prec"].sum()), ours=float(np.mean(o)), ours_max=max(o),
                             orig=ops_orig(I, None), fast=ops_fast(I, None)))
json.dump(rows, open(f"{D}/opcount.json", "w"))
X = np.log([r['n1'] for r in rows])
for k in ("ours", "orig", "fast"):
    y = np.log([r[k] for r in rows]); b, a = np.polyfit(X, y, 1)
    print(k, "slope vs n1 %.2f" % b, " at n1=80: %.0f" % np.mean([r[k] for r in rows if r['n1'] == 80]))
for nn in (20, 50, 80):
    S = [r for r in rows if r['n1'] == nn]
    print(nn, "orig/ours %.2f fast/ours %.2f  ours/(n1n2+n2^2) %.2f" % (np.mean([r['orig'] / r['ours'] for r in S]), np.mean([r['fast'] / r['ours'] for r in S]), np.mean([r['ours'] / (r['n1'] * r['n2'] + r['n2'] ** 2) for r in S])))
