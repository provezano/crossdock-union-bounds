"""Faithful Python port of Fonseca/Cota group's AMPL scripts MainJR.run (JRH), MainJLPTH.run (JLPTH)
and the LB block shared with MainVFR.run (HgR multi-dock). AMPL `for{j in S: cond} let x:=j` keeps the
LAST qualifying index -> ties broken by largest index; reproduced here. Reads Cota+2016 .dat files.
Reference implementations (not ours): third/fonseca_drive/*.run."""
import math, sys
import numpy as np

def parse(path):
    tok = [int(x) for x in open(path).read().split()]
    n1, n2, m1, m2 = tok[:4]
    p1 = tok[4:4+n1]; p2 = tok[4+n1:4+n1+n2]
    S = [tok[4+n1+n2+i*n1:4+n1+n2+(i+1)*n1] for i in range(n2)]  # S[i][j]: outbound i, inbound j
    return n1, n2, m1, m2, p1, p2, S

def lbs_lbr(n, m, maq1, maq2, p1, p2, S):
    pe1 = [max(sum(p1[i] for i in range(n) if S[j][i] > 0)/maq1, max(p1[i] for i in range(n) if S[j][i] > 0)) for j in range(m)]
    per2 = [max(sum(p2[i] for i in range(m) if S[i][j] > 0)/maq2, max(p2[i] for i in range(m) if S[i][j] > 0)) for j in range(n)]
    def smallest_k(vals, k):
        vals = list(vals); tot = 0.0; ind = None
        for _ in range(k):
            control = 10000
            for i, v in enumerate(vals):
                if v >= 0 and v <= control:
                    control = v; ind = i
            tot += vals[ind]; vals[ind] = -1
        return tot
    LBs = (smallest_k(pe1, maq2) + sum(p2)) / maq2
    LBr = (smallest_k(per2, maq1) + sum(p1)) / maq1
    return LBs, LBr, pe1, per2

def lb12(n, m, maq1, maq2, p1, p2, S, pe1, per2):
    c1 = [math.ceil(x) for x in pe1]; c2 = [math.ceil(x) for x in per2]
    LB1 = math.ceil(max(sum(p2)/maq2, max(p2)) + min(c1))
    LB2 = math.ceil(max(sum(p1)/maq1, max(p1)) + min(c2))
    return LB1, LB2

def johnson_list(n, m, p1, p2, S, ind_init):
    u = list(map(float, p1)); v = [0.0]*n; e = [0]*n; g = [0]*n
    for i in range(n):
        succ = [j for j in range(m) if S[j][i] == 1]
        v[i] = sum(p2[j] for j in succ) / len(succ)
        if u[i] < v[i]: e[i] = 1
        else: g[i] = 1
    el = [0]*n; gl = [0]*n  # store 1-based like AMPL (0 = empty)
    ind = ind_init  # AMPL carries `ind` over from the LB block (quirk reproduced)
    for j in range(n):
        control = 10000
        for i in range(n):
            if e[i] == 1 and u[i] >= 0 and u[i] <= control:
                control = u[i]; ind = i + 1
        if ind == -1: break
        u[ind-1] = -1; el[j] = ind; ind = -1
    for j in range(n):
        control = -1
        for i in range(n):
            if g[i] == 1 and v[i] >= 0 and v[i] >= control:
                control = v[i]; ind = i + 1
        if ind == -1: break
        v[ind-1] = -1; gl[j] = ind; ind = -1
    L = []; control = 0
    for i in range(n):
        if el[i] > 0: control += 1; L.append(el[i])
        else: L.append(gl[i-control])
    return [x-1 for x in L]

def two_stage(n, m, maq1, maq2, p1, p2, S, L, lpt):
    TX = [0]*maq1; rX = [0]*n
    for i in L:
        mn = min(TX); k = max(j for j in range(maq1) if TX[j] == mn)
        rX[i] = TX[k]; TX[k] += p1[i]
    rY = [max(rX[j]+p1[j] for j in range(n) if S[i][j] > 0) for i in range(m)]
    rYF = list(rY); TY = [0]*maq2; p2aux = list(p2)
    for _ in range(m):
        mn = min(TY); k = max(j for j in range(maq2) if TY[j] == mn)
        if (not lpt) or min(rY) > TY[k]:
            mr = min(rY); job = max(j for j in range(m) if rY[j] == mr)
        else:
            maior = -1
            for j in range(m):
                if rY[j] <= TY[k] and p2aux[j] >= maior: job = j; maior = p2aux[j]
        rYF[job] = max(rYF[job], TY[k]); rY[job] = 100000
        TY[k] = rYF[job] + p2[job]
    return max(TY), max(TX)

def run(path):
    n, m, maq1, maq2, p1, p2, S = parse(path)
    LBs, LBr, pe1, per2 = lbs_lbr(n, m, maq1, maq2, p1, p2, S)
    LB1, LB2 = lb12(n, m, maq1, maq2, p1, p2, S, pe1, per2)
    # value of `ind` at end of LBr loop (1-based index of last picked) -- AMPL quirk
    vals = list(per2); ind = None
    for _ in range(maq1):
        control = 10000
        for i, vv in enumerate(vals):
            if vv >= 0 and vv <= control: control = vv; ind = i + 1
        vals[ind-1] = -1
    L = johnson_list(n, m, p1, p2, S, ind)
    ubJR, c1 = two_stage(n, m, maq1, maq2, p1, p2, S, L, lpt=False)
    ubJL, _ = two_stage(n, m, maq1, maq2, p1, p2, S, L, lpt=True)
    nosucc = sum(1 for j in range(n) if not any(S[i][j] for i in range(m)))
    return dict(LBs=LBs, LBr=LBr, LB1=LB1, LB2=LB2, JRH=ubJR, JLPTH=ubJL, C1max=c1, nosucc=nosucc)

if __name__ == "__main__":
    print(run(sys.argv[1]))
