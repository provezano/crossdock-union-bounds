"""Output-preserving rewrite of EA-ML GA.load_individuals (Improved == 0 path, the one GA() uses):
same swap order, same restart rule, same float threshold rule for inbound docks, same ERD+LPT stage 2,
same (start, -p2) stable sort -> identical list of individuals. Pure-Python lists instead of per-swap numpy."""
def make(GA):
    def load_individuals(creator, n):
        n1, n2, maq1, maq2 = GA.n1, GA.n2, GA.maq1, GA.maq2
        p1 = list(GA.p1); p2 = list(GA.p2); P = GA.Prec
        preds = [[j for j in range(n1) if P[i][j] == 1] for i in range(n2)]
        pc = []
        for i in range(n1):
            s = 0; maxx = -1
            for j in range(n2):
                if P[j][i] == 1:
                    s += p2[j]
                    if p2[j] > maxx: maxx = p2[j]
            pc.append(maxx if float(s / maq2) < maxx else float(s / maq2))
        r = sorted([(i, pc[i], p1[i]) for i in range(n1)], reverse=True, key=lambda t: (t[1], t[2]))
        PCH = 0; Pacum = 0; sol1 = []
        for key, v1, v2 in r:
            Pacum += v2; PCH = max(Pacum + v1, PCH); sol1.append(key)
        individuals = []; FOMIN = 100000000
        i1 = 0
        while i1 < n1 - 1:
            i2 = i1
            while i2 <= n1 - 1:
                aux = sol1[i1]; sol1[i1] = sol1[i2]; sol1[i2] = aux
                PCH0 = p1[sol1[0]] + pc[sol1[0]]
                PCHC = PCH0 + (PCH - PCH0) / maq1
                MaqAc = [0.0] * maq1; C = [0.0] * n1; k = 0
                for i in range(n1):
                    s = sol1[i]
                    if MaqAc[k] + p1[s] + pc[s] >= PCHC and k < maq1 - 1: k += 1
                    MaqAc[k] = MaqAc[k] + p1[s]; C[s] = MaqAc[k]
                rc = [max([0.0] + [C[j] for j in preds[i]]) for i in range(n2)]
                MaqAc2 = [0] * maq2; C2 = [0] * n2; p2aux = list(p2); rcaux = list(rc)
                for _ in range(n2):
                    t = min(MaqAc2); k = MaqAc2.index(t)
                    p2MAX = -1; ind = -1
                    for j in range(n2):
                        if rc[j] <= t and p2aux[j] > p2MAX: p2MAX = p2aux[j]; ind = j
                    if ind == -1:
                        p2MAX = -1; rMIN = min(rcaux); ind = rcaux.index(rMIN)
                        for j in range(n2):
                            if rcaux[j] == rMIN and p2aux[j] > p2MAX: p2MAX = p2aux[j]; ind = j
                    MaqAc2[k] = int(max(t, rc[ind]) + p2[ind]); C2[ind] = MaqAc2[k]
                    p2aux[ind] = -2; rcaux[ind] = 1000000
                sol2 = [key for key, a, b in sorted([(i, C2[i] - p2[i], -p2[i]) for i in range(n2)], key=lambda t: (t[1], t[2]))]
                CMAXBL = max(C2)
                if CMAXBL < FOMIN:
                    FOMIN = CMAXBL; i1 = 0; i2 = 0
                else:
                    sol1[i2] = sol1[i1]; sol1[i1] = aux
                individuals.append(creator(sol2))
                i2 += 1
            i1 += 1
        return individuals
    return load_individuals
