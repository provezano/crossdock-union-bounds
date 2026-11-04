/* FB-ILS development: compiled decoder + search for P|M_j,prec|Cmax cross-dock (inbound permutation representation).
   dec(): exact port of g2.Dec (stage-1 list scheduling on earliest-free dock; releases = max pred completion;
   stage-2 non-idle ERD+LPT, ties by index). Searches (algo):
     0 ILS   : port of g2.ils (random insertion, accept <=, kick after stall > stall_mult*n: segment reversal + 2 insertions)
     1 IG    : Ruiz & Stuetzle 2007 iterated greedy: destroy d random jobs, greedy best-position reinsertion,
               insertion local search to a local optimum (random job order, first improvement), SA-like acceptance
               with Temperature = T * (P1+P2) / ((n1+n2) * 10)
     2 IG-B  : IG with bundle destruction: remove (up to d) predecessors of a random outbound truck (a shared-
               successor bundle); reinsertion order longest-p1 first
     3 LAHC-B: late-acceptance hill climbing (history length L) with insertion moves and, with prob pb, bundle
               moves (extract the predecessors of a random outbound truck and reinsert them contiguously)
   CPU budget = process CPU seconds (clock()); stops when cpu >= cpu_budget AND evals >= eval_budget, or cpu >= cap.
   Trace: (cpu, evals, best) at every improvement. */
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

typedef struct { int n1, n2, m1, m2; const int *p1, *p2, *pp, *pi; int *lpt; } Inst; /* pp/pi: CSR preds of outbound */
static double cpu(void) { return (double)clock() / CLOCKS_PER_SEC; }
static unsigned long long rs;
static inline unsigned long long xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline int rint_(int n) { return (int)(xr() % (unsigned long long)n); }
static inline double runif(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static long long NEV;
static int C1b[512], rb[512], fr[64], leftb[512], C2b[512], posb[512];
static int dec(const Inst *I, const int *ord) {
  NEV++;
  int m1 = I->m1, m2 = I->m2, cm = 0;
  for (int k = 0; k < m1; k++) fr[k] = 0;
  for (int q = 0; q < I->n1; q++) { int i = ord[q], b = 0; for (int k = 1; k < m1; k++) if (fr[k] < fr[b]) b = k;
    fr[b] += I->p1[i]; C1b[i] = fr[b]; if (fr[b] > cm) cm = fr[b]; }
  for (int j = 0; j < I->n2; j++) { int r = 0; for (int e = I->pp[j]; e < I->pp[j + 1]; e++) if (C1b[I->pi[e]] > r) r = C1b[I->pi[e]]; rb[j] = r; }
  int nl = I->n2; memcpy(leftb, I->lpt, sizeof(int) * nl);
  for (int k = 0; k < m2; k++) fr[k] = 0;
  while (nl) {
    int b = 0; for (int k = 1; k < m2; k++) if (fr[k] < fr[b]) b = k;
    int t = fr[b], pick = -1;
    for (int k = 0; k < nl; k++) if (rb[leftb[k]] <= t) { pick = k; break; }
    if (pick < 0) { int rm = 1 << 30; for (int k = 0; k < nl; k++) if (rb[leftb[k]] < rm) rm = rb[leftb[k]]; t = rm;
      for (int k = 0; k < nl; k++) if (rb[leftb[k]] <= t) { pick = k; break; } }
    int j = leftb[pick]; memmove(leftb + pick, leftb + pick + 1, sizeof(int) * (nl - pick - 1)); nl--;
    int c = t + I->p2[j]; fr[b] = c; C2b[j] = c; if (c > cm) cm = c;
  }
  return cm;
}
int decode_once(int n1, int n2, int m1, int m2, const int *p1, const int *p2, const int *pp, const int *pi, int *lpt, const int *ord) {
  Inst I = {n1, n2, m1, m2, p1, p2, pp, pi, lpt}; return dec(&I, ord);
}

static void ins(int *a, int n, int from, int to) { /* python: x = a.pop(from); a.insert(to, x) */
  int x = a[from];
  if (from < to) { if (to > n - 1) to = n - 1; memmove(a + from, a + from + 1, sizeof(int) * (to - from)); }
  else memmove(a + to + 1, a + to, sizeof(int) * (from - to));
  a[to] = x;
}
/* insert job x into seq a (length len) at its best position; returns new value; a becomes length len+1 */
static int best_insert(const Inst *I, int *a, int len, int x, int *tmp) {
  int bv = 1 << 30, bp = 0;
  for (int p = 0; p <= len; p++) {
    memcpy(tmp, a, sizeof(int) * p); tmp[p] = x; memcpy(tmp + p + 1, a + p, sizeof(int) * (len - p));
    if (len + 1 < I->n1) { /* partial sequence: evaluate with remaining jobs appended is not defined; use partial decode */ }
    int v = dec(I, tmp); if (v < bv) { bv = v; bp = p; }
  }
  memmove(a + bp + 1, a + bp, sizeof(int) * (len - bp)); a[bp] = x; return bv;
}

typedef struct { double *tc; long long *te; int *tv; int nt, maxt; double t0; } Tr;
static void rec(Tr *T, int v) { if (T->nt < T->maxt) { T->tc[T->nt] = cpu() - T->t0; T->te[T->nt] = NEV; T->tv[T->nt] = v; T->nt++; } }

/* partial-sequence decode for IG reconstruction: unscheduled jobs are appended in a fixed order (the removed set in
   current removal order) so every candidate is a full permutation. */
int search(int algo, int n1, int n2, int m1, int m2, const int *p1, const int *p2, const int *pp, const int *pi, int *lpt,
           int *order, double cpu_budget, long long eval_budget, double cap, unsigned long long seed,
           double fp1, double fp2, int ip1, int ip2,
           double *tc, long long *te, int *tv, int maxt, int *ntrace, long long *nev_out, long long nev0) {
  Inst I = {n1, n2, m1, m2, p1, p2, pp, pi, lpt};
  rs = seed * 2654435761ULL + 88172645463325252ULL; if (!rs) rs = 1; for (int k = 0; k < 10; k++) xr();
  NEV = nev0; double t0 = cpu(); Tr T = {tc, te, tv, 0, maxt, t0};
  int n = n1, *cur = malloc(sizeof(int) * n), *bo = malloc(sizeof(int) * n), *nb = malloc(sizeof(int) * n), *tmp = malloc(sizeof(int) * (n + 1));
  int *rem = malloc(sizeof(int) * n), *part = malloc(sizeof(int) * n), *full = malloc(sizeof(int) * (n + 1));
  memcpy(cur, order, sizeof(int) * n); int fc = dec(&I, cur), best = fc; memcpy(bo, cur, sizeof(int) * n); rec(&T, best);
  long long P = 0; for (int i = 0; i < n1; i++) P += p1[i]; for (int j = 0; j < n2; j++) P += p2[j];
  /* successor-sharing bundles: preds of outbound j = pi[pp[j]..pp[j+1]) */
  int *L = NULL; int Lh = 1;
  if (algo == 3 || (algo == 4 && ip2 > 0)) { Lh = ip2 > 0 ? ip2 : 50; L = malloc(sizeof(int) * Lh); for (int k = 0; k < Lh; k++) L[k] = fc; }
  long long it = 0; int stall = 0;
#define DONE() ((cpu() - t0 >= cpu_budget && NEV >= eval_budget) || cpu() - t0 >= cap)
  while (!DONE()) {
    it++;
    if (algo == 0 || algo == 5) {
      int a = rint_(n), b;
      if (algo == 5) { int w = (int)(fp2 * n); if (w < 1) w = 1; b = a + rint_(2 * w + 1) - w; if (b < 0) b = 0; if (b > n - 1) b = n - 1; if (runif() < fp1) b = rint_(n); }
      else b = rint_(n);
      if (a == b) continue;
      memcpy(nb, cur, sizeof(int) * n); ins(nb, n, a, b); int f = dec(&I, nb);
      if (f <= fc) { stall = f < fc ? 0 : stall + 1; memcpy(cur, nb, sizeof(int) * n); fc = f;
        if (f < best) { best = f; memcpy(bo, nb, sizeof(int) * n); rec(&T, best); } }
      else stall++;
      if (stall > ip1 * n) { memcpy(cur, bo, sizeof(int) * n); int x = rint_(n), y = rint_(n); if (x > y) { int s = x; x = y; y = s; }
        for (int l = x, r = y; l < r; l++, r--) { int s = cur[l]; cur[l] = cur[r]; cur[r] = s; }
        for (int k = 0; k < 2; k++) { int x2 = rint_(n), y2 = rint_(n); ins(cur, n, x2, y2); }
        fc = dec(&I, cur); stall = 0; }
    } else if (algo == 1 || algo == 2) {
      int d = ip1 < n ? ip1 : n - 1, nr = 0, len = 0;
      if (algo == 1) { memcpy(part, cur, sizeof(int) * n); len = n;
        for (int k = 0; k < d; k++) { int q = rint_(len); rem[nr++] = part[q]; memmove(part + q, part + q + 1, sizeof(int) * (len - q - 1)); len--; } }
      else { int j = rint_(n2); char *mk = calloc(n1, 1); int cnt = I.pp[j + 1] - I.pp[j];
        int idx[512], m = 0; for (int e = I.pp[j]; e < I.pp[j + 1]; e++) idx[m++] = I.pi[e];
        for (int k = m - 1; k > 0; k--) { int r = rint_(k + 1), s = idx[k]; idx[k] = idx[r]; idx[r] = s; }
        int take = cnt < d ? cnt : d; if (take < 2) take = cnt; for (int k = 0; k < take; k++) mk[idx[k]] = 1;
        for (int q = 0; q < n; q++) if (mk[cur[q]]) rem[nr++] = cur[q]; else part[len++] = cur[q];
        /* longest p1 first */
        for (int a = 1; a < nr; a++) { int x = rem[a], b2 = a - 1; while (b2 >= 0 && p1[rem[b2]] < p1[x]) { rem[b2 + 1] = rem[b2]; b2--; } rem[b2 + 1] = x; }
        free(mk); }
      /* greedy reconstruction: each removed job at its best position; still-unplaced removed jobs appended at end */
      int f = 0;
      for (int k = 0; k < nr; k++) {
        int bv = 1 << 30, bp = 0;
        for (int p = 0; p <= len; p++) {
          memcpy(full, part, sizeof(int) * p); full[p] = rem[k]; memcpy(full + p + 1, part + p, sizeof(int) * (len - p));
          memcpy(full + len + 1, rem + k + 1, sizeof(int) * (nr - k - 1));
          int v = dec(&I, full); if (v < bv) { bv = v; bp = p; }
          if (DONE()) break;
        }
        memmove(part + bp + 1, part + bp, sizeof(int) * (len - bp)); part[bp] = rem[k]; len++; f = bv;
      }
      if (len < n) { memcpy(part + len, rem + (nr - (n - len)), sizeof(int) * (n - len)); f = dec(&I, part); }
      /* local search: insertion neighbourhood, first improvement over a random job order, to a local optimum */
      if (fp2 > 0.5) { int improved = 1;
        while (improved && !DONE()) { improved = 0;
          int perm[512]; for (int k = 0; k < n; k++) perm[k] = part[k];
          for (int k = n - 1; k > 0; k--) { int r = rint_(k + 1), s = perm[k]; perm[k] = perm[r]; perm[r] = s; }
          for (int k = 0; k < n && !DONE(); k++) {
            int x = perm[k], pos = 0; while (part[pos] != x) pos++;
            memmove(part + pos, part + pos + 1, sizeof(int) * (n - pos - 1));
            int bv = 1 << 30, bp = pos;
            for (int p = 0; p < n; p++) { memcpy(full, part, sizeof(int) * p); full[p] = x; memcpy(full + p + 1, part + p, sizeof(int) * (n - 1 - p));
              int v = dec(&I, full); if (v < bv || (v == bv && p == pos)) { bv = v; bp = p; } }
            memmove(part + bp + 1, part + bp, sizeof(int) * (n - 1 - bp)); part[bp] = x;
            if (bv < f) { f = bv; improved = 1; }
          } } }
      double Temp = fp1 * (double)P / ((n1 + n2) * 10.0);
      if (f < fc) { fc = f; memcpy(cur, part, sizeof(int) * n); if (f < best) { best = f; memcpy(bo, part, sizeof(int) * n); rec(&T, best); } }
      else if (Temp > 0 && runif() < exp(-(f - fc) / Temp)) { fc = f; memcpy(cur, part, sizeof(int) * n); }
    } else if (algo == 4) { /* ILS + critical-predecessor moves (prob fp1); acceptance '<=' or LAHC (ip2 > 0) */
      memcpy(nb, cur, sizeof(int) * n);
      if (runif() < fp1) {
        dec(&I, cur); NEV--;  /* refresh C1b/C2b of the current order (state only; not counted as a search evaluation) */
        int k = I.m2 + 1; if (k > n2) k = n2; int top[64], nt = 0;  /* the k latest-finishing outbound trucks */
        for (int j = 0; j < n2; j++) { if (nt < k) top[nt++] = j; else { int w = 0; for (int q = 1; q < nt; q++) if (C2b[top[q]] < C2b[top[w]]) w = q; if (C2b[j] > C2b[top[w]]) top[w] = j; } }
        int j = top[rint_(nt)], ci = -1, cv = -1;
        for (int e = I.pp[j]; e < I.pp[j + 1]; e++) if (C1b[I.pi[e]] > cv) { cv = C1b[I.pi[e]]; ci = I.pi[e]; }
        if (ci < 0) continue; int a = 0; while (nb[a] != ci) a++; if (a == 0) continue;
        int b = rint_(a); ins(nb, n, a, b);
      } else { int a = rint_(n), b = rint_(n); if (a == b) continue; ins(nb, n, a, b); }
      int f = dec(&I, nb), acc;
      if (ip2 > 0) { int v = (int)(it % Lh); acc = (f <= fc || f <= L[v]); if (acc) { memcpy(cur, nb, sizeof(int) * n); fc = f; } L[v] = fc; }
      else { acc = f <= fc; if (acc) { stall = f < fc ? 0 : stall + 1; memcpy(cur, nb, sizeof(int) * n); fc = f; } else stall++; }
      if (acc && f < best) { best = f; memcpy(bo, nb, sizeof(int) * n); rec(&T, best); }
      if (ip2 <= 0 && stall > ip1 * n) { memcpy(cur, bo, sizeof(int) * n); int x = rint_(n), y = rint_(n); if (x > y) { int s2 = x; x = y; y = s2; }
        for (int l = x, r = y; l < r; l++, r--) { int s2 = cur[l]; cur[l] = cur[r]; cur[r] = s2; }
        for (int q = 0; q < 2; q++) { int x2 = rint_(n), y2 = rint_(n); ins(cur, n, x2, y2); }
        fc = dec(&I, cur); stall = 0; }
    } else { /* algo 3: LAHC with insertion + bundle moves */
      memcpy(nb, cur, sizeof(int) * n);
      if (runif() < fp1) { int j = rint_(n2); char mk[512] = {0}; for (int e = I.pp[j]; e < I.pp[j + 1]; e++) mk[I.pi[e]] = 1;
        int nr = 0, len = 0; for (int q = 0; q < n; q++) if (mk[nb[q]]) rem[nr++] = nb[q]; else part[len++] = nb[q];
        if (nr == 0 || len == 0) continue; int p = rint_(len + 1);
        memcpy(nb, part, sizeof(int) * p); memcpy(nb + p, rem, sizeof(int) * nr); memcpy(nb + p + nr, part + p, sizeof(int) * (len - p)); }
      else { int a = rint_(n), b = rint_(n); if (a == b) continue; ins(nb, n, a, b); }
      int f = dec(&I, nb), v = (int)(it % Lh);
      if (f <= fc || f <= L[v]) { memcpy(cur, nb, sizeof(int) * n); fc = f;
        if (f < best) { best = f; memcpy(bo, nb, sizeof(int) * n); rec(&T, best); } }
      L[v] = fc;  /* standard LAHC (Burke & Bykov): history stores the current value */
    }
  }
  memcpy(order, bo, sizeof(int) * n); *ntrace = T.nt; *nev_out = NEV;
  free(cur); free(bo); free(nb); free(tmp); free(rem); free(part); free(full); if (L) free(L);
  return best;
}

/* ---- algo 6: ILS + forward-backward improvement (FBI, 'double justification', Valls et al. 2005 for RCPSP) ----
   fbi(order): decode forward -> outbound completions C2; reversed order = outbound by decreasing C2 (ties by index);
   decode the reversed instance -> completion of each original inbound in reversed time; new forward order = inbound
   by decreasing reversed completion (i.e. increasing implied start); decode forward; repeat while the forward value
   strictly improves (max_rounds). Every decode is counted. Returns the best forward value; order is updated. */
static int ordr[512], keyv[512], ordf[512];
static void sort_desc(int *idx, int n, const int *key) { /* insertion sort by key desc, ties by index asc */
  for (int a = 1; a < n; a++) { int x = idx[a], b = a - 1; while (b >= 0 && (key[idx[b]] < key[x] || (key[idx[b]] == key[x] && idx[b] > x))) { idx[b + 1] = idx[b]; b--; } idx[b + 1] = x; }
}
static int fbi(const Inst *F, const Inst *R, int *order, int fval, int rounds) {
  int best = fval;
  for (int r = 0; r < rounds; r++) {
    dec(F, order); /* refresh C2b for current order (value known; counted) */
    for (int j = 0; j < F->n2; j++) { ordr[j] = j; keyv[j] = C2b[j]; }
    sort_desc(ordr, F->n2, keyv);
    dec(R, ordr);
    for (int i = 0; i < F->n1; i++) { ordf[i] = i; keyv[i] = C2b[i]; }
    sort_desc(ordf, F->n1, keyv);
    int v = dec(F, ordf);
    if (v < best) { best = v; memcpy(order, ordf, sizeof(int) * F->n1); } else break;
  }
  return best;
}
int search_fbi(int n1, int n2, int m1, int m2, const int *p1, const int *p2, const int *pp, const int *pi, int *lpt,
               const int *rpp, const int *rpi, int *rlpt, int *order, double cpu_budget, long long eval_budget, double cap,
               unsigned long long seed, double pfbi, int stall_mult, int rounds, double wfrac,
               double *tc, long long *te, int *tv, int maxt, int *ntrace, long long *nev_out, long long nev0) {
  Inst I = {n1, n2, m1, m2, p1, p2, pp, pi, lpt}, R = {n2, n1, m2, m1, p2, p1, rpp, rpi, rlpt};
  rs = seed * 2654435761ULL + 88172645463325252ULL; if (!rs) rs = 1; for (int k = 0; k < 10; k++) xr();
  NEV = nev0; double t0 = cpu(); Tr T = {tc, te, tv, 0, maxt, t0};
  int n = n1, *cur = malloc(sizeof(int) * n), *bo = malloc(sizeof(int) * n), *nb = malloc(sizeof(int) * n);
  memcpy(cur, order, sizeof(int) * n); int fc = dec(&I, cur); fc = fbi(&I, &R, cur, fc, rounds);
  int best = fc; memcpy(bo, cur, sizeof(int) * n); rec(&T, best); int stall = 0;
  while (!((cpu() - t0 >= cpu_budget && NEV >= eval_budget) || cpu() - t0 >= cap)) {
    int a = rint_(n), b;
    if (wfrac > 0 && runif() > 0.1) { int w = (int)(wfrac * n); if (w < 1) w = 1; b = a + rint_(2 * w + 1) - w; if (b < 0) b = 0; if (b > n - 1) b = n - 1; }
    else b = rint_(n);
    if (a == b) continue;
    memcpy(nb, cur, sizeof(int) * n); ins(nb, n, a, b); int f = dec(&I, nb);
    if (f <= fc) {
      int g = f; if (f < fc || runif() < pfbi) g = fbi(&I, &R, nb, f, rounds);
      stall = g < fc ? 0 : stall + 1; memcpy(cur, nb, sizeof(int) * n); fc = g;
      if (g < best) { best = g; memcpy(bo, nb, sizeof(int) * n); rec(&T, best); }
    } else stall++;
    if (stall > stall_mult * n) { memcpy(cur, bo, sizeof(int) * n); int x = rint_(n), y = rint_(n); if (x > y) { int s = x; x = y; y = s; }
      for (int l = x, r = y; l < r; l++, r--) { int s = cur[l]; cur[l] = cur[r]; cur[r] = s; }
      for (int k = 0; k < 2; k++) { int x2 = rint_(n), y2 = rint_(n); ins(cur, n, x2, y2); }
      fc = dec(&I, cur); stall = 0; }
  }
  memcpy(order, bo, sizeof(int) * n); *ntrace = T.nt; *nev_out = NEV; free(cur); free(bo); free(nb);
  return best;
}
