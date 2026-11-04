"""#4 summaries: tiny-instance search (search.jsonl) and benchmark n1 in {20,30} (bench_cs.jsonl)."""
import json, math, os
from collections import Counter
D = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(f"{D}/search.jsonl")]
print("tiny instances:", len(R))
for kind in ("bench", "free"):
    S = [r for r in R if r['kind'] == kind]; c = lambda k: sum(r[k] for r in S)
    print(f" {kind:5s} n={len(S)}: L_CS > LB_U* (fwd) {c('cs_gt_old')}; ceil {c('cs_gt_old_ceil')} | BLB > LB* {c('blb_gt_lbstar')}; ceil {c('blb_gt_lbstar_ceil')} | "
          f"door-cap bound < L_CS {c('new_lt_cs')} | door-cap bound > OPT {c('new_gt_opt')} | LB_U* > OPT {c('old_gt_opt')} | "
          f"door-cap == max(LB_U*, L_CS) {sum(abs(r['new_f'] - max(r['old_f'], r['cs_f'])) < 1e-9 for r in S)}/{len(S)}")
B = [json.loads(l) for l in open(f"{D}/bench_cs.jsonl")]
bind = [r for r in B if math.ceil(r['lb_hgr'] - 1e-9) > max(r['f'], r['r'])]
print(f"benchmark n1 in {{20,30}}: {len(B)} instances, walk exact on {sum(r['f_exact'] and r['r_exact'] for r in B)}; "
      f"ceil(Chen-Song BLB) > LB* computed WITHOUT the Chen-Song term on {len(bind)} ({100*len(bind)/len(B):.1f}%); "
      f"max excess {100*max((math.ceil(r['lb_hgr'])-max(r['f'], r['r']))/max(r['f'], r['r']) for r in bind):.2f}%; "
      f"mean excess on those {100*sum((math.ceil(r['lb_hgr'])-max(r['f'], r['r']))/max(r['f'], r['r']) for r in bind)/len(bind):.2f}%")
print(" by cell:", dict(sorted(Counter(f"{r['g']}/{r['n1']}" for r in bind).items())))
