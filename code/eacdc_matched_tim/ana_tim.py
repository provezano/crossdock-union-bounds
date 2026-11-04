import json, glob, sys, numpy as np
def boot(x, B=4000, rng=np.random.default_rng(0)):
    x = np.asarray(x, float); m = rng.choice(x, (B, len(x))).mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
for name in sys.argv[1:]:
    R = [json.load(open(f)) for f in glob.glob(f"{name}/*.json")]
    R = [r for r in R if r.get("tim") is not None]
    print(f"== {name}: n={len(R)}")
    known = np.array([max(r['lb_pch'], r['lb_e2'], r['lb_load'], r['lb_cs']) for r in R])
    u1 = np.array([r['lb_u1'] for r in R]); tim = np.array([np.ceil(r['tim'] - 1e-6) for r in R])
    ref = np.array([r.get('opt', r.get('ub_hgr')) for r in R])
    rows = [("TIM>LB_U*", tim > u1 + 1e-9), ("LB_U*>TIM", u1 > tim + 1e-9), ("TIM>known", tim > known + 1e-9), ("known>TIM", known > tim + 1e-9), ("LB_U*>known", u1 > known + 1e-9)]
    for k, v in rows: print(f"  {k:12s} {v.mean():.3f}")
    for k, v in (("TIM/LB_U*", tim / u1), ("TIM/known", tim / known), ("LB_U*/ref", u1 / ref), ("TIM/ref", tim / ref), ("known/ref", known / ref)):
        print("  %-10s mean %.4f [%.4f, %.4f]" % ((k,) + boot(v)))
    if 'tim_sumP' in R[0]:
        ts = np.array([np.ceil(r['tim_sumP'] - 1e-6) for r in R]); print("  TIM(T=sumP)/TIM(T=OPT) mean %.4f; tighter in %.3f" % ((ts / tim).mean(), (tim > ts).mean()))
        print("  OPT certified: %d/%d" % (sum(r['opt_status'] == 'OPTIMAL' for r in R), len(R)))
        print("  LB_U*==OPT %.3f  TIM==OPT %.3f" % ((u1 == ref).mean(), (tim == ref).mean()))
    print("  CPU s: TIM mean %.3f max %.2f | LB_U* mean %.3f max %.2f" % (np.mean([r['tim_cpu'] for r in R]), max(r['tim_cpu'] for r in R), np.mean([r['u1_cpu'] for r in R]), max(r['u1_cpu'] for r in R)))
    if 'n1' in R[0] and 'mt' in R[0]:
        for n1 in sorted({r['n1'] for r in R}):
            s = [i for i, r in enumerate(R) if r['n1'] == n1]
            print(f"   n1={n1}: TIM>U* {np.mean(tim[s] > u1[s]):.2f} U*>TIM {np.mean(u1[s] > tim[s]):.2f} TIM/U* {np.mean(tim[s]/u1[s]):.3f} U*/OPT {np.mean(u1[s]/ref[s]):.3f}")
