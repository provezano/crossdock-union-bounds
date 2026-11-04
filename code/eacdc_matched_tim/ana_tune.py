import json, glob, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
def boot(x, B=10000):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
for m in ("ils", "eacdc"):
    p = f"{D}/out_tune/{m}_trials.json"
    if not os.path.exists(p): continue
    T = json.load(open(p)); v = np.array([t['v'] for t in T]); rnd = v[1:12]
    print(f"== {m}: trials={len(T)} params={len(T[0]['prm'])} tuning CPU-h={sum(t['cpu'] for t in T)/3600:.2f}")
    print(f"  train objective mean UB/UB_HgR: default {v[0]:.4f}  best {v.min():.4f} (trial {int(v.argmin())}: {T[int(v.argmin())]['prm']})")
    print(f"  random-sample sensitivity (11 random configs): min {rnd.min():.4f} median {np.median(rnd):.4f} max {rnd.max():.4f}; worst vs best {100*(rnd.max()/v.min()-1):.2f}%  default vs best {100*(v[0]/v.min()-1):.2f}%; all trials worst vs best {100*(v.max()/v.min()-1):.2f}%")
    A, B = {}, {}
    for w, Dd in (("default", A), ("tuned", B)):
        for f in glob.glob(f"{D}/out_tune/test_{m}_{w}/*.json"):
            k = os.path.basename(f).rsplit("_s", 1)[0]; Dd.setdefault(k, []).append(json.load(open(f))['best'])
    ks = [k for k in A if k in B and len(A[k]) == 2 and len(B[k]) == 2]
    if ks:
        d = [(np.mean(B[k]) - np.mean(A[k])) / np.mean(A[k]) * 100 for k in ks]; mm, lo, hi = boot(d)
        print(f"  TEST (105 sample, 2 seeds, 4 s): tuned vs default {mm:+.3f}% [{lo:+.3f}, {hi:+.3f}] W/T/L {sum(x<0 for x in d)}/{sum(x==0 for x in d)}/{sum(x>0 for x in d)} (n={len(ks)})")
