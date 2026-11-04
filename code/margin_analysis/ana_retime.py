"""Task 1: same-machine re-timing of exact v3 vs v2lbu (55 co-solved instances, TL 60 s, one host). usage: ana_retime.py"""
import json, glob, collections, numpy as np
R = [json.load(open(f)) for f in glob.glob("retime/*.json")]
print("n", len(R), "hosts", collections.Counter(r["host"] for r in R))
for a in ("v3", "v2lbu"):
    s = [r["res"][a]["opt"] is not None for r in R]; print(a, "solved within 60 s:", sum(s), "of", len(R))
c = [r for r in R if r["res"]["v3"]["opt"] is not None and r["res"]["v2lbu"]["opt"] is not None]
for r in c: assert r["res"]["v3"]["opt"] == r["res"]["v2lbu"]["opt"]
t3 = np.array([r["res"]["v3"]["cpu"] for r in R]); t2 = np.array([r["res"]["v2lbu"]["cpu"] for r in R])
# censored at 60: medians over all 55 with unsolved counted as >= 60 (valid when median < 60)
print("median CPU to proof (unsolved = censored at 60 s): v3 %.1f s, v2lbu %.1f s" % (np.median(t3), np.median(t2)))
r = np.array([r["res"]["v3"]["cpu"] / r["res"]["v2lbu"]["cpu"] for r in c])
print("both solved within 60 s: %d; median %.1f vs %.1f s; geo-mean ratio %.3f; v3 faster on %.0f%%" % (len(c), np.median([x['res']['v3']['cpu'] for x in c]), np.median([x['res']['v2lbu']['cpu'] for x in c]), np.exp(np.log(r).mean()), (r < 1).mean() * 100))
rr = np.minimum(t3, 60) / np.minimum(t2, 60)
print("geo-mean ratio over all 55 with censoring at 60 s (lower bound on speed-up): %.3f" % np.exp(np.log(np.maximum(rr, 1e-3)).mean()))
print("only v3:", sum(r["res"]["v3"]["opt"] is not None and r["res"]["v2lbu"]["opt"] is None for r in R), " only v2lbu:", sum(r["res"]["v3"]["opt"] is None and r["res"]["v2lbu"]["opt"] is not None for r in R))
print("total CPU h: %.2f" % ((t3.sum() + t2.sum()) / 3600))
