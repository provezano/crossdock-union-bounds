"""Item 5 table: per (group, n1) cell: binding term share, forward cut inbound-limited share, t*/LB, and the marginal
value of one more inbound vs outbound door (LB* drop and ILS-2 s UB drop, %). usage: ana_diag.py [TXT]"""
import json, glob, sys, collections, numpy as np
R = [json.load(open(f)) for f in glob.glob("out_diag/*.json")]; out = []
G = ["m2", "m4", "m10", "u24", "u210"]
def lab(t):
    b = max(t, key=t.get); return {"fwd": "union", "rev": "union", "a": "in-load", "b": "out-load", "pch": "head/tail", "e2": "head/tail"}[b]
out.append("| Cell | n | binds: union / in-load / out-load / head-tail | cut inbound-limited | t*/LB | +1 inbound door: LB / UB drop | +1 outbound door: LB / UB drop | door that pays more (UB) |")
out.append("|---|---|---|---|---|---|---|---|")
for g in G:
    for grp in ((20, 30, 40), (50, 60), (70, 80)):
        rs = [r for r in R if r["g"] == g and r["n1"] in grp]
        if not rs: continue
        c = collections.Counter(lab(r["base"]["terms"]) for r in rs)
        inl = np.mean([r["base"]["limit"] == "in" for r in rs if r["base"]["limit"]]) * 100 if any(r["base"]["limit"] for r in rs) else float("nan")
        ts = np.mean([r["base"]["tstar"] / r["base"]["lb"] for r in rs if r["base"]["tstar"]])
        d = lambda tag, f: np.mean([(r["base"][f] - r[tag][f]) / r["base"][f] * 100 for r in rs])
        wi = np.mean([r["m1p"]["ub2"] < r["m2p"]["ub2"] for r in rs]) * 100; wo = np.mean([r["m2p"]["ub2"] < r["m1p"]["ub2"] for r in rs]) * 100
        out.append(f"| {g}, n1 {grp[0]}–{grp[-1]} | {len(rs)} | {c['union']}/{c['in-load']}/{c['out-load']}/{c['head/tail']} | {inl:.0f}% | {ts:.2f} | "
                   f"{d('m1p','lb'):.1f}% / {d('m1p','ub2'):.1f}% | {d('m2p','lb'):.1f}% / {d('m2p','ub2'):.1f}% | inbound {wi:.0f}% · outbound {wo:.0f}% |")
txt = "\n".join(out); print(txt)
if len(sys.argv) > 1: open(sys.argv[1], "w").write(txt + "\n")
