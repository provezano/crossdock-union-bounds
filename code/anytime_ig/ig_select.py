"""anytime profile experiment: IG configuration selection on DEV (rule fixed in the prereg before any tuning run).
grid  : score(c) = mean over the 105 DEV instances and the budgets 1x, 10x E0 of RPD = 100 (v - best)/best, where best = the
        best value of any grid configuration on that instance and budget. The 5 lowest scores are the finalists.
final : finalists re-run to 100x E0 (same seed: their 1x/10x values are identical prefixes); final score = mean of the
        three budget-wise RPDs (100x RPD relative to the best finalist). Lowest wins; ties -> lower 100x RPD.
usage: ig_select.py grid OUTGRID  -> finalists.json ;  ig_select.py final OUTGRID OUTFINAL -> ig_tuned.json"""
import sys, os, json, glob, collections
D = os.path.dirname(os.path.abspath(__file__))
def load(d): return [json.load(open(f)) for f in sorted(glob.glob(f"{d}/*.json"))]
def name(c): return f"d{c['d']}_T{c['T']}_ls{c['ls']}"
def rpd(rows, ks):
    best = collections.defaultdict(lambda: float("inf"))
    for x in rows:
        for k in ks: best[(x["key"], k)] = min(best[(x["key"], k)], x["vals"][k])
    sc = collections.defaultdict(list)
    for x in rows:
        for k in ks: sc[name(x["cfg"])].append((k, 100 * (x["vals"][k] - best[(x["key"], k)]) / best[(x["key"], k)]))
    return sc
mode = sys.argv[1]; G = load(sys.argv[2]); cfgs = {name(x["cfg"]): x["cfg"] for x in G}
assert len(G) == 105 * 72, len(G)
sg = rpd(G, ["1", "10"]); score = {c: sum(v for _, v in L) / len(L) for c, L in sg.items()}
rank = sorted(score, key=score.get)
if mode == "grid":
    json.dump([cfgs[c] for c in rank[:5]], open(f"{D}/finalists.json", "w"))
    for c in rank[:12]: print(c, round(score[c], 4))
else:
    F = load(sys.argv[3]); fin = {name(x["cfg"]) for x in F}; assert len(F) == 105 * len(fin)
    s100 = rpd(F, ["100"])
    per = {}
    for c in fin:
        r1 = [v for k, v in sg[c] if k == "1"]; r10 = [v for k, v in sg[c] if k == "10"]; r100 = [v for _, v in s100[c]]
        per[c] = dict(rpd1=sum(r1) / 105, rpd10=sum(r10) / 105, rpd100=sum(r100) / 105)
        per[c]["final"] = (per[c]["rpd1"] + per[c]["rpd10"] + per[c]["rpd100"]) / 3
    win = min(per, key=lambda c: (per[c]["final"], per[c]["rpd100"]))
    json.dump(dict(cfg=cfgs[win], name=win, finalists=per, grid_top12={c: score[c] for c in rank[:12]},
                   default_grid_score=score.get("d4_T0.4_ls1"), n_configs=72, n_dev=105, seed=7),
              open(f"{D}/ig_tuned.json", "w"), indent=1)
    print("winner", win, per[win])
