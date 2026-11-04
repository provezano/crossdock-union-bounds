"""Anytime profile experiment, analysis rule R1 (2026-11-03, written after the frozen ana_anytime.py crashed on 2 instances, before any endpoint was
seen): on u24_70_1 and u24_80_1 the HgR-with-LP run made only 2 decodes (HiGHS overran its budget, no LP solved,
stop = budget), so the targets hgr_1 = 2 and hgr_10 = 20 lie below the first trace entry of each search (14 constructive
decodes + start). Rule applied, the one of the earlier preregs (confirm_fbils.at): a search read below its first trace entry
takes its first trace value. The records are copied with this fill into OUT dirs; ana_anytime.py is run unchanged.
Optional EXCL keys are dropped (sensitivity run). usage: fill_first_schedule.py ANYDIR IGTDIR OUTANY OUTIGT [EXCL,...]"""
import sys, os, json, glob
a, t, oa, ot = sys.argv[1:5]; ex = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else set(); n = 0
for src, dst, igt in ((a, oa, False), (t, ot, True)):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(f"{src}/*.json"):
        x = json.load(open(f))
        if x["key"] in ex: continue
        for S in ([x] if igt else list(x["search"].values())):
            for k, v in S["at"].items():
                if v is None: S["at"][k] = S["trace"][0][2]; S["cpu_at"][k] = S["trace"][0][0]; n += 1
        json.dump(x, open(f"{dst}/{os.path.basename(f)}", "w"))
print("filled", n)
