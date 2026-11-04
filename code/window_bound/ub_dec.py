"""Phase 4 (UB, secondary): decoder headroom. For a fixed inbound schedule (from the ILS 10 s schedules of the step 5
exact-run bases (300 instances, n1 = 20/30) and of the step 6 ILS census, read-only), solve stage 2 exactly as
P|r_j|Cmax (CP-SAT, 1 worker) with r_j = max_{N(j)} C1_i, and also with stage 1 re-packed to minimise (left-shift).
Reports how much the ERD+LPT outbound rule leaves on the table. usage: ub_dec.py OUT SHARD NSH"""
import sys; sys.dont_write_bytecode = True
import os, json, glob, time
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, lbx, sset

def items():
    for f in sorted(glob.glob(f"{D}/snap/out_exact/*_base.json")):
        b = json.load(open(f)); yield b["key"], b["g"], b["n1"], b["seed"], b["ub_ils10"], b["s1"], b["s2"]
    if os.environ.get("UBDEC_CENSUS") != "1": return
    for f in sorted(glob.glob(f"{D}/../fbils_census_exact_cpuc/out_ils/ils_s1_*.jsonl")):
        for ln in open(f):
            try: r = json.loads(ln)
            except Exception: continue
            if r.get("src") == "run": yield r["key"], r["g"], r["n1"], r["seed"], r["ub10"], r["s1"], r["s2"]

def main():
    out, sh, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); os.makedirs(out, exist_ok=True)
    fo = f"{out}/dec_{sh}.jsonl"; done = set()
    if os.path.exists(fo): done = {json.loads(l)["key"] for l in open(fo)}
    f = open(fo, "a")
    for q_, (k, g, n1, sd, ub, s1, s2) in enumerate(items()):
        if q_ % ns != sh or k in done: continue
        I = lbx.parse(sset.path(g, n1, sd)); t0 = time.process_time()
        C1 = np.array(s1) + I["p1"]; P = I["prec"]
        r = np.where(P, C1[None, :], 0).max(axis=1).astype(np.int64)
        c2 = int(max((np.array(s2) + I["p2"]).max(), C1.max())); assert c2 == int(ub), (k, c2, ub)
        lb2, opt2 = lbx.stage_np_lb(I["p2"], r, I["m2"], tl=10.0, hi=c2)
        best = max(lb2, int(C1.max()))
        rec = dict(key=k, g=g, n1=n1, ub=ub, stage2_lb=int(lb2), stage2_opt=bool(opt2), dec_gain=(ub - best) / ub if opt2 else None, cpu=round(time.process_time() - t0, 2))
        f.write(json.dumps(rec) + "\n"); f.flush()
    print("done", flush=True)

if __name__ == "__main__":
    main()
