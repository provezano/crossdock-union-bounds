"""structured-instance experiment matched UB runs (prereg): confirmation_eacdc/g4_match.py protocol unchanged (one full static EACDC run and
ILS in one process, first arm alternating, ILS read at EACDC's CPU and evalPop count); only HORIZON (ILS-first fixed
horizon = 1.5 x max full-EACDC CPU on the PILOT set generated with seed 999, never the test set) is extended to the
structured n1 values. usage: match_struct.py LIST OUT SEEDS SHARD NSH"""
import sys, os, json
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{D}/../confirmation_eacdc")
import g4_match
H = json.load(open(f"{D}/horizon_struct.json"))
g4_match.HORIZON = {int(k): float(v) for k, v in H.items()}
if __name__ == "__main__":
    g4_match.main()
