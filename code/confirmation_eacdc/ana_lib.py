"""Shared helpers for the review-fix analyses (2026-11-02)."""
import numpy as np
REL_TOL = 1e-9

def at_cpu(trace, B):
    """Best value reached by CPU time B from a trace [(cpu, ..., best)]; None if no solution yet (NO fallback)."""
    v = [t[-1] for t in trace if t[0] <= B + 1e-9]
    return min(v) if v else None

def at_eval(trace, E, col=1):
    """Best value reached within E evaluations; trace entries (cpu, nev, best). None if none yet."""
    v = [t[-1] for t in trace if t[col] <= E]
    return min(v) if v else None

def wtl(a, b, tol=REL_TOL):
    """Win/tie/loss of a vs b (minimisation) with relative tolerance."""
    a, b = np.asarray(a, float), np.asarray(b, float); eps = tol * np.maximum(np.abs(b), 1.0)
    w = int(np.sum(a < b - eps)); l = int(np.sum(a > b + eps)); return w, len(a) - w - l, l

def boot(x, B=10000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def signflip_p(x, alt="less", B=100000, seed=1):
    """One-sided sign-flip permutation p-value for mean(x) < 0 ('less') or > 0 ('greater')."""
    x = np.asarray(x, float); rng = np.random.default_rng(seed); obs = x.mean()
    s = rng.choice([-1.0, 1.0], size=(B, len(x))); m = (s * x).mean(1)
    return float(((m <= obs).sum() + 1) / (B + 1)) if alt == "less" else float(((m >= obs).sum() + 1) / (B + 1))

def holm(ps):
    k = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(k): run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj
