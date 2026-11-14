import sys, numpy as np
D = np.load(sys.argv[1]); S = range(1, 6)
NAMES = {"full": "with negative beliefs (baseline, capacity 120)", "noneg120": "without, capacity 120",
         "noneg": "without, capacity 60", "noneg24": "without, capacity 24"}
def km(t, cens):
    """Kaplan-Meier survival; returns median and restricted mean up to the largest observed time."""
    o = np.argsort(t); t, cens = t[o], cens[o]; n = len(t); s = 1.0; times, surv = [0.0], [1.0]
    i = 0
    while i < n:
        j = i; d = 0
        while j < n and t[j] == t[i]:
            d += (not cens[j]); j += 1
        if d: s *= 1 - d / (n - i); times.append(t[i]); surv.append(s)
        i = j
    times, surv = np.array(times), np.array(surv)
    med = next((tt for tt, ss in zip(times, surv) if ss <= 0.5), np.nan)
    p90 = next((tt for tt, ss in zip(times, surv) if ss <= 0.1), np.nan)
    rmean = np.sum(np.diff(np.append(times, t.max())) * surv)
    return med, p90, rmean
for n, lab in (("uneven", "Scale Free"), ("even", "Regular")):
    print(lab)
    for v in NAMES:
        L = np.vstack([D[f"{n}|{v}|{s}|lin"] for s in S]); A = np.vstack([D[f"{n}|{v}|{s}|ag"] for s in S])
        believed = L[:, 0] > 0
        t, c = L[believed, 0], L[believed, 2].astype(bool)
        med, p90, rm = km(t, c)
        am, a90, arm = km(A[:, 0], A[:, 1].astype(bool))
        print(f"  {NAMES[v]:48s} constraints believed at retirement {believed.mean():.0%} | lingers in population: mean {rm:5.0f}, "
              f"median {med:5.0f}, 90% gone by {p90:5.0f} periods ({c.mean():.0%} still believed when observation ended) | "
              f"one agent keeps it: mean {arm:4.0f}, median {am:4.0f}")
