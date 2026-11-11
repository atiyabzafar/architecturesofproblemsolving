import sys, numpy as np
S = range(1, 41); out = {}
for f in sys.argv[1:]:
    D = np.load(f)
    for n in ("even", "uneven", "dense", "uneven_eq", "uneven_same"):
        if f"{n}|1|hold" not in D: continue
        tot = lambda q: np.sum([D[f"{n}|{s}|{q}"] for s in S], 0)
        hold, cacc, ev, first = tot("hold"), tot("cacc"), tot("ev"), tot("first")
        ncl = np.array([D[f"{n}|{s}|ncl_in"] for s in S]).mean(0)
        rp = hold[:, 0].sum(1) / cacc / 100; rm = hold[:, 1].sum(1) / cacc / 100
        a = np.arange(len(rp)); pk = int(np.argmax(rm[:400]))
        half = next(x for x in a if rp[x] <= rp[0] / 2); one = next(x for x in a if rp[x] < 0.01)
        reach = np.cumsum(first.sum(1) / cacc) / 100
        e = (ev[1:301] / cacc[1:301, None, None]).sum(0)
        newshare = e[2].sum() / (e[0].sum() + e[1].sum() + e[2].sum())
        evp = e[7].sum() / (e[0].sum() + e[1].sum() + e[7].sum())
        En = ev[:, 6].sum(1) / cacc
        line = (f"{n:12s} rho-: peak {100*rm[pk]:.1f}% at {pk}, at 200 {100*rm[200]:.1f}%, at 300 {100*rm[300]:.1f}% | rho+: 0 {100*rp[0]:.1f}%, half {half}, <1% {one} | "
                f"reach by 100 {100*reach[100]:.1f}% | negatives to agents without belief {newshare:.0%} | positives evicted {evp:.0%} | "
                f"neg evictions/period 51-100 {En[51:101].mean():.3f}, 101-150 {En[101:151].mean():.3f}")
        print(line)
        if ncl[0] > 0:
            kin = np.array([D[f"{n}|{s}|kin"] for s in S]); lam = kin / kin.mean(1, keepdims=True)
            lag = np.array([D[f"{n}|{s}|evlag"] for s in S]); evn = np.array([D[f"{n}|{s}|evn"] for s in S])
            cl = np.where(lam <= 0.5 + 1e-9, 0, np.where(lam >= 1.5 - 1e-9, 2, 1))
            for c, nm in enumerate(("low", "mid", "high")):
                r = hold[:, 1, c] / cacc / ncl[c]; p = int(np.argmax(r[:400])); k = cl == c
                rc = np.cumsum(first[:, c] / cacc) / ncl[c]
                pe = (e[0, c] + e[1, c]); print(f"    {nm:4s} ({ncl[c]:.0f} agents, mean intake {lam[k].mean():.2f}, share of messages {lam[k].sum()/lam.sum():.0%}): "
                      f"rho- peak {100*r[p]:.1f}% at {p} | rho+(0) {100*hold[0,0,c]/cacc[0]/ncl[c]:.1f}% | reach by 100 {100*rc[100]:.1f}% | "
                      f"memory time {lag[k].sum()/evn[k].sum():.0f} | positives evicted {e[7,c]/(pe+e[7,c]):.0%}")
