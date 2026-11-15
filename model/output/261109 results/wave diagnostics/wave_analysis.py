import sys, numpy as np
D = np.load(sys.argv[1]); S = range(1, 41)
NETS = [n for n in ("even", "uneven", "dense", "uneven_eq", "uneven_same") if f"{n}|1|hold" in D]
EVN = ["T_pn", "M_pn", "M_0n", "M_np", "M_0p", "T_nn", "E_n", "E_p"]
AG = [0, 10, 25, 50, 75, 100, 125, 150, 200, 300, 400]
CL = ["low", "mid", "high"]
def tot(n, q): return np.sum([D[f"{n}|{s}|{q}"] for s in S], 0)
for n in NETS:
    hold, cacc, ev, first = tot(n, "hold"), tot(n, "cacc"), tot(n, "ev"), tot(n, "first")
    sendn, sendp = tot(n, "sendn"), tot(n, "sendp")
    ncl = np.array([D[f"{n}|{s}|ncl_in"] for s in S]).mean(0)          # agents per in-class (same every seed)
    N = 100
    print("=" * 110); print(n, "agents per intake class (low/mid/high):", ncl)
    rp = hold[:, 0, :].sum(1) / cacc / N; rm = hold[:, 1, :].sum(1) / cacc / N
    print("  age:            " + " ".join(f"{a:6d}" for a in AG))
    print("  rho+ %          " + " ".join(f"{100*rp[a]:6.1f}" for a in AG))
    print("  rho- %          " + " ".join(f"{100*rm[a]:6.1f}" for a in AG))
    for c in range(3):
        if ncl[c] == 0: continue
        print(f"  rho+ {CL[c]:4s} %     " + " ".join(f"{100*hold[a,0,c]/cacc[a]/ncl[c]:6.1f}" for a in AG))
        print(f"  rho- {CL[c]:4s} %     " + " ".join(f"{100*hold[a,1,c]/cacc[a]/ncl[c]:6.1f}" for a in AG))
    reach = np.cumsum(first.sum(1) / cacc) / N
    print("  ever neg %      " + " ".join(f"{100*reach[a]:6.1f}" for a in AG))
    for c in range(3):
        if ncl[c] == 0: continue
        rc = np.cumsum(first[:, c] / cacc) / ncl[c]
        print(f"  ever neg {CL[c]:4s}% " + " ".join(f"{100*rc[a]:6.1f}" for a in AG))
    # event rates per constraint per period (counts of agents), in age windows
    W = [(1, 25), (26, 50), (51, 100), (101, 150), (151, 200), (201, 300), (301, 400)]
    print("  events per constraint per period (agents), age windows:", W)
    for e, nm in enumerate(EVN):
        r = ev[:, e, :].sum(1) / cacc
        print(f"    {nm:5s} " + " ".join(f"{r[a:b+1].mean():6.3f}" for a, b in W))
    rs = sendn.sum(1) / cacc
    # where do negatives come from, ages 1-300
    a, b = 1, 300
    tot_ev = (ev[a:b+1] / cacc[a:b+1, None, None]).sum(0)          # (event, class)
    corr = tot_ev[0].sum() + tot_ev[1].sum(); new = tot_ev[2].sum()
    print(f"  ages 1-300: negative acquisitions per constraint: by own test {tot_ev[0].sum():.1f}, correction by message {tot_ev[1].sum():.1f}, "
          f"to agents without a belief {new:.1f} -> share that corrects a positive belief {corr/(corr+new):.0%}")
    lossp = tot_ev[0].sum() + tot_ev[1].sum() + tot_ev[7].sum()
    print(f"  positive beliefs lost: corrected by test {tot_ev[0].sum():.1f}, by message {tot_ev[1].sum():.1f}, evicted {tot_ev[7].sum():.1f}; "
          f"re-infections neg->pos {tot_ev[3].sum():.1f}, new pos by message {tot_ev[4].sum():.1f}")
    for c in range(3):
        if ncl[c] == 0: continue
        cp, mp, ep = tot_ev[0, c], tot_ev[1, c], tot_ev[7, c]
        print(f"    {CL[c]:4s}: positives corrected {cp+mp:.2f} (test {cp:.2f}, message {mp:.2f}), evicted {ep:.2f} -> evicted share {ep/(cp+mp+ep):.0%} | "
              f"negatives gained: correction {tot_ev[0,c]+tot_ev[1,c]:.2f}, new {tot_ev[2,c]:.2f}, refreshed by test {tot_ev[5,c]:.2f}; lost by eviction {tot_ev[6,c]:.2f}")
    # memory time and turnover by class
    kin = np.array([D[f"{n}|{s}|kin"] for s in S]); lam = kin / kin.mean(1, keepdims=True)
    ins = np.array([D[f"{n}|{s}|ins"] for s in S]); evn = np.array([D[f"{n}|{s}|evn"] for s in S]); lag = np.array([D[f"{n}|{s}|evlag"] for s in S])
    cl = np.where(lam <= 0.5 + 1e-9, 0, np.where(lam >= 1.5 - 1e-9, 2, 1))
    for c in range(3):
        k = cl == c
        if k.sum() == 0: continue
        print(f"    {CL[c]:4s}: new beliefs per period {ins[k].mean():.2f}; memory time (periods from last touch to eviction) {lag[k].sum()/evn[k].sum():.0f}")
    print("  negative transmissions per constraint per period, by age window: " + " ".join(f"{rs[a:b+1].mean():.3f}" for a, b in W))
