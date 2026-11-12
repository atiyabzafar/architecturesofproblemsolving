"""Figures and numbers for Section 3 (Results) of the npj draft.

Reads the runs written by "261109 results runs.py" (main.npz, ablation.npz,
nocomm.npz and, for Supplementary Note 1, si_generators.npz in RUNDIR) and, for
the mass-retraction numbers, the stored shock runs in "output/261001 timeseries.npz"
(written by "261001 analysis.py" run).

  python "261109 results figures.py" RUNDIR TIMESERIES

Writes 261109 fig2_selection, fig3_agents, fig4_misinformation, and for the
Supplementary Information figS1_networks, figS2_generators and figS3_generators (.pdf, .png)
to RUNDIR, and prints every number quoted in the draft.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(sys.argv[1]); TS = Path(sys.argv[2])
OUT = RUN
NETS = ["even", "uneven", "dense", "random", "smallworld", "scalefree", "layered"]
MAIN3 = ["even", "uneven", "dense"]
SHORT = {"even": "Regular", "uneven": "Scale Free", "dense": "Scale Free, dense", "random": "Random",
         "smallworld": "Small world", "scalefree": "Barabási–Albert", "layered": "Layered"}
COL = {"even": "#7f7f7f", "uneven": "#1f77b4", "dense": "#9467bd", "random": "#c9a227",
       "smallworld": "#17becf", "scalefree": "#e377c2", "layered": "#8c564b"}
VAR = {"full": ("full model", "#0b0b0b", "-"), "norec": ("no recency rule", "#eb6834", "-"),
       "noneg": ("no negative beliefs", "#1baf7a", "-"), "neither": ("neither", "#6250d6", "-"),
       "noneg120": ("no negative beliefs, memory 120", "#1baf7a", ":"),
       "noneg24": ("no negative beliefs, memory 24", "#1baf7a", "--")}
INK, SUB, MUTED = "#0b0b0b", "#52514e", "#a3a29c"
NSEEDS=40
S40, S20 = range(1, NSEEDS+1), range(1, 21)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "axes.edgecolor": SUB,
                     "xtick.color": SUB, "ytick.color": SUB, "axes.labelcolor": INK, "savefig.bbox": "tight",
                     "legend.frameon": False, "font.family": "sans-serif"})


# Journal style shared by Figures 2-4: 180 mm wide, 7-8 pt type, no grid, panel letters in bold.
# Networks: black / blue / orange (Okabe-Ito; distinct as greys in black-and-white print) plus marker shapes.
NETC = {"even": "#000000", "uneven": "#0072B2", "dense": "#E69F00"}
NETM = {"even": "o", "uneven": "s", "dense": "^"}
CUR, OUTD = "#4a4a4a", "#c9c9c9"          # current / outdated positive beliefs (bars)
JOURNAL = {"font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
           "axes.grid": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
           "xtick.major.size": 3, "ytick.major.size": 3, "lines.linewidth": 1.4, "axes.edgecolor": "#222222",
           "xtick.color": "#222222", "ytick.color": "#222222", "axes.labelcolor": "#000000", "legend.frameon": False}
plt.rcParams.update(JOURNAL)


def letter(ax, t):
    ax.set_title(t, loc="left", fontweight="bold", fontsize=9, pad=4)


def save300(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"261109 {name}.{ext}", dpi=300)
    print("wrote", OUT / f"261109 {name}.png")


def ci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def welch(a, b):
    return a.mean() - b.mean(), 1.96 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))


def topk_value(score, value, k, larger=True):
    """Mean `value` of the k agents ranked best by `score`; ties at the cut share the remaining places."""
    s = score if larger else -score
    order = np.sort(s)[::-1]; th = order[k - 1]
    strict = s > th; tie = s == th
    n_s = strict.sum(); rest = k - n_s
    return (value[strict].sum() + rest * value[tie].mean()) / k


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"261109 {name}.{ext}", dpi=200)
    print("wrote", OUT / f"261109 {name}.png")


# ============================================================ main batch
M = np.load(RUN / "main.npz") if (RUN / "main.npz").exists() else None
if M is not None:
    G = lambda n, q: np.array([M[f"{n}|{s}|{q}"] for s in S40])
    print("=" * 100); print(f"NETWORK LEVEL ({NSEEDS} runs per network; long run = periods 1501-2000)")
    sel = {}
    for n in NETS:
        avg = G(n, "avgV")[:, 1501:].mean(1); best_tick = G(n, "minV")[:, 1501:].mean(1)
        win = G(n, "win"); kin = G(n, "kin")
        Vsel, Veval = win[:, :10].mean(1), win[:, 10:].mean(1)        # 1001-1500, 1501-2000
        pos1 = np.array([topk_value(kin[i], Veval[i], 1) for i in range(NSEEDS)])
        pos5 = np.array([topk_value(kin[i], Veval[i], 5) for i in range(NSEEDS)])
        rec1 = np.array([topk_value(Vsel[i], Veval[i], 1, larger=False) for i in range(NSEEDS)])
        rec5 = np.array([topk_value(Vsel[i], Veval[i], 5, larger=False) for i in range(NSEEDS)])
        bestmember = Veval.min(1)
        hom = G(n, "hom")[:, 1501:].mean(1)
        sel[n] = dict(avg=avg, tick=best_tick, pos1=pos1, pos5=pos5, rec1=rec1, rec5=rec5, member=bestmember, hom=hom)
        print(f"  {SHORT[n]:18s} average {ci(avg)[0]:6.2f}+-{ci(avg)[1]:.2f} | best each period {ci(best_tick)[0]:6.2f} | "
              f"most-heard 1 {ci(pos1)[0]:6.2f}+-{ci(pos1)[1]:.2f}, 5 {ci(pos5)[0]:6.2f}+-{ci(pos5)[1]:.2f} | "
              f"best record 1 {ci(rec1)[0]:6.2f}+-{ci(rec1)[1]:.2f}, 5 {ci(rec5)[0]:6.2f}+-{ci(rec5)[1]:.2f} | "
              f"best member ex post {ci(bestmember)[0]:6.2f} | homogeneity {ci(hom)[0]:.3f}")
    NC = np.load(RUN / "nocomm.npz")
    V0s = np.array([NC[f"none|{s}|avgV"][1501:].mean() for s in S40]); V0 = V0s.mean()
    print(f"  {'No communication':18s} average {ci(V0s)[0]:6.2f}+-{ci(V0s)[1]:.2f} | value of communication: "
          f"Regular {V0 - sel['even']['avg'].mean():.2f}, Scale Free {V0 - sel['uneven']['avg'].mean():.2f}")
    print("  Differences from Regular (95% interval):")
    for n in NETS[1:]:
        line = []
        for q, lab in (("avg", "average"), ("tick", "best each period"), ("pos5", "5 most-heard"), ("rec5", "5 best record"),
                       ("pos1", "most-heard"), ("rec1", "best record"), ("hom", "homogeneity")):
            d, e = welch(sel[n][q], sel["even"][q]); line.append(f"{lab} {d:+.2f}+-{e:.2f}" if q != "hom" else f"{lab} {d:+.3f}+-{e:.3f}")
        print(f"    {SHORT[n]:18s} " + " | ".join(line))
    for n in NETS:
        d5 = sel[n]["avg"] - sel[n]["pos5"]; r5 = sel[n]["avg"] - sel[n]["rec5"]
        print(f"    {SHORT[n]:18s} 5 most-heard beat own average by {ci(d5)[0]:.2f}; 5 best-record by {ci(r5)[0]:.2f}")

    print("\nINFORMATION QUALITY (per agent, sampled every 10 periods over 1501-2000)")
    iq = {}
    for n in NETS:
        q = {k: G(n, k).mean(1) for k in ("cov", "true", "stale", "neg", "wneg", "vb", "vu")}
        iq[n] = q
        prec = q["true"] / (q["true"] + q["stale"])
        print(f"  {SHORT[n]:18s} current rules believed {ci(q['cov'])[0]:5.2f} of 60 | true 'rule exists' {ci(q['true'])[0]:5.2f} | "
              f"outdated 'rule exists' {ci(q['stale'])[0]:5.2f}+-{ci(q['stale'])[1]:.2f} | precision {prec.mean():.2f} | "
              f"'rule gone' {q['neg'].mean():5.1f} (wrong {q['wneg'].mean():.2f}) | believed rules violated "
              f"{(q['vb'] / q['cov']).mean():.0%}, unbelieved {(q['vu'] / (60 - q['cov'])).mean():.0%}")
    for n in NETS[1:]:
        dt = welch(iq[n]["true"], iq["even"]["true"]); ds = welch(iq[n]["stale"], iq["even"]["stale"])
        print(f"    {SHORT[n]:18s} minus Regular: true {dt[0]:+.2f}+-{dt[1]:.2f}, outdated {ds[0]:+.2f}+-{ds[1]:.2f}")

    print("\nAGENT LEVEL")
    EDGES = [0.12, 0.2, 0.3, 0.45, 0.65, 0.9, 1.25, 1.8, 2.6, 3.8, 5.5, 12]
    agent = {}
    for n in NETS:
        kin = G(n, "kin"); lam = kin / kin.mean(1, keepdims=True)
        V = G(n, "win")[:, 10:].mean(1)
        fields = {k: G(n, k) for k in ("cov", "true", "stale", "neg", "wneg")}
        x = np.log(lam).ravel(); y = V.ravel(); ok = np.isfinite(x)
        b = np.polyfit(x[ok], y[ok], 1)[0] if lam.std() > 0 else np.nan
        bins = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = (lam > lo) & (lam <= hi)
            if s.sum() >= 20:
                bins.append(dict(lam=np.exp(np.log(lam[s]).mean()), V=V[s].mean(), n=int(s.sum()),Vsigma=V[s].std(),Vci=(1.96 * V[s].std(ddof=1) / np.sqrt(len(V[s]))),
                                 **{k: f[s].mean() for k, f in fields.items()}))
        zero = (kin == 0)
        agent[n] = dict(bins=bins, slope=b, V0=V[zero].mean() if zero.any() else np.nan, n0=int(zero.sum()),Vsigma=V[zero].std())
        if lam.std() > 0:
            r = np.corrcoef(x[ok], y[ok])[0, 1]
            print(f"  {SHORT[n]:18s} slope per doubling of intake {-b * np.log(2):.2f} (R^2 {r * r:.2f}); "
                  f"agents with no in-link: {agent[n]['n0']} at {agent[n]['V0']:.2f}")
    for n in ("uneven",):
        print(f"  {SHORT[n]} by intake: " + "; ".join(
            f"{bb['lam']:.2f}: V {bb['V']:.1f}, true {bb['true']:.1f}, outdated {bb['stale']:.1f}, prec "
            f"{bb['true'] / (bb['true'] + bb['stale']):.2f}, gone {bb['neg']:.0f}" for bb in agent[n]["bins"]))

    # ---------------------------------------------------------------- Figure 2
    # Five agents with the fewest violations over periods 1001-1500 (selection period), followed over 1501-2000.
    rngt = np.random.default_rng(0)
    for n in NETS:
        win = G(n, "win"); kin = G(n, "kin")
        idx = np.argsort(win[:, :10].mean(1), 1)[:, :5]
        sel[n]["sel_series"] = np.array([win[i][:, idx[i]].mean(1) for i in range(NSEEDS)])     # (40, 20 windows)
        sel[n]["mean_series"] = win.mean(2)
        sel[n]["sel_before"] = np.take_along_axis(win[:, :10].mean(1), idx, 1).mean(1)
        ov = []
        for i in range(NSEEDS):
            order = np.lexsort((rngt.random(kin.shape[1]), -kin[i]))
            ov.append(len(set(idx[i]) & set(order[:5])))
        sel[n]["overlap"] = np.array(ov)
    print("\nFIVE BEST AGENTS OF PERIODS 1001-1500 (selection period), followed over 1501-2000")
    for n in NETS:
        print(f"  {SHORT[n]:18s} {sel[n]['sel_before'].mean():6.2f} in 1001-1500 -> {ci(sel[n]['rec5'])[0]:6.2f}+-{ci(sel[n]['rec5'])[1]:.2f} "
              f"in 1501-2000 | of these 5, {sel[n]['overlap'].mean():.1f} are among the 5 agents with the largest in-degree")

    print("\nHOMOGENEITY AND SPREAD OF INDIVIDUAL PERFORMANCE (periods 1501-2000)")
    Hp, Vp = [], []
    for n in MAIN3:
        H = G(n, "hom")[:, 1501:].mean(1); V = G(n, "avgV")[:, 1501:].mean(1); Hp += list(H); Vp += list(V)
        w = G(n, "win")[:, 10:].mean(1)
        sp = np.mean(np.percentile(w, 90, axis=1) - np.percentile(w, 10, axis=1))
        print(f"  {SHORT[n]:18s} H {ci(H)[0]:.3f}+-{ci(H)[1]:.3f} | across-run r(H, V) {np.corrcoef(H, V)[0, 1]:+.2f} | "
              f"sd of individual violations {w.std(1, ddof=1).mean():.2f}, 10th-90th percentile spread {sp:.2f}")
    print(f"  pooled over the three networks: r(H, V) = {np.corrcoef(Hp, Vp)[0, 1]:+.2f}")

    from matplotlib.lines import Line2D
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.08, 2.55), sharey=True, gridspec_kw=dict(width_ratios=[2.1, 1], wspace=0.08))
    rows = MAIN3[::-1]
    mk = [("avg", "Population mean", "o", "#000000", "#000000", 5.5),
          ("pos5", "5 agents with the largest in-degree", "D", None, None, 5.0),
          ("tick", "Top-performing agent in each period", "o", "white", "#8c8c8c", 5.0)]
    for yi, n in enumerate(rows):
        a1.axhline(yi, color="#dddddd", lw=0.6, zorder=0); a2.axhline(yi, color="#dddddd", lw=0.6, zorder=0)
        vals = [sel[n][q].mean() for q, *_ in mk]
        a1.plot([vals[1], vals[0]], [yi, yi], color=NETC[n], lw=1.6, alpha=0.55, zorder=1, solid_capstyle="butt")
        for q, lab, m, fc, ec, ms in mk:
            mu, e = ci(sel[n][q])
            a1.errorbar([mu], [yi], xerr=[[e], [e]], fmt=m, ms=ms, mfc=(NETC[n] if fc is None else fc),
                        mec=(NETC[n] if ec is None else ec), mew=1.1, ecolor=(NETC[n] if ec is None else ec),
                        elinewidth=0.7, capsize=0, zorder=3)
        mu, e = ci(sel[n]["hom"])
        a2.errorbar([mu], [yi], xerr=[[e], [e]], fmt="o", ms=5.5, mfc=NETC[n], mec=NETC[n], ecolor=NETC[n],
                    elinewidth=0.9, capsize=2, zorder=3)
    a1.set_yticks(range(len(rows))); a1.set_yticklabels([SHORT[n] for n in rows])
    a1.tick_params(axis="y", length=0); a2.tick_params(axis="y", length=0)
    a1.set_xlim(19.3, 30.4); a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_xlabel("Mean number of violations, periods 1501–2000")
    a1.legend([Line2D([], [], marker=m, ls="none", ms=ms, mfc=("#666666" if fc is None else fc), mec=("#666666" if ec is None else ec),
                      mew=1.1) for q, lab, m, fc, ec, ms in mk],
              [lab for q, lab, m, fc, ec, ms in mk], loc="upper left", bbox_to_anchor=(-0.27, -0.22), ncol=3, handletextpad=0.4,
              columnspacing=1.2)
    letter(a1, "a")
    a2.set_xlim(0.684, 0.726); a2.set_xticks([0.69, 0.70, 0.71, 0.72])
    a2.set_xlabel("Homogeneity, $H$")
    letter(a2, "b")
    save300(fig, "fig2_selection")

    # ---------------------------------------------------------------- Figure 3 (main networks only)
    fig, axs3 = plt.subplots(2, 2, figsize=(7.08, 5.0), gridspec_kw=dict(wspace=0.38, hspace=0.62))
    b1, b2, b3, b4 = axs3.ravel()
    for n, ls in (("uneven", "-"), ("dense", "--")):
        bb = agent[n]["bins"]
        b1.plot([x["lam"] for x in bb], [x["V"] for x in bb], ls=ls, marker=NETM[n], color=NETC[n], ms=2, lw=1.2, label=SHORT[n])
        b1.errorbar([x["lam"] for x in bb], [x["V"] for x in bb],[x["Vci"] for x in bb],ls=ls,lw=1,ecolor=NETC[n],elinewidth=1, capsize=2)
    b1.plot([1], [agent["even"]["bins"][0]["V"]], NETM["even"], ms=3, color=NETC["even"], label="Regular", zorder=5)
    b1.axhline(V0, color="#777777", lw=0.9, ls=":")
    b1.text(0.24, V0 + 0.12, "No communication", fontsize=6.5, color="#555555", ha="left", va="bottom")
    b1.set_xscale("log"); b1.set_xticks([0.25, 0.5, 1, 2, 4, 8]); b1.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    b1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b1.set_xlabel("Communication intake, $\\lambda_i$"); b1.set_ylabel("Mean number of violations, $V_i$")
    b1.set_ylim(24.2, 35.2); b1.set_yticks([26, 28, 30, 32])
    b1.legend(loc="upper right", handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2); letter(b1, "a")
    bb = agent["uneven"]["bins"]; xs = np.arange(len(bb))
    tr = np.array([x["true"] for x in bb]); st = np.array([x["stale"] for x in bb])
    b2.bar(xs, tr, color=CUR, width=0.7, label="Current")
    b2.bar(xs, st, bottom=tr, color=OUTD, width=0.7, label="Outdated")
    b2.set_xticks(xs); b2.set_xticklabels([f"{x['lam']:.2g}" for x in bb], rotation=45, ha="right")
    b2.set_xlabel("Communication intake, $\\lambda_i$ (Scale Free)"); b2.set_ylabel("Positive beliefs per agent")
    b2.set_ylim(0, 27); b2.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8); letter(b2, "b")
    order = ["even", "uneven", "dense"]; ys = np.arange(len(order))[::-1]
    t_ = np.array([iq[n]["true"].mean() for n in order]); s_ = np.array([iq[n]["stale"].mean() for n in order])
    b3.barh(ys, t_, color=CUR, height=0.6); b3.barh(ys, s_, left=t_, color=OUTD, height=0.6)
    for y, a, b in zip(ys, t_, s_):
        b3.text(a / 2, y, f"{a:.1f}", ha="center", va="center", fontsize=6.8, color="white")
        b3.text(a + b / 2, y, f"{b:.1f}", ha="center", va="center", fontsize=6.8, color="#000000")
    b3.set_yticks(ys); b3.set_yticklabels([SHORT[n].replace(", ", ",\n") for n in order]); b3.set_xlim(0, 27)
    b3.set_xlabel("Positive beliefs per agent"); letter(b3, "c")
    tt = np.arange(20) * 50 + 1025
    b4.axvspan(1000, 1500, color="#ececec", lw=0, zorder=0)
    for n in ("even", "uneven"):
        for ser, ls in ((sel[n]["sel_series"], "-"), (sel[n]["mean_series"], "--")):
            mu = ser.mean(0); e = 1.96 * ser.std(0, ddof=1) / np.sqrt(ser.shape[0])
            b4.fill_between(tt, mu - e, mu + e, color=NETC[n], alpha=0.12, lw=0, zorder=1)
            b4.plot(tt, mu, color=NETC[n], ls=ls, lw=1.3, zorder=2)
    b4.text(1250, 30.05, "Selection period", ha="center", fontsize=6.8, color="#333333")
    b4.text(1750, 30.05, "Following period", ha="center", fontsize=6.8, color="#333333")
    end = lambda n, q: sel[n][q].mean(0)[-4:].mean()
    b4.text(2015, end("uneven", "mean_series"), "Scale Free,\npopulation mean", color=NETC["uneven"], fontsize=6.5, va="center")
    b4.text(2015, (end("even", "mean_series") + end("even", "sel_series")) / 2 - 0.1, "Regular, selected\nagents and\npopulation mean",
            color=NETC["even"], fontsize=6.5, va="center")
    b4.text(2015, end("uneven", "sel_series"), "Scale Free,\nselected agents", color=NETC["uneven"], fontsize=6.5, va="center")
    b4.set_xlim(1000, 2000); b4.set_ylim(24, 30.6); b4.set_xticks([1000, 1250, 1500, 1750, 2000])
    b4.set_xlabel("Period"); b4.set_ylabel("Mean number of violations")
    letter(b4, "d")
    save300(fig, "fig3_agents")

# ============================================================ Section 3.3 runs (2,000 periods, 40 realisations)
B = np.load(RUN / "ablation.npz") if (RUN / "ablation.npz").exists() else None
if B is not None:
    from matplotlib.lines import Line2D
    H = lambda n, v, q: np.array([B[f"{n}|{v}|{s}|{q}"] for s in S40])
    NAMES = {"full": "baseline model", "norec": "no recency rule", "noneg120": "no negative beliefs, capacity 120",
             "noneg": "no negative beliefs, capacity 60", "noneg24": "no negative beliefs, capacity 24",
             "neither": "neither mechanism, capacity 60", "old": "earlier model (observation, no testing)"}
    AGES = (0, 100, 300, 600, 800)
    print("=" * 100)
    print(f"SECTION 3.3 RUNS ({NSEEDS} realisations, T = 2000; constraints retired at 1000-1199 followed through their first life;"
          " stocks and violations over periods 1501-2000)")
    coh = {}
    for n in ("even", "uneven"):
        Vf = H(n, "full", "avgV")[:, 1501:].mean(1)
        for v in NAMES:
            h, c, g = H(n, v, "coh_h"), H(n, v, "coh_c"), H(n, v, "coh_n")
            w = c / np.maximum(c.sum(0, keepdims=True), 1)
            hb = np.nansum(np.nan_to_num(h) * w, 0); nb = np.nansum(np.nan_to_num(g) * w, 0)
            coh[(n, v)] = (hb, nb, c.sum(0))
            st, tr, neg = (H(n, v, q)[:, 151:].mean() for q in ("ts_stale", "ts_true", "ts_neg"))
            V = H(n, v, "avgV")[:, 1501:].mean(1); d, e = welch(V, Vf)
            ages = H(n, v, "age_hist").sum(0); cdf = np.cumsum(ages) / max(ages.sum(), 1)
            half = next((a for a in range(len(hb)) if hb[a] <= hb[0] / 2), None)
            tenth = next((a for a in range(len(hb)) if hb[a] <= hb[0] / 10), None)
            cross = next((a for a in range(1, 400) if nb[a] >= hb[a]), None) if v in ("full", "norec") else None
            print(f"  {SHORT[n]:10s} {NAMES[v]:42s} V {V.mean():5.2f}+-{ci(V)[1]:.2f} (minus baseline {d:+.2f}+-{e:.2f}) | "
                  f"outdated {st:5.1f} current {tr:5.1f} negative {neg:5.1f} per agent | believers at age "
                  + "/".join(str(a) for a in AGES) + ": " + "/".join(f"{100 * hb[a]:.2f}" for a in AGES) +
                  f"% | halves {half}, tenth {tenth}" + (f", negatives exceed believers at {cross}" if cross else "") +
                  f" | outdated beliefs older than 500 periods {1 - cdf[500]:.0%}")
    T5 = np.load(TS)                    # shock runs: numbers for the text only (not plotted)
    for n in MAIN3:
        op = np.nanmean([T5[f"{n}|{s}|shock|old_pos"] for s in S40], 0)
        sp = np.nanmean([T5[f"{n}|{s}|shock|spread"] for s in S40], 0)
        half = next(a for a in range(len(op)) if op[a] <= op[0] / 2)
        cross = next((a for a in range(1, len(op)) if sp[a] >= op[a]), None)
        pk = int(np.nanargmax(sp))
        print(f"  shock ({SHORT[n]}, stored 261001 runs, 40 seeds): retired constraints believed by {100 * op[0]:.1f}% at the shock, "
              f"halved after {half} periods | new constraints overtake at {cross} | new constraints peak {100 * sp[pk]:.1f}% at {pk}")

    fig, (c1, c3) = plt.subplots(1, 2, figsize=(7.08, 2.7), gridspec_kw=dict(wspace=0.3))
    ages = np.arange(501)
    for n in ("even", "uneven"):
        hb, nb, _ = coh[(n, "full")]
        c1.plot(ages, 100 * hb[:501], color=NETC[n], lw=1.4, ls="-")
        c1.plot(ages, 100 * nb[:501], color=NETC[n], lw=1.2, ls="--")
    c1.legend([Line2D([], [], color=NETC["even"]), Line2D([], [], color=NETC["uneven"]),
               Line2D([], [], color="#555555", ls="-"), Line2D([], [], color="#555555", ls="--")],
              ["Regular", "Scale Free", "Positive belief, $\\rho_+$", "Negative belief, $\\rho_-$"], loc="upper right", ncol=2,
              columnspacing=1.0)
    c1.set_xlim(0, 500); c1.set_ylim(0, 55)
    c1.set_xlabel("Periods since retirement, $\\tau$"); c1.set_ylabel("Fraction of agents (%)"); letter(c1, "a")
    n = "uneven"
    show = ["full", "norec", "noneg120", "noneg24", "neither"]
    LBL = {"full": "Baseline model", "norec": "No recency rule", "noneg120": "No negative beliefs, capacity 120",
           "noneg24": "No negative beliefs, capacity 24", "neither": "Neither mechanism, capacity 60"}
    STY = {"full": ("#000000", "-"), "norec": ("#D55E00", (0, (5, 2))), "noneg120": ("#009E73", (0, (6, 2, 1.5, 2))),
           "noneg24": ("#009E73", (0, (1.2, 1.6))), "neither": ("#CC79A7", (0, (9, 2.5)))}
    for v in show:
        hb, _, cc = coh[(n, v)]; a = np.arange(len(hb)); ok = (cc > 0) & (a <= 800)
        c3.plot(a[ok], 100 * np.clip(hb[ok], 1e-4, None), color=STY[v][0], ls=STY[v][1], lw=1.4, label=LBL[v])
    c3.set_yscale("log"); c3.set_ylim(0.05, 9000); c3.set_xlim(0, 800)
    c3.set_yticks([0.1, 1, 10, 100]); c3.set_yticklabels(["0.1", "1", "10", "100"])
    c3.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    c3.set_xlabel("Periods since retirement, $\\tau$"); c3.set_ylabel("Agents with a positive belief, $\\rho_+$ (%)")
    c3.legend(loc="upper right", handlelength=3.0, labelspacing=0.3, borderaxespad=0.2); letter(c3, "b")
    save300(fig, "fig4_misinformation")

# ============================================================ Supplementary Note 1: the four standard generators
# Same measures as Sections 3.1-3.3, on Random, Small world, Barabasi-Albert and Layered, with Regular and Scale Free
# as references. Network-level and agent-level numbers come from main.npz; the life cycle of retired constraints from
# si_generators.npz ("261109 results runs.py" ... si). Writes 261109 figS2_generators and figS3_generators.
SIF = RUN / "si_generators.npz"
if M is not None and SIF.exists():
    from matplotlib.lines import Line2D
    SG = np.load(SIF)
    GEN = ["random", "smallworld", "scalefree", "layered"]
    SIN = ["even", "smallworld", "random", "layered", "scalefree", "uneven"]        # ordered by spread of intake
    SIC = {"random": "#009E73", "smallworld": "#CC79A7", "scalefree": "#D55E00", "layered": "#56B4E9",
           "even": NETC["even"], "uneven": NETC["uneven"]}
    SIM = {"random": "D", "smallworld": "v", "scalefree": "P", "layered": "X", "even": NETM["even"], "uneven": NETM["uneven"]}
    Q = lambda n, q: [SG[f"{n}|full|{s}|{q}"] for s in S40]

    def km_median(t, cens):
        """Kaplan-Meier median of durations t, with cens True where observation ended before the event."""
        o = np.argsort(t, kind="stable"); t, cens = t[o], cens[o]; n = len(t); s = 1.0; i = 0
        while i < n:
            j = i; d = 0
            while j < n and t[j] == t[i]:
                d += (not cens[j]); j += 1
            if d:
                s *= 1 - d / (n - i)
                if s <= 0.5:
                    return float(t[i])
            i = j
        return float("nan")

    life = {}
    for n in SIN:
        h, g, c = np.array(Q(n, "coh_h")), np.array(Q(n, "coh_n")), np.array(Q(n, "coh_c"))
        w = c / np.maximum(c.sum(0, keepdims=True), 1)
        hb = np.nansum(np.nan_to_num(h) * w, 0); nb = np.nansum(np.nan_to_num(g) * w, 0)
        last = np.concatenate(Q(n, "lin_last")); end = np.concatenate(Q(n, "lin_end")); ok = last >= 0
        cens = last[ok] >= end[ok] - 1
        pk = int(np.argmax(nb[:400]))
        life[n] = dict(hb=hb, nb=nb, med=km_median(last[ok] + 1.0, cens), cens=cens.mean(),
                       cross=next((a for a in range(1, 400) if nb[a] >= hb[a]), None), pk=pk)
    if B is not None:                                   # the Regular and Scale Free runs repeat those of Section 3.3
        same = all(np.allclose(life[n]["hb"][:801], coh[(n, "full")][0][:801]) for n in ("even", "uneven"))
        print(f"\nSUPPLEMENTARY NOTE 1 (life cycle of Regular and Scale Free identical to Section 3.3 runs: {same})")
    for n in SIN:
        kin = G(n, "kin"); cv2 = (kin.var(1) / kin.mean(1) ** 2).mean()
        s_, q_, L_ = sel[n], iq[n], life[n]
        d, e = welch(s_["avg"], sel["even"]["avg"])
        print(f"  {SHORT[n]:16s} Var(k)/<k>^2 {cv2:.2f} | V {ci(s_['avg'])[0]:.2f}+-{ci(s_['avg'])[1]:.2f} (minus Regular {d:+.2f}+-{e:.2f}) | "
              f"Vmin {ci(s_['tick'])[0]:.2f} | H {ci(s_['hom'])[0]:.3f}+-{ci(s_['hom'])[1]:.3f} | 5 largest in-degree "
              f"{ci(s_['pos5'])[0]:.2f}+-{ci(s_['pos5'])[1]:.2f} ({(s_['avg'] - s_['pos5']).mean():.2f} below mean) | 5 best of 1001-1500 "
              f"{s_['sel_before'].mean():.2f} -> {ci(s_['rec5'])[0]:.2f}+-{ci(s_['rec5'])[1]:.2f} ({(s_['avg'] - s_['rec5']).mean():.2f} below mean; "
              f"{s_['overlap'].mean():.1f} of 5 among largest in-degree)")
        print(f"  {'':16s} slope per doubling {-agent[n]['slope'] * np.log(2):.2f} | agents with no in-link {agent[n]['n0'] / (100 * NSEEDS):.1%}"
              + (f" at V {agent[n]['V0']:.2f}" if agent[n]["n0"] else "") + f" | current {q_['true'].mean():.1f}, outdated "
              f"{q_['stale'].mean():.1f} positive beliefs per agent | rho+(0) {100 * L_['hb'][0]:.1f}%, negatives exceed believers at "
              f"{L_['cross']} | last believer gone after {L_['med']:.0f} periods (median; {L_['cens']:.0%} of constraints re-entered or "
              f"outlived the run first) | rho- peak {100 * L_['nb'][L_['pk']]:.1f}% at {L_['pk']}, {100 * L_['nb'][200]:.1f}% at 200")
    print("  no communication: V {:.2f}".format(V0))
    print("  Table S1 (means over realisations): <k>, sd k_in, max k_in, Var(k)/<k>^2, corr(k_in,k_out), clustering, no in-link, largest SCC")
    for n in GEN:
        ns_ = G(n, "netstats").mean(0)
        print(f"    {SHORT[n]:16s} {ns_[0]:.1f} & {ns_[1]:.1f} & {ns_[2]:.0f} & {ns_[3]:.2f} & {ns_[4]:+.2f} & {ns_[5]:.2f} & "
              f"{100 * ns_[6]:.0f}\\% & {100 * ns_[7]:.0f}\\%")
    fig, axs = plt.subplots(2, 4, figsize=(7.08, 3.6), gridspec_kw=dict(wspace=0.35, hspace=0.45))
    for j, n in enumerate(["random", "smallworld", "scalefree", "layered"]):
        kin, kout = G(n, "kin")[0], G(n, "kout")[0]                          # one realisation (seed 1)
        top, bot = axs[0, j], axs[1, j]
        top.hist(kin, bins=np.arange(-0.5, kin.max() + 1.5), color=SIC[n], edgecolor="white", lw=0.4)
        top.set_title(SHORT[n], fontsize=7.5); top.set_xlabel("In-degree"); top.set_xlim(left=-0.5)
        bot.scatter(kin, kout, s=7, color=SIC[n], alpha=0.7, lw=0)
        mx = max(kin.max(), kout.max()) + 1; bot.set_xlim(-0.5, mx); bot.set_ylim(-0.5, mx)
        bot.set_xlabel("In-degree")
        for ax_ in (top, bot):
            ax_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
            ax_.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        if j == 0:
            top.set_ylabel("Agents"); bot.set_ylabel("Out-degree")
    letter(axs[0, 0], "a"); letter(axs[1, 0], "b")
    save300(fig, "figS1_networks")

    # ---------------------------------------------------------------- Figure S2: as Fig. 2, six networks
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.08, 3.4), sharey=True, gridspec_kw=dict(width_ratios=[2.1, 1], wspace=0.08))
    rows = SIN[::-1]
    for yi, n in enumerate(rows):
        a1.axhline(yi, color="#dddddd", lw=0.6, zorder=0); a2.axhline(yi, color="#dddddd", lw=0.6, zorder=0)
        vals = [sel[n][q].mean() for q, *_ in mk]
        a1.plot([vals[1], vals[0]], [yi, yi], color=SIC[n], lw=1.6, alpha=0.55, zorder=1, solid_capstyle="butt")
        for q, lab, m, fc, ec, ms in mk:
            mu, e = ci(sel[n][q])
            a1.errorbar([mu], [yi], xerr=[[e], [e]], fmt=m, ms=ms, mfc=(SIC[n] if fc is None else fc),
                        mec=(SIC[n] if ec is None else ec), mew=1.1, ecolor=(SIC[n] if ec is None else ec),
                        elinewidth=0.7, capsize=0, zorder=3)
        mu, e = ci(sel[n]["hom"])
        a2.errorbar([mu], [yi], xerr=[[e], [e]], fmt="o", ms=5.5, mfc=SIC[n], mec=SIC[n], ecolor=SIC[n],
                    elinewidth=0.9, capsize=2, zorder=3)
    a1.set_yticks(range(len(rows))); a1.set_yticklabels([SHORT[n] for n in rows])
    a1.tick_params(axis="y", length=0); a2.tick_params(axis="y", length=0)
    a1.set_xlim(19.3, 30.4); a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_xlabel("Mean number of violations, periods 1501–2000")
    a1.legend([Line2D([], [], marker=m, ls="none", ms=ms, mfc=("#666666" if fc is None else fc), mec=("#666666" if ec is None else ec),
                      mew=1.1) for q, lab, m, fc, ec, ms in mk],
              [lab for q, lab, m, fc, ec, ms in mk], loc="upper left", bbox_to_anchor=(-0.27, -0.16), ncol=3, handletextpad=0.4,
              columnspacing=1.2)
    letter(a1, "a")
    a2.set_xlim(0.684, 0.726); a2.set_xticks([0.69, 0.70, 0.71, 0.72]); a2.set_xlabel("Homogeneity, $H$"); letter(a2, "b")
    save300(fig, "figS2_generators")

    # ---------------------------------------------------------------- Figure S3: as Figs. 3a, 3c and 4a, six networks
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.0), gridspec_kw=dict(wspace=0.38, hspace=0.5))
    s1, s2, s3, s4 = axs.ravel()
    for n in ["uneven"] + GEN:
        bb = agent[n]["bins"]
        s1.plot([x["lam"] for x in bb], [x["V"] for x in bb], marker=SIM[n], color=SIC[n], ms=3.2, lw=1.1, label=SHORT[n])
    s1.plot([1], [agent["even"]["bins"][0]["V"]], NETM["even"], ms=4, color=NETC["even"], label="Regular", zorder=5)
    s1.axhline(V0, color="#777777", lw=0.9, ls=":")
    s1.text(0.23, V0 + 0.12, "No communication", fontsize=6.5, color="#555555", ha="left", va="bottom")
    s1.set_xscale("log"); s1.set_xticks([0.25, 0.5, 1, 2, 4, 8]); s1.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    s1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    s1.set_xlabel("Communication intake, $\\lambda_i$"); s1.set_ylabel("Mean number of violations, $V_i$")
    s1.set_ylim(21.5, 33.2); s1.set_yticks([22, 24, 26, 28, 30, 32])
    s1.legend(loc="lower left", handletextpad=0.4, labelspacing=0.2, borderaxespad=0.1, fontsize=6.5); letter(s1, "a")
    ys = np.arange(len(SIN))[::-1]
    t_ = np.array([iq[n]["true"].mean() for n in SIN]); o_ = np.array([iq[n]["stale"].mean() for n in SIN])
    s2.barh(ys, t_, color=CUR, height=0.62, label="Current"); s2.barh(ys, o_, left=t_, color=OUTD, height=0.62, label="Outdated")
    for y, a, b in zip(ys, t_, o_):
        s2.text(a / 2, y, f"{a:.1f}", ha="center", va="center", fontsize=6.5, color="white")
        s2.text(a + b / 2, y, f"{b:.1f}", ha="center", va="center", fontsize=6.5, color="#000000")
    s2.set_yticks(ys); s2.set_yticklabels([SHORT[n] for n in SIN]); s2.tick_params(axis="y", length=0)
    s2.set_xlim(0, 30); s2.set_xlabel("Positive beliefs per agent")
    s2.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, bbox_to_anchor=(1.0, 1.13)); letter(s2, "b")
    ages = np.arange(501)
    for ax, key, lab, let in ((s3, "hb", "Agents with a positive belief, $\\rho_+$ (%)", "c"),
                              (s4, "nb", "Agents with a negative belief, $\\rho_-$ (%)", "d")):
        for n in SIN:
            ax.plot(ages, 100 * life[n][key][:501], color=SIC[n], lw=1.2, marker=SIM[n], markevery=(25, 50), ms=3.2, label=SHORT[n])
        ax.set_xlim(0, 500); ax.set_xlabel("Periods since retirement, $\\tau$"); ax.set_ylabel(lab); letter(ax, let)
    s3.set_ylim(0, 30); s3.set_xlim(0, 300)
    s4.set_ylim(0, 48)
    s4.legend(loc="upper right", fontsize=6.3, labelspacing=0.15, handlelength=2.0, borderaxespad=0.1)
    save300(fig, "figS3_generators")
