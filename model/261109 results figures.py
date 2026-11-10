"""Figures and numbers for Section 3 (Results) of the npj draft.

Reads the runs written by "261109 results runs.py" (main.npz, ablation.npz,
nocomm.npz in RUNDIR) and, for the mass-retraction numbers, the stored shock
runs in "output/261001 timeseries.npz" (written by "261001 analysis.py" run).

  python "261109 results figures.py" RUNDIR TIMESERIES

Writes 261109 fig2_selection, fig3_agents, fig4_misinformation (.pdf, .png)
to RUNDIR and prints every number quoted in the draft.
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
S40, S20 = range(1, 41), range(1, 21)
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
    print("=" * 100); print("NETWORK LEVEL (40 runs per network; long run = periods 1501-2000)")
    sel = {}
    for n in NETS:
        avg = G(n, "avgV")[:, 1501:].mean(1); best_tick = G(n, "minV")[:, 1501:].mean(1)
        win = G(n, "win"); kin = G(n, "kin")
        Vsel, Veval = win[:, :10].mean(1), win[:, 10:].mean(1)        # 1001-1500, 1501-2000
        pos1 = np.array([topk_value(kin[i], Veval[i], 1) for i in range(40)])
        pos5 = np.array([topk_value(kin[i], Veval[i], 5) for i in range(40)])
        rec1 = np.array([topk_value(Vsel[i], Veval[i], 1, larger=False) for i in range(40)])
        rec5 = np.array([topk_value(Vsel[i], Veval[i], 5, larger=False) for i in range(40)])
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
                bins.append(dict(lam=np.exp(np.log(lam[s]).mean()), V=V[s].mean(), n=int(s.sum()),
                                 **{k: f[s].mean() for k, f in fields.items()}))
        zero = (kin == 0)
        agent[n] = dict(bins=bins, slope=b, V0=V[zero].mean() if zero.any() else np.nan, n0=int(zero.sum()))
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
        sel[n]["sel_series"] = np.array([win[i][:, idx[i]].mean(1) for i in range(40)])     # (40, 20 windows)
        sel[n]["mean_series"] = win.mean(2)
        sel[n]["sel_before"] = np.take_along_axis(win[:, :10].mean(1), idx, 1).mean(1)
        ov = []
        for i in range(40):
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
          ("rec5", "5 agents with the fewest violations in periods 1001–1500", "s", "white", None, 5.0),
          ("tick", "Top-performing agent in each period", "o", "white", "#8c8c8c", 5.0)]
    for yi, n in enumerate(rows):
        a1.axhline(yi, color="#dddddd", lw=0.6, zorder=0); a2.axhline(yi, color="#dddddd", lw=0.6, zorder=0)
        vals = [sel[n][q].mean() for q, *_ in mk]
        a1.plot([min(vals[1:3]), vals[0]], [yi, yi], color=NETC[n], lw=1.6, alpha=0.55, zorder=1, solid_capstyle="butt")
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
              [lab for q, lab, m, fc, ec, ms in mk], loc="upper left", bbox_to_anchor=(-0.27, -0.22), ncol=2, handletextpad=0.4,
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
        b1.plot([x["lam"] for x in bb], [x["V"] for x in bb], ls=ls, marker=NETM[n], color=NETC[n], ms=3.8, lw=1.2, label=SHORT[n])
    b1.plot([1], [agent["even"]["bins"][0]["V"]], NETM["even"], ms=5, color=NETC["even"], label="Regular", zorder=5)
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
    print("SECTION 3.3 RUNS (40 realisations, T = 2000; constraints retired at 1000-1199 followed through their first life;"
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
    ages = np.arange(301)
    for n in ("even", "uneven"):
        hb, nb, _ = coh[(n, "full")]
        c1.plot(ages, 100 * hb[:301], color=NETC[n], lw=1.4, ls="-")
        c1.plot(ages, 100 * nb[:301], color=NETC[n], lw=1.2, ls="--")
    c1.legend([Line2D([], [], color=NETC["even"]), Line2D([], [], color=NETC["uneven"]),
               Line2D([], [], color="#555555", ls="-"), Line2D([], [], color="#555555", ls="--")],
              ["Regular", "Scale Free", "Positive belief, $\\rho_+$", "Negative belief, $\\rho_-$"], loc="upper right", ncol=2,
              columnspacing=1.0)
    c1.set_xlim(0, 300); c1.set_ylim(0, 55)
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
