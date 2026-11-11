"""Two comparison figures, Scale Free network only: baseline model against the model without negative
beliefs and with a knowledge base of 24 ("memory 24"). Seeds 1-40, periods 1501-2000.
  261111 compare_violations   violations of the population and of selected agents, and the no-communication level
  261111 compare_messages     what the delivered messages carry
Run from model/:  python "output/261111 noneg24/compare_figs.py"
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
BASE = Path("output/261109 results")
S = range(1, 41)
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "axes.grid": False, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "xtick.major.size": 3, "ytick.major.size": 3, "axes.edgecolor": "#222222", "xtick.color": "#222222",
                     "ytick.color": "#222222", "axes.labelcolor": "#000000", "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})
MODELS = [("Baseline model", BASE, "#000000"), ("No negative beliefs,\nmemory 24", HERE, "#009E73")]   # colours as in Fig. 6b
CUR, OUTD = "#4a4a4a", "#c9c9c9"


def ci(x):
    x = np.asarray(x, float); return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def topk(score, value, k, larger=True):
    s = score if larger else -score; th = np.sort(s)[::-1][k - 1]; st = s > th; tie = s == th
    return (value[st].sum() + (k - st.sum()) * value[tie].mean()) / k


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"261111 {name}.{ext}", dpi=300, bbox_inches="tight")
    print("wrote", HERE / f"261111 {name}.png")


# ------------------------------------------------------------------ violations
vals = []
for lab, d, col in MODELS:
    M = np.load(d / "main.npz"); NC = np.load(d / "nocomm.npz")
    G = lambda q: np.array([M[f"uneven|{s}|{q}"] for s in S])
    win, kin = G("win"), G("kin"); Vs, Ve = win[:, :10].mean(1), win[:, 10:].mean(1)
    v = dict(avg=G("avgV")[:, 1501:].mean(1), tick=G("minV")[:, 1501:].mean(1),
             pos5=np.array([topk(kin[i], Ve[i], 5) for i in range(len(S))]),
             rec5=np.array([topk(Vs[i], Ve[i], 5, larger=False) for i in range(len(S))]),
             none=np.array([NC[f"none|{s}|avgV"][1501:].mean() for s in S]))
    vals.append(v)
    print(lab.replace("\n", " "), {k: f"{ci(x)[0]:.2f}+-{ci(x)[1]:.2f}" for k, x in v.items()})

mk = [("avg", "Population mean", "o", None, None, 5.5),
      ("pos5", "5 agents with the largest in-degree", "D", None, None, 5.0),
      ("tick", "Top-performing agent in each period", "o", "white", "#8c8c8c", 5.0)]
fig, ax = plt.subplots(figsize=(7.08, 1.75))
rows = list(range(len(MODELS)))[::-1]
for yi, (lab, d, col), v in zip(rows, MODELS, vals):
    ax.axhline(yi, color="#dddddd", lw=0.6, zorder=0)
    ax.plot([v["pos5"].mean(), v["avg"].mean()], [yi, yi], color=col, lw=1.6, alpha=0.45, zorder=1,
            solid_capstyle="butt")
    for q, _, m, fc, ec, ms in mk:
        mu, e = ci(v[q])
        ax.errorbar([mu], [yi], xerr=[[e], [e]], fmt=m, ms=ms, mfc=(col if fc is None else fc), mec=(col if ec is None else ec),
                    mew=1.3 if m == "|" else 1.1, ecolor=(col if ec is None else ec), elinewidth=0.7, capsize=2.5, capthick=0.7, zorder=3)
ax.set_yticks(rows); ax.set_yticklabels([lab for lab, _, _ in MODELS]); ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(MODELS) - 0.4); ax.set_xlim(18, 30.5)
ax.set_xlabel("Mean number of violations, Scale Free network, periods 1501–2000")
ax.legend([Line2D([], [], marker=m, ls="none", ms=ms, mfc=("#666666" if fc is None else fc), mec=("#666666" if ec is None else ec),
                  mew=1.3 if m == "|" else 1.1) for _, _, m, fc, ec, ms in mk],
          [l for _, l, *_ in mk], loc="upper left", bbox_to_anchor=(-0.2, -0.38), ncol=3, handletextpad=0.4, columnspacing=1.6)
save(fig, "compare_violations")

# ------------------------------------------------------------------ messages
C = json.load(open(HERE / "messages_counts.json"))
shares = []
for key in ("uneven|full", "uneven|noneg24"):
    a = C[key]; tot = a["fired"] - a["empty"]
    shares.append(dict(cur=100 * (a["cur_new"] + a["cur_flip"]) / tot, out=100 * (a["out_new"] + a["out_flip"]) / tot,
                       neg=100 * (a["neg_ok"] + a["neg_bad"]) / tot, learn=(a["cur_new"] + a["cur_flip"]) / (100 * 500 * 40)))
fig, ax = plt.subplots(figsize=(7.08, 1.55))
for yi, s in zip(rows, shares):
    left = 0
    for q, fc, hatch, tc in (("cur", CUR, None, "white"), ("out", OUTD, None, "black"), ("neg", "white", "////", "black")):
        w = s[q]
        if w <= 0:
            continue
        ax.barh(yi, w, left=left, height=0.55, color=fc, hatch=hatch, edgecolor=CUR if hatch else fc, lw=0.6)
        if w >= 6:
            ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=7, color=tc,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none") if hatch else None)
        left += w
    ax.text(103, yi, f"{s['learn']:.2f}", ha="left", va="center", fontsize=7.5)
ax.text(103, len(MODELS) - 0.48, "Current constraints learned\nper agent per period", ha="left", va="bottom", fontsize=6.8, color="#333333")
ax.set_yticks(rows); ax.set_yticklabels([lab for lab, _, _ in MODELS]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(MODELS) - 0.4)
ax.set_xlabel("Share of delivered messages, Scale Free network, periods 1501–2000 (%)")
ax.legend([Patch(fc=CUR, ec=CUR), Patch(fc=OUTD, ec=OUTD), Patch(fc="white", ec=CUR, hatch="////")],
          ["A current constraint", "An outdated positive belief", "A negative belief"],
          loc="upper left", bbox_to_anchor=(-0.2, -0.5), ncol=3, handletextpad=0.5, columnspacing=1.5)
save(fig, "compare_messages")
