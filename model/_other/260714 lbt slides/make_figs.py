"""
Generate the figures for the learning-by-testing slide deck (lbt_slides.tex).

Sources:
  - stale trajectories: 5-seed means from the tau=10 baseline-vs-LbT comparison
    (produced 2026-06-30 by the comparison scripts; values hardcoded below with
    provenance) -- the diagnosis figure.
  - H1 / curve figures: computed from the per-agent CSVs written by
    _other/claude_investigations/centrality_perf_lbt_recency.py and
    centrality_perf_baseline_recency.py (tau=1 default model, 10 seeds).
Run:  py make_figs.py   (from this folder or anywhere)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "figs"
OUT.mkdir(exist_ok=True)
CSV_DIR = HERE.parent / "claude_investigations" / "output"

# Palette (colorblind-safe set, fixed order)
C_SF, C_RAND, C_SW, C_HIER = "#2a78d6", "#1baf7a", "#eda100", "#008300"
C_BASE, C_LBT, C_GOOD = "#e34948", "#2a78d6", "#1baf7a"
TOPOS = ["Scale Free", "Random", "Small World", "Hierarchical"]
COLORS = dict(zip(TOPOS, [C_SF, C_RAND, C_SW, C_HIER]))
MARKERS = dict(zip(TOPOS, ["o", "s", "^", "x"]))
LINES = dict(zip(TOPOS, ["-", "--", ":", "-."]))

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
})


# ------------------------------------------------------------------
# Fig 1: stale-information trajectory --- three variants
# (Scale Free, tau=10, 5-seed means; from stale_three_series.py)
# ------------------------------------------------------------------
t          = [50, 100, 250, 500, 1000, 1500, 2000]
s_original = [7.5, 12.0, 24.6, 41.6, 62.5, 70.3, 73.7]
s_lbt      = [4.9,  7.4, 13.0, 13.6, 19.0, 20.6, 20.3]
s_reclbt   = [5.4,  8.6, 13.6, 15.7, 17.2, 12.6, 10.2]

fig, ax = plt.subplots(figsize=(6.6, 3.7))
ax.plot(t, s_original, "-o", color=C_BASE, lw=2.2, ms=5,   label="original model")
ax.plot(t, s_lbt,      "-s", color=C_LBT,  lw=2.2, ms=5,   label="+ learning by testing")
ax.plot(t, s_reclbt,   "-^", color=C_GOOD, lw=2.4, ms=5.5, label="+ recency")
ax.set_xlabel("tick")
ax.set_ylabel("stale share of knowledge (%)")
ax.set_ylim(0, 80)
ax.set_xlim(0, 2050)
ax.legend(frameon=False, loc="upper left", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "stale_trajectory.pdf")
plt.close(fig)


# ------------------------------------------------------------------
# Helpers: pooled V-vs-in-degree curve from a per-agent CSV
# ------------------------------------------------------------------
def load_curve(fname):
    rows = list(csv.DictReader(open(CSV_DIR / fname)))
    by = defaultdict(list)
    for r in rows:
        by[(r["topo"], int(r["indeg"]))].append(float(r["meanV"]))
    curves = {}
    for tn in TOPOS:
        ks = sorted(k for (t2, k) in by if t2 == tn)
        pts = []          # (k, meanV, n), tail bins pooled so every point n>=5
        pend_k, pend_v = [], []
        for k in ks:
            v = by[(tn, k)]
            if k <= 12 and len(v) >= 5:
                pts.append((k, float(np.mean(v)), len(v)))
            else:
                pend_k += [k] * len(v)
                pend_v += v
                if len(pend_v) >= 10:
                    pts.append((float(np.mean(pend_k)), float(np.mean(pend_v)), len(pend_v)))
                    pend_k, pend_v = [], []
        if pend_v:
            pts.append((float(np.mean(pend_k)), float(np.mean(pend_v)), len(pend_v)))
        curves[tn] = pts
    return curves


def draw_curves(ax, curves, hub_band=True):
    for tn in TOPOS:
        pts = curves[tn]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, LINES[tn], marker=MARKERS[tn], ms=4.5, lw=1.9,
                color=COLORS[tn], label=tn)
    if hub_band:
        ax.axvspan(11.5, 25.5, color=C_SF, alpha=0.07)
    ax.set_xlabel("in-degree (communication intake)")
    ax.set_xlim(-0.5, 25.5)


# ------------------------------------------------------------------
# Fig 2: H1 -- average and best-agent violations by topology (tau=1 default)
# ------------------------------------------------------------------
avgV = {"Scale Free": (29.63, 0.42), "Random": (29.11, 0.67),
        "Small World": (28.94, 0.48), "Hierarchical": (29.26, 0.57)}
minV = {"Scale Free": (20.15, 0.60), "Random": (20.37, 0.77),
        "Small World": (20.30, 0.64), "Hierarchical": (20.28, 0.70)}

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
for ax, data, title, ylim in [(axes[0], avgV, "average violations", (27.5, 31)),
                              (axes[1], minV, "best-agent violations", (18.5, 22))]:
    for i, tn in enumerate(TOPOS):
        m, s = data[tn]
        ax.errorbar(i, m, yerr=s, fmt=MARKERS[tn] if MARKERS[tn] != "x" else "X",
                    color=COLORS[tn], ms=8, capsize=4, lw=1.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Scale\nFree", "Random", "Small\nWorld", "Hierar-\nchical"], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.6, 3.6)
fig.suptitle("")
fig.tight_layout()
fig.savefig(OUT / "h1_topology.pdf")
plt.close(fig)


# ------------------------------------------------------------------
# Fig 3: the universal curve (learning-by-testing, tau=1 default)
# ------------------------------------------------------------------
curves_lbt = load_curve("centrality_perf_agents.csv")

fig, ax = plt.subplots(figsize=(6.8, 3.8))
draw_curves(ax, curves_lbt)
ax.set_ylabel("long-run violations (lower = better)")
ax.set_ylim(23.5, 34.5)
ax.annotate("positions only Scale Free creates", xy=(18.3, 33.6),
            fontsize=9, color="#666", ha="center")
ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(0.87, 1.0))
fig.tight_layout()
fig.savefig(OUT / "curve_lbt.pdf")
plt.close(fig)


# ------------------------------------------------------------------
# Fig 4: with vs without testing (the cliff) -- same axes
# ------------------------------------------------------------------
curves_base = load_curve("centrality_perf_agents_baseline.csv")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
draw_curves(axes[0], curves_lbt)
axes[0].set_title("with learning by testing", fontsize=11)
axes[0].set_ylabel("long-run violations")
draw_curves(axes[1], curves_base)
axes[1].set_title("without testing (passive observation)", fontsize=11)
axes[0].set_ylim(23.5, 35.5)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "curve_compare.pdf")
plt.close(fig)

print("figures written to", OUT)
