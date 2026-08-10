"""Re-plot the multi-network performance-vs-in-degree figure from the saved CSV,
adding (i) an X at each series' average (mean in-degree, mean violations) and
(ii) a least-squares line fitted through those five average points."""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent / "260714 lbt slides" / "figs"
rows = list(csv.DictReader(open(HERE / "output" / "centrality_perf_multinet.csv")))

recs = defaultdict(list)
for r in rows:
    recs[r["network"]].append((int(r["indeg"]), float(r["meanV"])))

NETS = [
    ("uniform 1-10",  "#86b6ef", "o"),
    ("uniform 1-50",  "#2a78d6", "s"),
    ("uniform 1-100", "#0d366b", "^"),
    ("scale free",    "#eb6834", "D"),
    ("random",        "#1baf7a", "v"),
]

def binned(pts, nbins=13):
    ks = np.array([k for k, v in pts], float)
    vs = np.array([v for k, v in pts], float)
    edges = np.linspace(ks.min(), ks.max() + 1e-6, nbins + 1)
    xs, ys, es = [], [], []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        sel = (ks >= lo) & (ks <= hi if i == nbins - 1 else ks < hi)
        if sel.sum() >= 5:
            xs.append(ks[sel].mean()); ys.append(vs[sel].mean())
            es.append(vs[sel].std() / np.sqrt(sel.sum()))
    return np.array(xs), np.array(ys), np.array(es)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
fig, ax = plt.subplots(figsize=(7.6, 4.6))

cx, cy = [], []          # series-average points
for name, color, mk in NETS:
    pts = recs[name]
    x, y, e = binned(pts)
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.14, lw=0)
    kbar = np.mean([k for k, v in pts])
    ax.plot(x, y, "-" + mk, color=color, lw=1.9, ms=4.5,
            label=f"{name}  ($\\langle k_{{in}}\\rangle\\approx{kbar:.0f}$)")
    cx.append(kbar)
    cy.append(np.mean([v for k, v in pts]))   # population-mean violations

cx, cy = np.array(cx), np.array(cy)
# X markers at each series average
ax.scatter(cx, cy, marker="x", s=90, linewidths=2.4, color="black", zorder=6,
           label="series average")
# least-squares line through the five averages
slope, intercept = np.polyfit(cx, cy, 1)
xline = np.array([0, 100])
ax.plot(xline, slope * xline + intercept, "--", color="black", lw=1.6, alpha=0.8,
        label=f"fit through averages (slope ${slope:+.3f}$/in-degree)", zorder=5)

ax.set_xlabel("in-degree (communication intake)")
ax.set_ylabel("long-run violations (lower = better)")
ax.set_xlim(0, 101)
ax.legend(frameon=False, loc="upper right", fontsize=8.8)
fig.tight_layout()
fig.savefig(FIGDIR / "curve_multinet_degree.pdf")
fig.savefig(FIGDIR / "curve_multinet_degree.png", dpi=150)
plt.close(fig)

print("series averages (mean in-degree, mean violations):")
for (name, *_), a, b in zip(NETS, cx, cy):
    print(f"  {name:<14} <k_in>={a:5.1f}   mean V={b:5.2f}")
r = np.corrcoef(cx, cy)[0, 1]
print(f"\nfit through the 5 averages: V = {slope:+.4f}*<k_in> + {intercept:.2f}   (r={r:+.2f})")
print("fig ->", FIGDIR / "curve_multinet_degree.pdf")
