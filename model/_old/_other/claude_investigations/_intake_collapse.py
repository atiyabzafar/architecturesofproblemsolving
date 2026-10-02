"""Diagnostic: do the two performance-vs-in-degree curves collapse onto ONE
curve when the x-axis is INTAKE = in-degree / <k_in> (= in-degree * comm_scale)?
If yes, the 'milder slope' on the uniform-degree net is just the normalisation
(high <k_in> -> small comm_scale), not clause_interval or anything else.

Natural-topology data: centrality_perf_agents.csv (recency+LbT, matched <k_in>~4).
Uniform-degree data:   centrality_perf_uniform.csv (recency+LbT, <k_in>~50).
Both were clause_interval=1, tau=1, K=30, alpha=2, T=2000, last-500 window.
"""
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "output"
FIG = Path(__file__).resolve().parent.parent / "260714 lbt slides" / "figs"

# ---- natural topologies: intake = indeg / <k_in>(topo,seed) ----
nat = list(csv.DictReader(open(OUT / "centrality_perf_agents.csv")))
mean_k = defaultdict(list)
for r in nat:
    mean_k[(r["topo"], r["seed"])].append(int(r["indeg"]))
mean_k = {g: np.mean(v) for g, v in mean_k.items()}
nat_pts = [(int(r["indeg"]) / mean_k[(r["topo"], r["seed"])], float(r["meanV"]))
           for r in nat]

# ---- uniform-degree: intake = indeg / <k_in>  (<k_in> = mean over nodes) ----
uni = [r for r in csv.DictReader(open(OUT / "centrality_perf_uniform.csv"))
       if r["condition"].startswith("recency")]
uk = np.mean([int(r["indeg"]) for r in uni])
uni_pts = [(int(r["indeg"]) / uk, float(r["meanV"])) for r in uni]
print(f"<k_in>: natural per-group ~{np.mean(list(mean_k.values())):.1f}, uniform ~{uk:.1f}")

# ---- bin both by intake (log-spaced) ----
edges = np.logspace(np.log10(0.02), np.log10(6), 16)
def binned(pts):
    xs, ys = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        v = [y for x, y in pts if lo <= x < hi]
        if len(v) >= 5:
            xs.append(np.sqrt(lo * hi)); ys.append(np.mean(v))
    return xs, ys

nx_, ny_ = binned(nat_pts)
ux_, uy_ = binned(uni_pts)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
fig, ax = plt.subplots(figsize=(6.8, 4.0))
ax.plot(nx_, ny_, "-o", color="#e34948", lw=2.2, ms=6,
        label="natural topologies  ($\\langle k_{in}\\rangle\\approx4$)")
ax.plot(ux_, uy_, "-^", color="#2a78d6", lw=2.2, ms=6,
        label="uniform-degree net  ($\\langle k_{in}\\rangle\\approx50$)")
ax.axvline(1.0, color="#888", lw=1, ls=":")
ax.text(1.02, ax.get_ylim()[0] + 0.3, "average-\nconnected", fontsize=8, color="#888")
ax.set_xscale("log")
ax.set_xlabel("intake  =  in-degree / $\\langle k_{in}\\rangle$   (elicitations per tick)")
ax.set_ylabel("long-run violations")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "curve_intake_collapse.png", dpi=150)
fig.savefig(FIG / "curve_intake_collapse.pdf")
plt.close(fig)

print("\nintake  natural_V  uniform_V")
allx = sorted(set([round(x, 3) for x in nx_] + [round(x, 3) for x in ux_]))
nd = dict(zip([round(x, 3) for x in nx_], ny_))
ud = dict(zip([round(x, 3) for x in ux_], uy_))
for x in allx:
    print(f"{x:6.3f}   {nd.get(x, float('nan')):8.1f}  {ud.get(x, float('nan')):8.1f}")
print("\nfig ->", FIG / "curve_intake_collapse.png")
