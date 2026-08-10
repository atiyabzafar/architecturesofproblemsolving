"""
Performance vs. in-degree, default model (recency + learning by testing), for
several network families on ONE plot --- to show that the slope of the curve
depends on the network's average degree (via comm_scale = 1/<k_in>).

Networks (all recency + LbT, tau=1, K=30, alpha=2, T=2000, last-500 window):
  uniform 1-10   (N=100, in-degrees uniform on 1..10,  <k_in> ~ 5.5)
  uniform 1-50   (N=100, in-degrees uniform on 1..50,  <k_in> ~ 25.5)
  uniform 1-100  (N=101, in-degrees 1..100,            <k_in> ~ 50)
  scale free     (N=100, Barabasi-Albert min_deg=4,    <k_in> ~ 4, heavy tail)
  random         (N=100, Erdos-Renyi p=0.05,           <k_in> ~ 5)
Only in-degree matters for intake; the original (no-testing) model is flat for
all of these (shown previously), so we plot the default model alone here.
Pin PYTHONHASHSEED.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import csv
import random
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_learningbytesting import LearningByTestingModel

T, WINDOW = 2000, 500
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
BASE = dict(K=30, alpha=2, obs_prob=0.01, clause_interval=1,
            learning_by_testing=True, recency=True)

FIGDIR = ROOT / "_other" / "260714 lbt slides" / "figs"
OUTDIR = Path(__file__).parent / "output"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)


def build_uniform(seed, degrees):
    rng = random.Random(seed * 977 + 1)
    N = len(degrees)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for j in range(N):
        for s in rng.sample([s for s in range(N) if s != j], degrees[j]):
            G.add_edge(s, j, weight=1.0)
    return G


def uni10(seed):  return build_uniform(seed, list(range(1, 11)) * 10)    # N=100
def uni50(seed):  return build_uniform(seed, list(range(1, 51)) * 2)     # N=100
def uni100(seed): return build_uniform(seed, [max(1, i) for i in range(101)])  # N=101

# (name, kind, spec, colour, marker)
NETS = [
    ("uniform 1-10",  "graph", uni10,  "#86b6ef", "o"),
    ("uniform 1-50",  "graph", uni50,  "#2a78d6", "s"),
    ("uniform 1-100", "graph", uni100, "#0d366b", "^"),
    ("scale free",    "gen",   dict(type_network="Scale Free", min_deg=4), "#eb6834", "D"),
    ("random",        "gen",   dict(type_network="Random", connect_prob=0.05), "#1baf7a", "v"),
]

records = {n[0]: [] for n in NETS}
kbar = {}
for name, kind, spec, _c, _m in NETS:
    ks_all = []
    for seed in SEEDS:
        if kind == "graph":
            G = spec(seed)
            m = LearningByTestingModel(N=G.number_of_nodes(), R=T, seed=seed,
                                       setup_source="graph", input_graph=G, **BASE)
        else:
            m = LearningByTestingModel(N=100, R=T, seed=seed,
                                       setup_source="generate", **spec, **BASE)
        vsum = np.zeros(m.N)
        for t in range(1, T + 1):
            m.step()
            if t > T - WINDOW:
                for i in range(m.N):
                    vsum[i] += m.agent_list[i].true_violations
        vmean = vsum / WINDOW
        indeg = np.array([m.network.in_degree(i) for i in range(m.N)])
        records[name].extend(zip(indeg.tolist(), vmean.tolist()))
        ks_all.append(indeg.mean())
    kbar[name] = float(np.mean(ks_all))
    print(f"  done: {name:<14} <k_in>~{kbar[name]:.1f}")

# per-node CSV
with open(OUTDIR / "centrality_perf_multinet.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["network", "indeg", "meanV"])
    for name, *_ in NETS:
        for k, v in records[name]:
            w.writerow([name, k, round(v, 3)])


def binned(recs, nbins=13):
    ks = np.array([k for k, v in recs], float)
    vs = np.array([v for k, v in recs], float)
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
fig, ax = plt.subplots(figsize=(7.4, 4.4))
printout = {}
for name, _k, _s, color, mk in NETS:
    x, y, e = binned(records[name])
    printout[name] = [(round(a, 1), round(b, 1)) for a, b in zip(x, y)]
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, lw=0)
    ax.plot(x, y, "-" + mk, color=color, lw=2.0, ms=5,
            label=f"{name}  ($\\langle k_{{in}}\\rangle\\approx{kbar[name]:.0f}$)")
ax.set_xlabel("in-degree (communication intake)")
ax.set_ylabel("long-run violations (lower = better)")
ax.set_xlim(0, 101)
ax.legend(frameon=False, loc="upper right", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIGDIR / "curve_multinet_degree.pdf")
fig.savefig(FIGDIR / "curve_multinet_degree.png", dpi=150)
plt.close(fig)

print("\nBinned means (in-degree -> violations):")
for name, *_ in NETS:
    print(f"  {name}: {printout[name]}")
print("\nfig ->", FIGDIR / "curve_multinet_degree.pdf")
