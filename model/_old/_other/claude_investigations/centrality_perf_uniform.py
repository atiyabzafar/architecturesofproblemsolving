"""
Performance vs. in-degree on a UNIFORM-DEGREE network (in-degrees 1..100), so we
can see the whole curve --- natural topologies cap in-degree around 10-30.

Construction: N=101 nodes; node i is given in-degree max(1, i), so realised
in-degrees sweep 1..100 (roughly one node per value). For each node we draw that
many random distinct in-neighbours. Communication rate is normalised as usual
(comm_scale = 1/<k_in>, here ~1/50), so per-agent intake is proportional to own
in-degree, sweeping ~0.02 to ~2 elicitations/tick.

Two conditions on the SAME network per seed (paired):
  (1) recency + learning by testing   (the default model)
  (2) original model                  (no testing, uniform-random communication)

Default parameters: K=30, alpha=2, tau=1 (fast drift), T=2000, long run = last 500
ticks, 10 seeds. Saves a per-node CSV and the figure (PDF+PNG). Pin PYTHONHASHSEED.
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

N = 101
T, WINDOW = 2000, 500
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
BASE = dict(K=30, alpha=2, obs_prob=0.01, clause_interval=1)

CONDS = [
    ("recency + learning by testing", dict(learning_by_testing=True,  recency=True)),
    ("original model",                dict(learning_by_testing=False, recency=False)),
]

FIGDIR = ROOT / "_other" / "260714 lbt slides" / "figs"
OUTDIR = Path(__file__).parent / "output"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)


def build_uniform_indegree_graph(seed):
    rng = random.Random(seed * 977 + 1)
    degrees = [max(1, i) for i in range(N)]      # node i -> in-degree max(1,i)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for j in range(N):
        for s in rng.sample([s for s in range(N) if s != j], degrees[j]):
            G.add_edge(s, j, weight=1.0)
    return G


# records[cond] = list of (indeg, meanV)
records = {name: [] for name, _ in CONDS}
for seed in SEEDS:
    G = build_uniform_indegree_graph(seed)
    for name, opts in CONDS:
        m = LearningByTestingModel(N=N, R=T, seed=seed, setup_source="graph",
                                   input_graph=G, **BASE, **opts)
        vsum = np.zeros(m.N)
        for t in range(1, T + 1):
            m.step()
            if t > T - WINDOW:
                for i in range(m.N):
                    vsum[i] += m.agent_list[i].true_violations
        vmean = vsum / WINDOW
        indeg = np.array([m.network.in_degree(i) for i in range(m.N)])
        records[name].extend(zip(indeg.tolist(), vmean.tolist()))
    print(f"  done seed {seed}")

# write per-node CSV
with open(OUTDIR / "centrality_perf_uniform.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["condition", "indeg", "meanV"])
    for name, _ in CONDS:
        for k, v in records[name]:
            w.writerow([name, k, round(v, 3)])

# bin by in-degree (width 5): 1-5, 6-10, ... 96-100
EDGES = list(range(1, 102, 5))            # 1,6,11,...,96,101
def binned(recs):
    xs, ys, es = [], [], []
    for lo in EDGES[:-1]:
        hi = lo + 4
        vals = [v for k, v in recs if lo <= k <= hi]
        if vals:
            xs.append(lo + 2)             # bin centre
            ys.append(float(np.mean(vals)))
            es.append(float(np.std(vals) / np.sqrt(len(vals))))
    return np.array(xs), np.array(ys), np.array(es)

# ---- plot ----
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150})
C_BRIGHT, C_GRAY = "#2a78d6", "#b0b0b0"
STYLE = {"recency + learning by testing": (C_BRIGHT, "o", 2.6, 1.0),
         "original model": (C_GRAY, "s", 2.2, 1.0)}

fig, ax = plt.subplots(figsize=(6.8, 4.0))
printout = {}
for name, _ in CONDS:
    x, y, e = binned(records[name])
    printout[name] = list(zip(x.tolist(), [round(v, 1) for v in y.tolist()]))
    color, mk, lw, a = STYLE[name]
    ax.fill_between(x, y - e, y + e, color=color, alpha=0.18, lw=0)
    ax.plot(x, y, "-" + mk, color=color, lw=lw, ms=5, alpha=a, label=name)
ax.set_xlabel("in-degree (communication intake)")
ax.set_ylabel("long-run violations (lower = better)")
ax.set_xlim(0, 101)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIGDIR / "curve_uniform_degree.pdf")
fig.savefig(FIGDIR / "curve_uniform_degree.png", dpi=150)
plt.close(fig)

print("\nBinned means (in-degree bin centre -> long-run violations):")
for name, _ in CONDS:
    print(f"  {name}: {printout[name]}")
print("\nfigure ->", FIGDIR / "curve_uniform_degree.pdf")
