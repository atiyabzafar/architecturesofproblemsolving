"""
Two hypotheses under the NEW DEFAULT model
(learning-by-testing + RECENCY communication, tau = clause_interval = 1, T = 2000):

H1  BETWEEN networks: long-run average violations (and best-agent violations)
    are still topology-independent when the communication rate is normalised.

H2  WITHIN networks: in more centralised topologies, the agents who perform
    best are systematically the HIGH-CENTRALITY ones -- i.e. centralisation
    doesn't buy average performance, but it lets you choose WHO performs best.

Design
------
Topologies matched on average in-degree ~4 (so H1 isn't confounded by density):
    Scale Free    min_deg=4          (heavy tail: hubs to "aim" performance at)
    Random        connect_prob=0.05  (flat)
    Small World   n_size=4, rw=0.1   (flat, clustered)
    Hierarchical  nl=3, 0.10/0.03    (the model's layered generator; NB from a
                                      previous audit it has NO hubs -- degree-
                                      homogeneous blocks -- so it is a second
                                      "flat" benchmark, not a centralised one)
10 seeds per topology. Long-run = time-average over the last W=500 ticks.

Per (topology, seed):
    avg_V, min_V             population metrics, time-averaged over the window
    Spearman rho             per-agent mean violations vs centrality
                             (in-degree, and eigenvector on the reversed graph)
    top8-bot8 gap            mean V of the 8 most central minus the 8 least
                             central agents (NEGATIVE = central agents better)
    best-agent percentile    centrality percentile of the best performer

Note: intake is proportional to own in-degree (comm_scale = 1/<k_in> fixes the
POPULATION average at ~1 elicitation/tick, not each agent's), so hubs receive
proportionally more communication. That is the mechanism H2 leans on.

Run with PYTHONHASHSEED pinned. Writes per-agent records to output/ for plots.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import csv
import numpy as np
import networkx as nx
from model_learningbytesting import LearningByTestingModel

T       = 2000
WINDOW  = 500                       # long-run window: last 500 ticks
SEEDS   = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
BASE    = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=1,
               setup_source="generate", learning_by_testing=True)

TOPOS = [
    ("Scale Free",   dict(type_network="Scale Free",  min_deg=4)),
    ("Random",       dict(type_network="Random",      connect_prob=0.05)),
    ("Small World",  dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Hierarchical", dict(type_network="Hierarchical", nlayers=3,
                          intra_layer_connectance=0.10, inter_layer_connectance=0.03)),
]


# ---------- helpers ----------
def avg_ranks(x):
    """Average ranks with proper tie handling (1-based)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = avg_ranks(x), avg_ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def eigencentrality(G):
    """Eigenvector centrality on the reversed graph (in-links confer
    centrality), normalised to [0,1]; falls back to in-degree."""
    try:
        ec = nx.eigenvector_centrality_numpy(G.reverse(copy=True))
        ec = {n: abs(float(v)) for n, v in ec.items()}
        if max(ec.values()) <= 0:
            raise ValueError
    except Exception:
        ec = {n: float(d) for n, d in G.in_degree()}
    mx = max(ec.values()) or 1.0
    return np.array([ec[i] / mx for i in range(G.number_of_nodes())])


# ---------- run ----------
rows = []          # per-(topo, seed) summary
agent_rows = []    # per-agent pooled records for plotting

for tn, net in TOPOS:
    for seed in SEEDS:
        m = LearningByTestingModel(R=T, seed=seed, **BASE, **net)
        N = m.N
        indeg = np.array([m.network.in_degree(i) for i in range(N)], float)
        eig = eigencentrality(m.network)

        vsum = np.zeros(N)
        avgV_acc = 0.0
        minV_acc = 0.0
        for t in range(1, T + 1):
            m.step()
            if t > T - WINDOW:
                for i in range(N):
                    vsum[i] += m.agent_list[i].true_violations
                avgV_acc += m.avg_true_V
                minV_acc += m.min_true_V
        vmean = vsum / WINDOW                      # per-agent long-run mean V
        avgV = avgV_acc / WINDOW
        minV = minV_acc / WINDOW

        rho_in = spearman(indeg, vmean)
        rho_eig = spearman(eig, vmean)
        top8 = np.argsort(-indeg)[:8]              # most central (by in-degree)
        bot8 = np.argsort(indeg)[:8]
        gap = float(vmean[top8].mean() - vmean[bot8].mean())
        best = int(np.argmin(vmean))
        best_pct = float((indeg < indeg[best]).mean()
                         + 0.5 * (indeg == indeg[best]).mean())
        rows.append(dict(topo=tn, seed=seed, avgV=avgV, minV=minV,
                         rho_in=rho_in, rho_eig=rho_eig, gap=gap,
                         best_pct=best_pct,
                         centr_cv=float(indeg.std() / indeg.mean())))
        for i in range(N):
            agent_rows.append((tn, seed, i, int(indeg[i]),
                               round(float(eig[i]), 4), round(float(vmean[i]), 3)))
        print(f"  done: {tn:<13} seed={seed}  avgV={avgV:6.2f}  minV={minV:5.2f}  "
              f"rho_in={rho_in:+.2f}  gap(top8-bot8)={gap:+6.2f}")

# ---------- output ----------
out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
with open(out / "centrality_perf_agents.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["topo", "seed", "agent", "indeg", "eig", "meanV"])
    w.writerows(agent_rows)

def agg(tn, key):
    a = np.array([r[key] for r in rows if r['topo'] == tn])
    return a.mean(), a.std()

print("\n" + "=" * 96)
print(f"H1 -- LONG-RUN POPULATION PERFORMANCE (time-avg over last {WINDOW} ticks; "
      f"mean ± sd over {len(SEEDS)} seeds)")
print("=" * 96)
print(f"{'topology':<14}{'avg_V':>16}{'min_V':>16}{'centrality CV':>16}")
for tn, _ in TOPOS:
    a, sa = agg(tn, 'avgV'); mn, smn = agg(tn, 'minV'); cv, _ = agg(tn, 'centr_cv')
    print(f"{tn:<14}{a:>10.2f}±{sa:4.2f}{mn:>10.2f}±{smn:4.2f}{cv:>16.2f}")

print("\n" + "=" * 96)
print("H2 -- WITHIN-NETWORK: does centrality predict individual long-run performance?")
print("(rho < 0 and gap < 0  =>  more central agents have FEWER violations)")
print("=" * 96)
print(f"{'topology':<14}{'Spearman rho(indeg)':>22}{'rho(eigen)':>14}"
      f"{'gap top8-bot8':>16}{'best-agent pct':>16}")
for tn, _ in TOPOS:
    ri, sri = agg(tn, 'rho_in'); re, sre = agg(tn, 'rho_eig')
    g, sg = agg(tn, 'gap'); bp, sbp = agg(tn, 'best_pct')
    n_neg = sum(1 for r in rows if r['topo'] == tn and r['gap'] < 0)
    print(f"{tn:<14}{ri:>15.2f}±{sri:4.2f}{re:>9.2f}±{sre:4.2f}"
          f"{g:>10.2f}±{sg:4.2f}{bp:>15.0%} ({n_neg}/{len(SEEDS)} gaps<0)")

print("\nPer-agent records written to", out / "centrality_perf_agents.csv")
