"""
Same two-hypothesis experiment as centrality_perf_lbt_recency.py, but WITHOUT
learning-by-testing: passive observation (obs_prob=0.01, now active), unsigned
KB capped at M, strict FIFO -- while KEEPING recency-ordered communication.
Everything else identical: tau=1, T=2000, matched <k_in>~4, 10 seeds, long-run
= last-500-tick averages. Adds a pooled STALE SHARE column (fraction of KB
clauses no longer in C at the final tick) for context: with tau=1 and no
flushing, baseline KBs should be heavily stale.
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
WINDOW  = 500
SEEDS   = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
BASE    = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=1,
               setup_source="generate", learning_by_testing=False)   # <-- baseline

TOPOS = [
    ("Scale Free",   dict(type_network="Scale Free",  min_deg=4)),
    ("Random",       dict(type_network="Random",      connect_prob=0.05)),
    ("Small World",  dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Hierarchical", dict(type_network="Hierarchical", nlayers=3,
                          intra_layer_connectance=0.10, inter_layer_connectance=0.03)),
]


def avg_ranks(x):
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


rows = []
agent_rows = []

for tn, net in TOPOS:
    for seed in SEEDS:
        m = LearningByTestingModel(R=T, seed=seed, **BASE, **net)
        N = m.N
        indeg = np.array([m.network.in_degree(i) for i in range(N)], float)

        vsum = np.zeros(N)
        avgV_acc = minV_acc = 0.0
        for t in range(1, T + 1):
            m.step()
            if t > T - WINDOW:
                for i in range(N):
                    vsum[i] += m.agent_list[i].true_violations
                avgV_acc += m.avg_true_V
                minV_acc += m.min_true_V
        vmean = vsum / WINDOW
        avgV = avgV_acc / WINDOW
        minV = minV_acc / WINDOW

        C_set = set(m.C)
        tot_kb = sum(len(a.kb) for a in m.agents)
        tot_stale = sum(sum(1 for c in a.kb if c not in C_set) for a in m.agents)
        stale = tot_stale / tot_kb if tot_kb else 0.0

        rho_in = spearman(indeg, vmean)
        top8 = np.argsort(-indeg)[:8]
        bot8 = np.argsort(indeg)[:8]
        gap = float(vmean[top8].mean() - vmean[bot8].mean())
        best = int(np.argmin(vmean))
        best_pct = float((indeg < indeg[best]).mean()
                         + 0.5 * (indeg == indeg[best]).mean())
        rows.append(dict(topo=tn, seed=seed, avgV=avgV, minV=minV,
                         rho_in=rho_in, gap=gap, best_pct=best_pct, stale=stale))
        for i in range(N):
            agent_rows.append((tn, seed, i, int(indeg[i]), round(float(vmean[i]), 3)))
        print(f"  done: {tn:<13} seed={seed}  avgV={avgV:6.2f}  minV={minV:5.2f}  "
              f"rho_in={rho_in:+.2f}  gap={gap:+6.2f}  stale={stale:5.1%}")

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
with open(out / "centrality_perf_agents_baseline.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["topo", "seed", "agent", "indeg", "meanV"])
    w.writerows(agent_rows)

def agg(tn, key):
    a = np.array([r[key] for r in rows if r['topo'] == tn])
    return a.mean(), a.std()

print("\n" + "=" * 100)
print(f"H1 (BASELINE + recency) -- long-run population performance "
      f"(last {WINDOW} ticks; mean ± sd over {len(SEEDS)} seeds)")
print("=" * 100)
print(f"{'topology':<14}{'avg_V':>16}{'min_V':>16}{'stale share':>14}")
for tn, _ in TOPOS:
    a, sa = agg(tn, 'avgV'); mn, smn = agg(tn, 'minV'); st, _ = agg(tn, 'stale')
    print(f"{tn:<14}{a:>10.2f}±{sa:4.2f}{mn:>10.2f}±{smn:4.2f}{st:>13.1%}")

print("\n" + "=" * 100)
print("H2 (BASELINE + recency) -- centrality vs individual long-run performance")
print("=" * 100)
print(f"{'topology':<14}{'Spearman rho(indeg)':>22}{'gap top8-bot8':>16}{'best-agent pct':>16}")
for tn, _ in TOPOS:
    ri, sri = agg(tn, 'rho_in'); g, sg = agg(tn, 'gap'); bp, _ = agg(tn, 'best_pct')
    n_neg = sum(1 for r in rows if r['topo'] == tn and r['gap'] < 0)
    print(f"{tn:<14}{ri:>15.2f}±{sri:4.2f}{g:>10.2f}±{sg:4.2f}"
          f"{bp:>15.0%} ({n_neg}/{len(SEEDS)} gaps<0)")

print("\nPer-agent records written to", out / "centrality_perf_agents_baseline.csv")
