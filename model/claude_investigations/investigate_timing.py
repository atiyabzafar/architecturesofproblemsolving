"""
Test two hypotheses for amplifying network differences in avg violations.

Hypothesis A (clause_interval sweet spot):
    At very slow drift (cint=100+), KBs saturate and networks all find similar
    greedy-SAT local optima -> convergence. At very fast drift (cint=1), no one
    tracks the environment and everyone equilibrates at a random-assignment
    baseline -> convergence. Somewhere between these extremes the 1-2 tick
    timing advantage of being central should matter most.

Hypothesis B (harder SAT landscape):
    At alpha=2, L=2 the local optima of random 2-SAT all have similar violation
    counts in the UNSAT phase. Raising alpha or moving to 3-SAT should make
    local optima more distinct and amplify network differences.

We vary (clause_interval, alpha) across a small grid and compare 3 networks
of different topologies, measuring the CROSS-NETWORK SPREAD of avg_V at a
long-run step. Larger spread = more network differentiation.
"""

import sys
from pathlib import Path
# Allow `from model import ...` when this script is run from a subfolder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from model import ProblemSolvingModel

# ---------- config ----------
N       = 80
K       = 30
OBS     = 0.01
T       = 400
SEEDS   = [42, 43, 44]

NETWORKS = [
    ("Random (dense)",  dict(type_network="Random",      connect_prob=0.20)),
    ("Small World",     dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Scale Free",      dict(type_network="Scale Free",  min_deg=3)),
]

# Grid: (clause_interval, alpha)
GRID = [
    # clause_interval sweep at alpha=2 (hypothesis A)
    (1,   2),
    (3,   2),
    (5,   2),
    (10,  2),   # paper default
    (25,  2),
    (100, 2),
    # alpha sweep at clause_interval=10 (hypothesis B)
    (10, 1),
    (10, 4),
    (10, 8),
]


def run_one(cint, alpha, net_kwargs, seed):
    m = ProblemSolvingModel(
        N=N, K=K, alpha=alpha, obs_prob=OBS, clause_interval=cint, R=T,
        setup_source="generate", seed=seed, **net_kwargs,
    )
    for _ in range(T):
        m.step()
    return m.avg_true_V, m.homogeneity, np.mean([len(a.kb) for a in m.agents])


rows = []
for cint, alpha in GRID:
    for label, net_kw in NETWORKS:
        for seed in SEEDS:
            v, h, kb = run_one(cint, alpha, net_kw, seed)
            rows.append({
                "clause_interval": cint,
                "alpha":  alpha,
                "M":      alpha * K,
                "network": label,
                "seed":   seed,
                "avg_V":  v,
                "hom":    h,
                "avg_kb": kb,
            })

df = pd.DataFrame(rows)

# mean across seeds
per_net = df.groupby(["clause_interval", "alpha", "network"]).agg(
    avg_V=("avg_V", "mean"),
    hom=("hom", "mean"),
    avg_kb=("avg_kb", "mean"),
).reset_index()

# cross-network spread for each (cint, alpha) cell
spread = per_net.groupby(["clause_interval", "alpha"]).agg(
    avg_V_min=("avg_V", "min"),
    avg_V_max=("avg_V", "max"),
    avg_V_mean=("avg_V", "mean"),
).reset_index()
spread["abs_gap"] = spread["avg_V_max"] - spread["avg_V_min"]
spread["rel_gap"] = spread["abs_gap"] / spread["avg_V_min"]
spread = spread.round(2)

print("\n========= PER-NETWORK AVG VIOLATIONS =========")
pivot = per_net.pivot_table(
    index=["clause_interval", "alpha"],
    columns="network",
    values="avg_V",
).round(2)
print(pivot)

print("\n========= CROSS-NETWORK SPREAD =========")
print("(abs_gap = max-min avg_V across networks; rel_gap = abs_gap / min)")
print(spread.to_string(index=False))
