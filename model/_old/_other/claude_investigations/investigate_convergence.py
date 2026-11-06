"""
Quick investigation: why do all networks converge to similar avg violations?

Hypothesis: agents' knowledge bases saturate toward the full clause set M,
after which greedy local optimisation hits a network-independent floor.

We compare 4 synthetic networks of different topologies on avg violations,
avg KB size, and KB "freshness" (what fraction of each agent's clauses
match the CURRENT universal clause set, as opposed to stale ones from before
clause replacement).
"""

import sys
from pathlib import Path
# Allow `from model import ...` -- model.py is at the grandparent folder
# (this script lives in _other/claude_investigations/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from model import ProblemSolvingModel

# ---------- config ----------
N       = 100
K       = 30
ALPHA   = 2
M       = ALPHA * K     # 60 clauses
OBS     = 0.01
CINT    = 10
T       = 500
SEEDS   = [42, 43, 44]

NETWORKS = [
    # NOTE: setup_random_network doesn't pre-populate nodes, so sparse random
    # graphs can have isolated nodes that crash cache_neighbors. Using denser
    # settings + other topologies to avoid the issue.
    ("Random (dense)",   dict(type_network="Random",      connect_prob=0.20)),
    ("Random (med)",     dict(type_network="Random",      connect_prob=0.10)),
    ("Small World",      dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Scale Free",       dict(type_network="Scale Free",  min_deg=3)),
]


def freshness(model):
    """Fraction of agents' KB clauses that equal a clause currently in C."""
    current = set(model.C)
    total_kb = 0
    fresh = 0
    for a in model.agents:
        total_kb += len(a.kb)
        fresh += sum(1 for c in a.kb if c in current)
    return fresh / total_kb if total_kb else 0.0


rows = []
for label, net_kwargs in NETWORKS:
    for seed in SEEDS:
        m = ProblemSolvingModel(
            N=N, K=K, alpha=ALPHA, obs_prob=OBS, clause_interval=CINT, R=T,
            setup_source="generate", seed=seed, **net_kwargs,
        )
        for _ in range(T):
            m.step()
            avg_kb = np.mean([len(a.kb) for a in m.agents])
            rows.append({
                "network":   label,
                "seed":      seed,
                "step":      m.steps,
                "avg_V":     m.avg_true_V,
                "hom":       m.homogeneity,
                "avg_kb":    avg_kb,
                "freshness": freshness(m),
            })

df = pd.DataFrame(rows)

# ---------- summary: means by network at a few reference steps ----------
def snapshot(step):
    sub = df[df.step == step].groupby("network").agg(
        avg_V=("avg_V", "mean"),
        hom=("hom", "mean"),
        avg_kb=("avg_kb", "mean"),
        freshness=("freshness", "mean"),
    ).round(3)
    return sub

for t in [20, 100, 250, 500]:
    print(f"\n=== step {t} ===")
    print(snapshot(t))

# ---------- final-step spread across networks ----------
final = df[df.step == T].groupby("network").agg(
    avg_V_mean=("avg_V", "mean"),
    avg_V_std=("avg_V", "std"),
    avg_kb_mean=("avg_kb", "mean"),
    freshness_mean=("freshness", "mean"),
    hom_mean=("hom", "mean"),
).round(3)
print("\n=== final step ({}) summary ===".format(T))
print(final)

# spread of avg_V across networks at final step (across-network std of means)
cross_net_std = final["avg_V_mean"].std()
print(f"\ncross-network std of avg_V at step {T}: {cross_net_std:.3f}")
print(f"floor (minimum across networks):          {final['avg_V_mean'].min():.3f}")
print(f"ceiling (maximum across networks):        {final['avg_V_mean'].max():.3f}")
print(f"relative gap (ceiling-floor)/floor:       "
      f"{(final['avg_V_mean'].max() - final['avg_V_mean'].min()) / final['avg_V_mean'].min():.3f}")
