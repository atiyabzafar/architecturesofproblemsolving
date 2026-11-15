"""Quick check: homogeneity across topologies under the DEFAULT model
(LbT + recency, tau=1, T=2000). Time-avg over last 500 ticks, 5 seeds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from model_learningbytesting import LearningByTestingModel

T, WINDOW = 2000, 500
SEEDS = [42, 43, 44, 45, 46]
BASE = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=1,
            setup_source="generate", learning_by_testing=True)
TOPOS = [
    ("Scale Free",   dict(type_network="Scale Free",  min_deg=4)),
    ("Random",       dict(type_network="Random",      connect_prob=0.05)),
    ("Small World",  dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Hierarchical", dict(type_network="Hierarchical", nlayers=3,
                          intra_layer_connectance=0.10, inter_layer_connectance=0.03)),
]

print(f"{'topology':<14}{'homogeneity (last 500, mean ± sd over 5 seeds)':>48}")
for tn, net in TOPOS:
    vals = []
    for seed in SEEDS:
        m = LearningByTestingModel(R=T, seed=seed, **BASE, **net)
        acc = 0.0
        for t in range(1, T + 1):
            m.step()
            if t > T - WINDOW:
                acc += m.homogeneity
        vals.append(acc / WINDOW)
    a = np.array(vals)
    print(f"{tn:<14}{a.mean():>24.4f} ± {a.std():.4f}")
