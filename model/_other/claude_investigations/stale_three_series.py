"""
Stale-information trajectory for THREE model variants (for the slide figure):
  (1) original model            (no testing, uniform-random communication)
  (2) learning by testing       (signed beliefs, uniform-random communication)
  (3) recency + learning by testing   (signed beliefs, most-recent communication)

Identical everything else: Scale Free (min_deg=3), N=80, K=30, alpha=2, tau=10,
obs_prob=0.01, T=2000, 5 seeds. Stale share = pooled fraction of held (positive)
beliefs whose clause is no longer in C. Run with PYTHONHASHSEED pinned.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from model_learningbytesting import LearningByTestingModel

T = 2000
CHECKPOINTS = [50, 100, 250, 500, 1000, 1500, 2000]
SEEDS = [42, 43, 44, 45, 46]
BASE = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
            setup_source="generate", type_network="Scale Free", min_deg=3)

CONDS = [
    ("original",      dict(learning_by_testing=False, recency=False)),
    ("LbT",           dict(learning_by_testing=True,  recency=False)),
    ("recency+LbT",   dict(learning_by_testing=True,  recency=True)),
]


def stale_share(m):
    C = set(m.C)
    tot = sum(len(a.kb) for a in m.agents)
    stale = sum(sum(1 for c in a.kb if c not in C) for a in m.agents)
    return 100.0 * stale / tot if tot else 0.0


results = {name: {cp: [] for cp in CHECKPOINTS} for name, _ in CONDS}
cps = set(CHECKPOINTS)
for name, opts in CONDS:
    for seed in SEEDS:
        m = LearningByTestingModel(R=T, seed=seed, **BASE, **opts)
        for t in range(1, T + 1):
            m.step()
            if t in cps:
                results[name][t].append(stale_share(m))
    print(f"  done: {name}")

print("\nStale share (%), Scale Free tau=10, mean over 5 seeds:")
print("cond          " + "".join(f"{('t'+str(cp)):>8}" for cp in CHECKPOINTS))
for name, _ in CONDS:
    row = [np.mean(results[name][cp]) for cp in CHECKPOINTS]
    print(f"{name:<14}" + "".join(f"{v:>8.1f}" for v in row))

print("\n--- for make_figs.py ---")
print("t =", CHECKPOINTS)
for name, _ in CONDS:
    row = [round(float(np.mean(results[name][cp])), 1) for cp in CHECKPOINTS]
    key = name.replace("+", "_").replace(" ", "")
    print(f"{key} = {row}")
