"""
Test whether adding L=3 XOR (parity) clauses amplifies cross-network
differences in avg violations.

Design:
    * Subclass ProblemSolvingModel and override only random_clause() so a
      fraction `frac_l3` of generated clauses are 3-XOR (parity-3) instead
      of the normal L=2 AND/XOR mix.
    * Nothing in model.py is touched. clause_violated, local_update_around,
      canonicalise_clause and replace_universal_clause all happen to be
      L-agnostic, so the override is enough.

Hypothesis:
    L=2 AND/XOR landscapes are smooth: greedy converges to the same floor
    on every network, killing differentiation. 3-XOR creates plateaus
    (any single-variable flip changes the clause's status by the same
    amount), so the order in which fresh clauses arrive starts to matter.
    Networks with faster info diffusion should pull away.

We sweep frac_l3 ∈ {0.0, 0.3, 0.6} on 3 topologies, 3 seeds, two cint
values, and report per-network avg_V plus cross-network spread.
"""

import sys
from pathlib import Path
# Allow `from model import ...` -- model.py is at the grandparent folder
# (this script lives in _other/claude_investigations/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import random
import numpy as np
import pandas as pd
from model import ProblemSolvingModel

# ---------- config ----------
N       = 80
K       = 30
ALPHA   = 2
OBS     = 0.01
T       = 400
SEEDS   = [42, 43, 44]
CINTS   = [10, 25]
FRAC_L3 = [0.0, 0.3, 0.6]

NETWORKS = [
    ("Random (dense)",  dict(type_network="Random",      connect_prob=0.20)),
    ("Small World",     dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
    ("Scale Free",      dict(type_network="Scale Free",  min_deg=3)),
]


# ============================================================
# Subclass: inject L=3 XOR clauses with prob frac_l3
# ============================================================
class L3Model(ProblemSolvingModel):
    def __init__(self, *args, frac_l3=0.0, **kwargs):
        # store BEFORE super().__init__ because generate_clauses fires there
        self.frac_l3 = frac_l3
        super().__init__(*args, **kwargs)

    def random_clause(self):
        # With prob frac_l3 emit a 3-XOR clause; otherwise behave normally.
        if random.random() < self.frac_l3:
            indices = random.sample(range(1, self.K + 1), 3)
            return self.canonicalise_clause(("XOR", tuple(indices)))
        return super().random_clause()


# ============================================================
# Run one configuration
# ============================================================
def run_one(frac_l3, cint, net_kwargs, seed):
    m = L3Model(
        N=N, K=K, alpha=ALPHA, obs_prob=OBS, clause_interval=cint, R=T,
        setup_source="generate", seed=seed,
        frac_l3=frac_l3,
        **net_kwargs,
    )
    for _ in range(T):
        m.step()
    avg_kb = float(np.mean([len(a.kb) for a in m.agents]))
    return m.avg_true_V, m.homogeneity, avg_kb


# ============================================================
# Main sweep
# ============================================================
rows = []
total = len(FRAC_L3) * len(CINTS) * len(NETWORKS) * len(SEEDS)
done = 0
for frac in FRAC_L3:
    for cint in CINTS:
        for label, net_kw in NETWORKS:
            for seed in SEEDS:
                v, h, kb = run_one(frac, cint, net_kw, seed)
                rows.append({
                    "frac_l3":         frac,
                    "clause_interval": cint,
                    "network":         label,
                    "seed":            seed,
                    "avg_V":           v,
                    "hom":             h,
                    "avg_kb":          kb,
                })
                done += 1
                print(f"[{done}/{total}] frac_l3={frac} cint={cint:>2} "
                      f"{label:<16} seed={seed}  avg_V={v:.2f}")

df = pd.DataFrame(rows)

# ---------- mean across seeds ----------
per_net = (df.groupby(["frac_l3", "clause_interval", "network"])
             .agg(avg_V=("avg_V", "mean"),
                  hom=("hom", "mean"),
                  avg_kb=("avg_kb", "mean"))
             .reset_index())

# ---------- cross-network spread ----------
spread = (per_net.groupby(["frac_l3", "clause_interval"])
                 .agg(avg_V_min=("avg_V", "min"),
                      avg_V_max=("avg_V", "max"),
                      avg_V_mean=("avg_V", "mean"))
                 .reset_index())
spread["abs_gap"] = spread["avg_V_max"] - spread["avg_V_min"]
spread["rel_gap"] = spread["abs_gap"] / spread["avg_V_min"]
spread = spread.round(2)

print("\n========= PER-NETWORK AVG VIOLATIONS =========")
pivot = per_net.pivot_table(
    index=["frac_l3", "clause_interval"],
    columns="network",
    values="avg_V",
).round(2)
print(pivot)

print("\n========= CROSS-NETWORK SPREAD =========")
print("(abs_gap = max-min avg_V across networks; rel_gap = abs_gap / min)")
print(spread.to_string(index=False))
