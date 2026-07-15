"""
Long-run (T=2000) topology comparison under LEARNING-BY-TESTING (no stubbornness).

Question: with the signed-belief learning-by-testing model and the standard
communication-rate normalisation (comm_scale = 1/avg-in-strength, so each agent
absorbs ~1 elicitation per tick regardless of topology), is there a long-run
performance GAP between Scale Free, Random and Small World?

Networks are matched on AVERAGE IN-DEGREE (~4) so the comparison isolates
structure (degree-distribution shape, clustering, path length) rather than
density:
    Scale Free   min_deg = 4
    Random       connect_prob = 0.05   (~4/79)
    Small World  n_size = 4, rewire_prob = 0.1
The realised average in-degree is printed so the matching can be checked.

Metric: avg violations vs the true universe C (lower = better), plus best-agent
(min) violations and the pooled stale-positive fraction. Averaged over 5 seeds
(single runs wobble because of hash-randomised set ordering).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # model/

import numpy as np
from model_learningbytesting import LearningByTestingModel

CHECKPOINTS = [50, 100, 250, 500, 1000, 1500, 2000]
SEEDS       = [42, 43, 44, 45, 46]
T           = 2000
BASE        = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
                   setup_source="generate", learning_by_testing=True)

TOPOS = [
    ("Scale Free",  dict(type_network="Scale Free",  min_deg=4)),
    ("Random",      dict(type_network="Random",      connect_prob=0.05)),
    ("Small World", dict(type_network="Small World", n_size=4, rewire_prob=0.1)),
]


def stale_fraction(m):
    C = set(m.C)
    tot_pos = tot_stale = 0
    for a in m.agents:
        tot_pos += len(a.kb)
        tot_stale += sum(1 for c in a.kb if c not in C)
    return (tot_stale / tot_pos) if tot_pos else 0.0


res = {name: {cp: {'avgV': [], 'minV': [], 'stale': []} for cp in CHECKPOINTS}
       for name, _ in TOPOS}
realised_indeg = {name: [] for name, _ in TOPOS}

cp_set = set(CHECKPOINTS)
for name, net in TOPOS:
    for seed in SEEDS:
        m = LearningByTestingModel(R=T, seed=seed, **BASE, **net)
        realised_indeg[name].append(
            np.mean([d for _, d in m.network.in_degree()]))
        for t in range(1, T + 1):
            m.step()
            if t in cp_set:
                res[name][t]['avgV'].append(float(m.avg_true_V))
                res[name][t]['minV'].append(float(m.min_true_V))
                res[name][t]['stale'].append(stale_fraction(m))
        print(f"  done: {name:<12} seed={seed}")

# ---- realised density check ----
print("\nRealised average in-degree (matched target ~4):")
for name, _ in TOPOS:
    a = np.array(realised_indeg[name])
    print(f"  {name:<12} {a.mean():.2f} ± {a.std():.2f}")


def row(label, vals_by_cp, nd=2):
    cells = []
    for cp in CHECKPOINTS:
        arr = np.array(vals_by_cp[cp])
        cells.append(f"{arr.mean():6.2f}±{arr.std():4.2f}")
    return f"{label:<14}" + "".join(f"{c:>14}" for c in cells)

hdr = f"{'':<14}" + "".join(f"{('t='+str(cp)):>14}" for cp in CHECKPOINTS)

print("\n" + "=" * 112)
print(f"AVG VIOLATIONS vs the true universe C  (mean ± sd over {len(SEEDS)} seeds; "
      f"lower = better)")
print("=" * 112)
print(hdr)
for name, _ in TOPOS:
    print(row(name, {cp: res[name][cp]['avgV'] for cp in CHECKPOINTS}))

print("\n" + "=" * 112)
print("BEST-AGENT (min) VIOLATIONS vs C")
print("=" * 112)
print(hdr)
for name, _ in TOPOS:
    print(row(name, {cp: res[name][cp]['minV'] for cp in CHECKPOINTS}))

print("\n" + "=" * 112)
print("POOLED STALE-POSITIVE FRACTION (%)")
print("=" * 112)
print(hdr)
for name, _ in TOPOS:
    print(row(name, {cp: [v*100 for v in res[name][cp]['stale']] for cp in CHECKPOINTS}))

# ---- machine-readable means for plotting ----
print("\n--- MEANS for plotting ---")
for metric in ('avgV', 'minV'):
    print(f"{metric}:")
    for name, _ in TOPOS:
        means = [round(float(np.mean(res[name][cp][metric])), 2) for cp in CHECKPOINTS]
        sds   = [round(float(np.std(res[name][cp][metric])), 2) for cp in CHECKPOINTS]
        print(f"  {name}: means={dict(zip(CHECKPOINTS, means))} sds={dict(zip(CHECKPOINTS, sds))}")
