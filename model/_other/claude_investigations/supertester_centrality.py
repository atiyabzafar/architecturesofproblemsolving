"""
Claim under test:
  Give m agents the ability to test at 5x the normal rate (5 test-actions/tick
  instead of 1). If those m "super-testers" are the HIGHEST-CENTRALITY agents,
  then Scale Free / Hierarchical networks outperform Random networks.

Design (learning-by-testing model, comm normalisation ON, matched avg in-degree ~4):
  topologies : Scale Free (min_deg=4), Hierarchical (3 layers), Random (p=0.05)
  targeting  : none (no boost) | central (top-m by in-degree centrality) | random (m random)
  m = 8 (10% of N=80), boost = 5x, T = 2000, 5 seeds.

For a given seed the network, universe and initial state are identical across the
three targeting conditions (random-target uses a SEPARATE RNG so it doesn't perturb
the simulation), so conditions differ only in WHICH agents are boosted.

Metrics at t=2000 (lower = better):
  full      = avg violations vs C over ALL agents
  nonboost  = avg violations vs C over the NON-super-testers (isolates spreading)
The claim predicts: under 'central', Scale Free (and Hierarchical) beat Random,
i.e. a positive Random-minus-topology gap that is larger than under 'random'/'none'.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # model/

import random
import numpy as np
from model_learningbytesting import LearningByTestingModel

CHECKPOINTS = [500, 1000, 2000]
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
T, M, BOOST = 2000, 8, 5
BASE = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
            setup_source="generate", learning_by_testing=True)

TOPOS = [
    ("Scale Free",   dict(type_network="Scale Free",  min_deg=4)),
    ("Hierarchical", dict(type_network="Hierarchical", nlayers=3,
                          intra_layer_connectance=0.10, inter_layer_connectance=0.03)),
    ("Random",       dict(type_network="Random",      connect_prob=0.05)),
]
CONDS = ["none", "central", "random"]


def select(model, target, seed):
    if target == "none":
        return []
    if target == "central":   # top-m by in-degree centrality (the model's centr)
        return sorted(range(model.N), key=lambda i: model.agent_list[i].centr,
                      reverse=True)[:M]
    # random target: separate RNG so the simulation's randomness is unperturbed
    return random.Random(seed * 131 + 7).sample(range(model.N), M)


def run(net, seed, target):
    m = LearningByTestingModel(R=T, seed=seed, **BASE, **net)
    sel = select(m, target, seed)
    sset = set(sel)
    for i in range(m.N):
        m.agent_list[i].test_rate = BOOST if i in sset else 1
    rec = {}
    cps = set(CHECKPOINTS)
    for t in range(1, T + 1):
        m.step()
        if t in cps:
            full = float(m.avg_true_V)
            nb = (float(np.mean([m.agent_list[i].true_violations
                                 for i in range(m.N) if i not in sset]))
                  if sset else full)
            rec[t] = (full, nb)
    top_out = float(np.mean([m.network.out_degree(i) for i in sel])) if sel else float('nan')
    return rec, top_out


# results[(topo, cond)] = {cp: {'full': [...], 'nb': [...]}}, topout[(topo,cond)] = [...]
results = {(tn, c): {cp: {'full': [], 'nb': []} for cp in CHECKPOINTS}
           for tn, _ in TOPOS for c in CONDS}
topout = {(tn, c): [] for tn, _ in TOPOS for c in CONDS}

for tn, net in TOPOS:
    for cond in CONDS:
        for seed in SEEDS:
            rec, to = run(net, seed, cond)
            for cp in CHECKPOINTS:
                results[(tn, cond)][cp]['full'].append(rec[cp][0])
                results[(tn, cond)][cp]['nb'].append(rec[cp][1])
            topout[(tn, cond)].append(to)
        print(f"  done: {tn:<13} {cond}")


def arr(tn, cond, cp, key):
    return np.array(results[(tn, cond)][cp][key])

def mean(tn, cond, cp, key):
    return float(arr(tn, cond, cp, key).mean())

def sd(tn, cond, cp, key):
    return float(arr(tn, cond, cp, key).std())

N_SEEDS = len(SEEDS)

print("\n" + "=" * 78)
print(f"AVG VIOLATIONS vs C at t={T}  (full population; mean ± sd over {N_SEEDS} seeds)")
print("=" * 78)
print(f"{'targeting':<12}" + "".join(f"{tn:>16}" for tn, _ in TOPOS))
for cond in CONDS:
    cells = [f"{mean(tn,cond,T,'full'):6.2f}±{sd(tn,cond,T,'full'):4.2f}" for tn, _ in TOPOS]
    print(f"{cond:<12}" + "".join(f"{c:>16}" for c in cells))

print("\n" + "=" * 78)
print("PAIRED within-topology effects at t=2000 (same network per seed -> cancels")
print("network/hash variance). Positive = lower violations = better.")
print("=" * 78)
print(f"{'':<26}" + "".join(f"{tn:>16}" for tn, _ in TOPOS))
# boost benefit: none - central, and none - random (paired per seed)
for label, a, b in [("central boost helps (none-central)", 'none', 'central'),
                    ("random boost helps (none-random)",  'none', 'random')]:
    cells = []
    for tn, _ in TOPOS:
        d = arr(tn, a, T, 'full') - arr(tn, b, T, 'full')
        cells.append(f"{d.mean():6.2f}±{d.std():4.2f}")
    print(f"{label:<26}" + "".join(f"{c:>16}" for c in cells))
print("-" * 78)
# THE KEY TEST: does central targeting beat random targeting? (random - central, paired)
print("central beats random targeting (random-central): the claim's mechanism")
cells = []
for tn, _ in TOPOS:
    d = arr(tn, 'random', T, 'full') - arr(tn, 'central', T, 'full')
    n_pos = int((d > 0).sum())
    cells.append(f"{d.mean():+5.2f}±{d.std():4.2f}({n_pos}/{N_SEEDS})")
    results[(tn, 'adv')] = d
print(f"{'':<26}" + "".join(f"{c:>16}" for c in cells))
print("   (value = mean paired advantage of central over random targeting;")
print("    (k/n) = in how many seeds central beat random. >0 supports the claim.)")

print("\n" + "=" * 78)
print("NON-SUPER-TESTER avg violations at t=2000 (isolates spreading to others)")
print("=" * 78)
print(f"{'targeting':<12}" + "".join(f"{tn:>16}" for tn, _ in TOPOS))
for cond in CONDS:
    cells = [f"{mean(tn,cond,T,'nb'):6.2f}±{sd(tn,cond,T,'nb'):4.2f}" for tn, _ in TOPOS]
    print(f"{cond:<12}" + "".join(f"{c:>16}" for c in cells))

print("\n" + "=" * 78)
print("Super-testers' average OUT-degree (how widely the boost can spread)")
print("=" * 78)
for tn, _ in TOPOS:
    print(f"  {tn:<13} central={np.nanmean(topout[(tn,'central')]):4.1f}   "
          f"random={np.nanmean(topout[(tn,'random')]):4.1f}")
