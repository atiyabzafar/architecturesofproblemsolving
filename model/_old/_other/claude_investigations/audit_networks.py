"""
Audit: do the networks the ANIMATION generates actually match the selected
category? Drives serve.run_simulation() -- the exact code path behind the
Generate button -- for all five topology options, rebuilds the graph from the
payload's edge list, and measures category signatures:

  density / mean in-degree     (did the requested parameters take effect?)
  degree CV, max/mean          (heavy tail -> Scale Free signature)
  clustering (undirected)      (high -> Small World signature)
  avg shortest path on LCC     (short despite clustering -> Small World)
  greedy modularity Q, #comms  (high Q, ~4 blocks -> Modular/Hierarchical)
  intra-block edge fraction    (Modular only: blocks are known index ranges)
  seed-to-seed edge Jaccard    (are instances actually random within category?)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent      # model/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'animation'))

import networkx as nx
import numpy as np
import serve   # the animation module (monkey-patches model.Person; fine for audit)

BASE = dict(N=80, K=30, alpha=2.0, obs_prob=0.01, clause_interval=10, T=2,
            redundant_comm=False, normalize_comm=True, learning_by_testing=False)

CONFIGS = [
    ('Random',       dict(network='Random', connect_prob=0.10)),
    ('Small World',  dict(network='Small World', n_size=4, rewire_prob=0.1)),
    ('Scale Free',   dict(network='Scale Free', min_deg=3)),
    ('Hierarchical', dict(network='Hierarchical', nlayers=4,
                          intra_layer_conn=0.20, inter_layer_conn=0.05)),
    ('Modular',      dict(network='Modular', p_in=0.20, p_out=0.02)),
]


def graph_from_payload(payload):
    G = nx.DiGraph()
    G.add_nodes_from(range(payload['N']))
    G.add_edges_from((u, v) for u, v in payload['edges'])
    return G


def stats(G, label):
    N = G.number_of_nodes()
    U = G.to_undirected()
    indeg = np.array([d for _, d in G.in_degree()])
    tot = np.array([d for _, d in U.degree()])
    lcc = max(nx.connected_components(U), key=len)
    Usub = U.subgraph(lcc)
    apl = nx.average_shortest_path_length(Usub) if len(lcc) > 1 else float('nan')
    comms = list(nx.algorithms.community.greedy_modularity_communities(U))
    Q = nx.algorithms.community.modularity(U, comms)
    out = {
        'edges': G.number_of_edges(),
        'mean_in': indeg.mean(),
        'deg_CV': tot.std() / tot.mean() if tot.mean() else 0,
        'max/mean': tot.max() / tot.mean() if tot.mean() else 0,
        'clust': nx.average_clustering(U),
        'APL': apl,
        'Q': Q,
        'ncomm': len(comms),
        'lcc': len(lcc) / N,
    }
    if label == 'Modular':   # known sequential blocks of N/4
        block = lambda n: n // 20
        intra = sum(1 for u, v in G.edges() if block(u) == block(v))
        out['intra_frac'] = intra / G.number_of_edges()
    return out


print(f"{'network':<14}{'edges':>6}{'<k_in>':>8}{'degCV':>7}{'mx/mn':>7}"
      f"{'clust':>7}{'APL':>6}{'Q':>6}{'#com':>5}{'LCC':>5}  extra")
print('-' * 88)

jaccards = {}
for label, cfg in CONFIGS:
    edge_sets = []
    for seed in (101, 202):
        p = dict(BASE); p.update(cfg); p['seed'] = seed
        payload = serve.run_simulation(p)
        G = graph_from_payload(payload)
        edge_sets.append(set(G.edges()))
        if seed == 101:
            s = stats(G, label)
            extra = f"intra={s['intra_frac']:.0%}" if 'intra_frac' in s else ''
            print(f"{label:<14}{s['edges']:>6}{s['mean_in']:>8.2f}{s['deg_CV']:>7.2f}"
                  f"{s['max/mean']:>7.2f}{s['clust']:>7.3f}{s['APL']:>6.2f}"
                  f"{s['Q']:>6.2f}{s['ncomm']:>5}{s['lcc']:>5.0%}  {extra}")
    inter = len(edge_sets[0] & edge_sets[1])
    union = len(edge_sets[0] | edge_sets[1])
    jaccards[label] = inter / union if union else 1.0

print('\nSeed-to-seed edge overlap (Jaccard; ~0 = fresh random instance per seed):')
for label, j in jaccards.items():
    print(f"  {label:<14}{j:.3f}")

print("""
Reference signatures:
  Random        low degCV (~0.2-0.3), clustering ~ density (~0.19), low Q
  Small World   HIGH clustering (>0.4), APL a bit above ER, moderate-high Q
  Scale Free    HIGH degCV (>0.7) and max/mean (>3): hubs; low clustering
  Hierarchical  block-ish Q with ~nlayers communities, low degCV
  Modular       HIGH Q (>0.4), ~4 communities, intra fraction >> p_out share
""")
