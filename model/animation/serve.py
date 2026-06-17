"""
Interactive viewer for the problem-solving model.

Run:
    py -u animation/serve.py

A browser window opens automatically on http://localhost:8765.

You pick parameters in the form, click "Generate", and a Python simulation
runs in the background using the real model.py. The animation appears in
the browser when ready. Click any agent on the network to see its detailed
stats and a sparkline of its violation count over time.

Press Ctrl+C in the terminal to stop the server.
"""

import http.server
import json
import random
import socket
import socketserver
import sys
import threading
import time
import traceback
import webbrowser
from array import array
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import networkx as nx

# Find model.py one folder up
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as _mm
from model import Person, ProblemSolvingModel


# ============================================================
# Tracking subclass (logs comm/obs events per tick)
# ============================================================
# Mirrors Person.step() exactly; just adds two log lines on successful
# transfers. Monkey-patched into model.py so the rest of the pipeline
# (network construction, scheduling) uses TrackingPerson agents.

class TrackingPerson(Person):
    def step(self):
        my_kb_set = set(self.kb)

        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)
                self.model._obs_events_this_tick.append(self.unique_id)

        if self.in_neighbors:
            comm_scale = self.model.comm_scale
            for nbr_id in self.in_neighbors:
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get('weight', 1.0)
                link_probability = min(1.0, comm_scale * link_weight)
                if random.random() < link_probability:
                    nbr_agent = self.model.agent_list[nbr_id]
                    if nbr_agent.kb:
                        if self.model.redundant_comm:
                            # Redundant-communication variant: the neighbour
                            # sends a clause drawn from its WHOLE knowledge base,
                            # not knowing what the recipient already has. If the
                            # recipient already knows it, nothing is transferred
                            # (the elicitation is "wasted").
                            cprime = random.choice(nbr_agent.kb)
                            if cprime not in my_kb_set:
                                self.add_clause_to_kb(cprime)
                                self.local_update_around(cprime)
                                my_kb_set.add(cprime)
                                self.model._comm_events_this_tick.append(
                                    (nbr_id, self.unique_id))
                        else:
                            # Default: the neighbour shares only something the
                            # recipient does not yet know (no wasted transfers).
                            nbr_kb_set = set(nbr_agent.kb)
                            unknowns = nbr_kb_set - my_kb_set
                            if unknowns:
                                cprime = random.choice(list(unknowns))
                                self.add_clause_to_kb(cprime)
                                self.local_update_around(cprime)
                                my_kb_set.add(cprime)
                                self.model._comm_events_this_tick.append(
                                    (nbr_id, self.unique_id))


_mm.Person = TrackingPerson


class TrackingModel(ProblemSolvingModel):
    def __init__(self, *args, redundant_comm=False, **kwargs):
        self._comm_events_this_tick = []
        self._obs_events_this_tick = []
        self._replace_events = []        # (tick, new_clause) for each drift replacement
        self.redundant_comm = redundant_comm   # variant flag read by TrackingPerson
        super().__init__(*args, **kwargs)

    def step(self):
        self._comm_events_this_tick = []
        self._obs_events_this_tick = []
        super().step()

    def replace_universal_clause(self):
        # Capture each environment-driven new clause and the tick it was born.
        res = super().replace_universal_clause()
        cid, old_c, new_c = res
        self._replace_events.append((int(self.steps), new_c))
        return res


# ============================================================
# Network constructors
# ============================================================
# Most types are handled by model.py directly via setup_source="generate".
# Modular (SBM) is built externally and fed in as an input_graph.

def build_modular_graph(N, seed, p_in=0.20, p_out=0.02, n_communities=4):
    """Stochastic block model converted to directed by random edge orientation."""
    base = N // n_communities
    rem = N - base * n_communities
    sizes = [base] * n_communities
    for i in range(rem):
        sizes[i] += 1
    p_matrix = [[p_in if i == j else p_out
                 for j in range(n_communities)]
                for i in range(n_communities)]
    G_und = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for u, v in G_und.edges():
        if u == v:
            continue
        if rng.random() < 0.5:
            G.add_edge(u, v, weight=1.0)
        else:
            G.add_edge(v, u, weight=1.0)
    return G


# ============================================================
# Eigenvector centrality (information-receiver centrality)
# ============================================================

def compute_eigencentrality(G):
    """Return {node: centrality in [0,1]}. Eigenvector centrality on the
    reversed graph (so in-links confer centrality, matching the direction
    information flows in the model). Falls back to normalised in-degree if
    the eigenvector solver does not converge."""
    N = G.number_of_nodes()
    try:
        ec = nx.eigenvector_centrality_numpy(G.reverse(copy=True))
        ec = {n: abs(float(v)) for n, v in ec.items()}   # fix sign ambiguity
        if max(ec.values()) <= 0:
            raise ValueError("degenerate eigenvector")
    except Exception:
        indeg = dict(G.in_degree())
        ec = {n: float(indeg.get(n, 0)) for n in G.nodes()}
    mx = max(ec.values()) if ec else 0.0
    if mx <= 0:
        return {n: 0.0 for n in G.nodes()}
    return {n: ec[n] / mx for n in G.nodes()}


# ============================================================
# Server-side store of the most recent run's full history
# ============================================================
# The browser payload stays light (only KB size + violations per tick). The
# heavy per-tick detail -- each agent's full variable vector x and the exact
# clauses in its knowledge base -- is kept here and fetched on demand via the
# /state and /track endpoints. We keep only the latest run to bound memory.

LAST_RUN = {}          # filled by run_simulation
RUN_COUNTER = [0]


def clause_to_str(clause):
    """Human-readable clause label, e.g. 'x3 AND x7' (variables are 1-indexed
    in the model's clause tuples)."""
    op, var_indices = clause
    a, b = var_indices
    sym = '∧' if op == 'AND' else '⊕'
    return f"x{a} {sym} x{b}"


# ============================================================
# Run one simulation and package the result for the viewer
# ============================================================

def run_simulation(p):
    """p is a parameter dict. Returns a JSON-serialisable payload."""
    random.seed(p['seed'])
    np.random.seed(p['seed'])

    kwargs = dict(
        N=p['N'], K=p['K'], alpha=p['alpha'],
        obs_prob=p['obs_prob'], clause_interval=p['clause_interval'],
        R=p['T'], seed=p['seed'],
    )

    net = p['network']
    if net == 'Random':
        kwargs.update(setup_source='generate', type_network='Random',
                      connect_prob=p['connect_prob'])
    elif net == 'Small World':
        kwargs.update(setup_source='generate', type_network='Small World',
                      n_size=p['n_size'], rewire_prob=p['rewire_prob'])
    elif net == 'Scale Free':
        kwargs.update(setup_source='generate', type_network='Scale Free',
                      min_deg=p['min_deg'])
    elif net == 'Hierarchical':
        kwargs.update(setup_source='generate', type_network='Hierarchical',
                      nlayers=p['nlayers'],
                      intra_layer_connectance=p['intra_layer_conn'],
                      inter_layer_connectance=p['inter_layer_conn'])
    elif net == 'Modular':
        G = build_modular_graph(p['N'], p['seed'],
                                p_in=p['p_in'], p_out=p['p_out'])
        kwargs.update(setup_source='graph', input_graph=G)
    else:
        raise ValueError(f"Unknown network type: {net}")

    m = TrackingModel(redundant_comm=bool(p.get('redundant_comm', False)), **kwargs)

    # Communication-rate normalisation. By default comm_scale = 1/avg-in-strength,
    # so the average agent absorbs ~1 elicitation per tick regardless of topology.
    # Unticking it sets comm_scale = 1, so each in-edge fires at its raw weight
    # (with uniform weights, every in-edge fires every tick) -- higher-degree
    # agents then receive proportionally more, and density/degree starts to matter.
    if not bool(p.get('normalize_comm', True)):
        m.comm_scale = 1.0

    # Spring layout, normalised to 0..1
    G_layout = m.network.to_undirected()
    positions = nx.spring_layout(G_layout, seed=p['seed'], k=0.18, iterations=300)
    xs = [pp[0] for pp in positions.values()]
    ys = [pp[1] for pp in positions.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    positions = {
        int(k): [round(float((v[0] - x_min) / (x_max - x_min)), 4),
                 round(float((v[1] - y_min) / (y_max - y_min)), 4)]
        for k, v in positions.items()
    }

    # Eigenvector centrality. Information flows INTO an agent along its
    # in-edges, so a node is "central" when many central nodes point to it;
    # that is eigenvector centrality on the reversed graph. Robust fallbacks
    # in case the eigvec solver struggles on a poorly-connected digraph.
    centrality = compute_eigencentrality(m.network)

    # Static per-agent info (degrees, neighbour lists, centrality)
    agent_info = []
    for i in range(m.N):
        agent_info.append({
            'id': i,
            'in_deg':  m.network.in_degree(i),
            'out_deg': m.network.out_degree(i),
            'centr':   round(float(centrality[i]), 4),
            'in_nbrs':  sorted([int(n) for n in m.network.predecessors(i)]),
            'out_nbrs': sorted([int(n) for n in m.network.successors(i)]),
        })

    # Clause registry: map each distinct clause (by value) to a small integer
    # id so we can store knowledge bases compactly and label them in the UI.
    clause_ids = {}
    id_to_str = []
    def cid(clause):
        i = clause_ids.get(clause)
        if i is None:
            i = len(id_to_str)
            clause_ids[clause] = i
            id_to_str.append(clause_to_str(clause))
        return i

    N, K, T = m.N, m.K, p['T']
    bpa = (K + 7) // 8                       # bytes needed to pack one agent's K bits

    # Step and record. frames[] is the light browser payload; x_hist / kb_hist
    # are the heavy server-side detail.
    frames = []
    x_hist = []                              # per tick: bytes packing all agents' x
    kb_hist = []                             # per tick: list of array('H') clause-id lists
    c_hist = []                              # per tick: array('H') of the universe's clause ids
    avgV_series = []                         # per tick: population-average violations
    minV_series = []                         # per tick: best-agent violations
    hom_series  = []                         # per tick: homogeneity in [0.5, 1]
    for t in range(T):
        m.step()
        agents_state = []
        xbuf = bytearray(N * bpa)
        kb_tick = []
        for i in range(N):
            ag = m.agent_list[i]
            # [kb_size, violations vs universe C, violations vs own knowledge base]
            kb_viol = int(ag.violation_count(ag.kb, ag.x))
            agents_state.append([len(ag.kb), int(ag.true_violations), kb_viol])
            # pack x bits
            base = i * bpa
            for j, bit in enumerate(ag.x):
                if bit:
                    xbuf[base + (j >> 3)] |= (1 << (j & 7))
            # kb as clause ids
            kb_tick.append(array('H', [cid(c) for c in ag.kb]))
        frames.append({
            'a': agents_state,
            'c': [list(e) for e in m._comm_events_this_tick],
            'o': list(m._obs_events_this_tick),
        })
        x_hist.append(bytes(xbuf))
        kb_hist.append(kb_tick)
        c_hist.append(array('H', [cid(c) for c in m.C]))    # universe at this tick
        avgV_series.append(round(float(m.avg_true_V), 3))
        minV_series.append(int(m.min_true_V))
        hom_series.append(round(float(m.homogeneity), 4))

    # Replacement (drift) events: (birth_tick, clause_id), in order of creation.
    replace_events = [(tk, cid(c)) for (tk, c) in m._replace_events]

    run_id = RUN_COUNTER[0] = RUN_COUNTER[0] + 1
    LAST_RUN.clear()
    LAST_RUN.update({
        'id': run_id, 'N': N, 'K': K, 'T': T, 'bpa': bpa,
        'x_hist': x_hist, 'kb_hist': kb_hist, 'c_hist': c_hist,
        'id_to_str': id_to_str, 'replace_events': replace_events,
    })

    return {
        'run_id': run_id,
        'N': int(N), 'K': int(K), 'M': int(m.M), 'T': int(T),
        'positions': positions,
        'edges': [[int(u), int(v)] for u, v in m.network.edges()],
        'frames': frames,
        'agent_info': agent_info,
        'clause_registry': id_to_str,
        'universe': [list(c) for c in c_hist],   # per-tick list of clause-ids in C
        'series': {'avgV': avgV_series, 'minV': minV_series, 'hom': hom_series},
        'params': p,
    }


# ============================================================
# On-demand detail queries against the stored run
# ============================================================

def get_agent_state(run_id, tick, agent):
    """Return one agent's variable vector and knowledge base at a tick."""
    if not LAST_RUN or LAST_RUN['id'] != run_id:
        return None
    N, K, bpa = LAST_RUN['N'], LAST_RUN['K'], LAST_RUN['bpa']
    T = LAST_RUN['T']
    if not (0 <= tick < T) or not (0 <= agent < N):
        return None
    xbytes = LAST_RUN['x_hist'][tick]
    base = agent * bpa
    x = [(xbytes[base + (j >> 3)] >> (j & 7)) & 1 for j in range(K)]
    # Return clause ids; the client maps them to text and computes which are
    # stale (no longer in the universe) / unknown against its local universe copy.
    kb_ids = [int(i) for i in LAST_RUN['kb_hist'][tick][agent]]
    return {'x': x, 'kb_ids': kb_ids}


def get_track(run_id, after_tick):
    """Pick the next drift-created clause born at or after `after_tick`, and
    return, for every tick, the list of agents whose KB contains it -- i.e.
    its discovery and propagation across the network over time."""
    if not LAST_RUN or LAST_RUN['id'] != run_id:
        return None
    T, N = LAST_RUN['T'], LAST_RUN['N']
    # find the first replacement event strictly after the current tick
    target = None
    for (birth, cl_id) in LAST_RUN['replace_events']:
        if birth >= after_tick:
            target = (birth, cl_id)
            break
    if target is None:
        return {'found': False}
    birth, cl_id = target
    kb_hist = LAST_RUN['kb_hist']
    c_hist = LAST_RUN['c_hist']
    known_by_tick = []
    for t in range(T):
        if t < birth:
            known_by_tick.append([])
            continue
        knowers = [i for i in range(N) if cl_id in kb_hist[t][i]]
        known_by_tick.append(knowers)
    # The clause "dies" when it is replaced out of the universe: first tick
    # at/after birth where it is no longer in C. None if it lasts to the end.
    death_tick = None
    for t in range(birth, T):
        if cl_id not in c_hist[t]:
            death_tick = t
            break
    return {
        'found': True,
        'clause_id': cl_id,
        'clause_str': LAST_RUN['id_to_str'][cl_id],
        'birth_tick': birth,
        'death_tick': death_tick,
        'known_by_tick': known_by_tick,
    }


# ============================================================
# Embedded single-page HTML
# ============================================================

PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Problem-solving model -- interactive</title>
<script src="/d3.v7.min.js"></script>
<style>
  body  { margin: 0; padding: 18px 24px; font-family: -apple-system, system-ui, sans-serif;
          background: #0b0d10; color: #ddd; }
  h1    { font-size: 18px; margin: 0 0 4px 0; color: #fff; font-weight: 500; }
  p.subtitle { font-size: 13px; line-height: 1.5; max-width: 950px; color: #aaa; margin: 0 0 14px 0; }

  details.form-card { background: #15181c; border-radius: 8px; padding: 8px 14px; margin-bottom: 14px; }
  details.form-card summary { font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
                              color: #aaa; cursor: pointer; padding: 4px 0; }
  form .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-top: 10px; }
  form label { display: flex; flex-direction: column; gap: 3px; font-size: 11px; color: #888; }
  form input[type=number], form select {
    background: #0d0f12; color: #eee; border: 1px solid #2a2f37; border-radius: 4px;
    padding: 5px 7px; font-size: 13px; width: 90px;
  }
  form select { width: 200px; }
  form button { padding: 7px 14px; border: 0; background: #3a5; color: #fff; border-radius: 4px;
                cursor: pointer; font-size: 13px; }
  form button:hover { background: #4b6; }
  form button:disabled { background: #444; cursor: not-allowed; }
  #loading-msg { font-size: 12px; color: #ccc; margin-left: 8px; }

  .layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 18px; align-items: start; }
  /* Network column stays put at the top of the viewport; the side panel
     scrolls independently so the agent stats are never pushed off-screen. */
  .netcol { position: sticky; top: 12px; }
  svg#netsvg { background: #07080a; border-radius: 8px; display: block; width: 100%; height: auto; }
  .edge       { stroke: #1c1f24; stroke-width: 0.6; opacity: 0.9; fill: none; }
  .node       { stroke: #000; stroke-opacity: 0.4; stroke-width: 0.5; cursor: pointer; }
  .node:hover { stroke: #fff; stroke-opacity: 0.6; stroke-width: 1.2; }
  .packet     { fill: #ffdd44; }
  .obs-pulse  { fill: none; stroke: #66ddff; stroke-width: 1.4; opacity: 0.85; }
  .sel-ring   { fill: none; stroke: #ffffff; stroke-width: 2; opacity: 0.95; }
  /* V label drawn over each node. Black outline + white fill for legibility
     against any color in the green-yellow-red colormap. */
  .node-label { fill: #fff; stroke: #000; stroke-width: 2.5px; paint-order: stroke;
                font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
                font-weight: 700; pointer-events: none;
                text-anchor: middle; dominant-baseline: central; }

  .panel { background: #15181c; border-radius: 8px; padding: 12px 14px; font-size: 12px;
           position: sticky; top: 12px; max-height: calc(100vh - 24px); overflow-y: auto; }
  .panel h3 { margin: 14px 0 6px 0; font-size: 11px; text-transform: uppercase;
              letter-spacing: 0.06em; color: #888; font-weight: 600; }
  .panel h3:first-child { margin-top: 0; }
  .panel button { padding: 6px 12px; border: 0; background: #2a2f37; color: #ddd;
                  border-radius: 4px; cursor: pointer; font-size: 13px; margin-right: 6px; }
  .panel button:hover { background: #3a414b; }
  .panel input[type=range] { width: 100%; margin: 4px 0 10px 0; }
  .panel label { display: block; font-size: 11px; color: #888; margin-top: 6px; }

  .stat { display: flex; justify-content: space-between; padding: 2px 0; }
  .stat .v { color: #fff; font-variant-numeric: tabular-nums; }

  .agent-detail .nbrs { margin-top: 10px; }
  .nbr-list {
    background: #0a0c0f; border-radius: 4px; padding: 4px 7px; font-family: ui-monospace, monospace;
    font-size: 11px; color: #aaa; word-break: break-all; max-height: 80px; overflow-y: auto;
    margin-top: 3px;
  }
  .nbr-list div { line-height: 1.35; }
  .placeholder { color: #777; font-style: italic; padding: 6px 0; }

  .legend { font-size: 11px; line-height: 1.5; color: #aaa; }
  .legend .bar { height: 8px; border-radius: 4px;
                 background: linear-gradient(to right, #d62828, #f3a712, #4caf50);
                 margin: 4px 0 2px 0; }
  .legend .row { margin-top: 6px; }
  .legend .swatch { display: inline-block; width: 9px; height: 9px; background: #ffdd44;
                    border-radius: 50%; vertical-align: middle; margin: 0 4px 0 0; }
  .legend .swatch.obs { background: transparent; border: 1.5px solid #66ddff; }

  /* Tabs */
  .tabbar { display: flex; gap: 6px; margin: 0 0 12px 0; }
  .tabbar button { padding: 7px 16px; border: 1px solid #232830; background: #14171b; color: #9aa3ad;
                   border-radius: 6px 6px 0 0; cursor: pointer; font-size: 13px; }
  .tabbar button.active { background: #1e2937; color: #fff; border-color: #2e3b4d; font-weight: 600; }

  /* Performance tab */
  #tab-perf { background: #14171b; border: 1px solid #232830; border-radius: 8px; padding: 14px 18px; }
  .perf-toolbar { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .perf-toolbar button { padding: 7px 14px; border: 0; border-radius: 5px; cursor: pointer;
                         font-size: 13px; font-weight: 600; }
  #snap-save { background: #2e9d5b; color: #fff; }
  #snap-save:hover { background: #36b369; }
  .perf-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 1100px) { .perf-charts { grid-template-columns: 1fr; } }
  .perf-charts h3 { font-size: 12px; color: #cdd3da; margin: 0 0 4px 0; font-weight: 600; }
  .perf-charts svg { width: 100%; height: auto; background: #07080a; border-radius: 8px; }
  .chart-actions { margin-top: 4px; }
  .chart-actions button { font-size: 11px; padding: 3px 9px; border: 0; border-radius: 4px;
                          background: #2a2f37; color: #ccc; cursor: pointer; }
  #snap-list { margin-top: 6px; }
  .snap-row { display: flex; align-items: center; gap: 8px; padding: 5px 0;
              border-bottom: 1px solid #1c2128; font-size: 12px; }
  .snap-row .sw { width: 12px; height: 12px; border-radius: 3px; flex: none; }
  .snap-row .lbl { flex: 1; color: #cdd3da; }
  .snap-row button { font-size: 11px; padding: 2px 8px; border: 0; border-radius: 4px;
                     background: #3a2a2f; color: #e8a; cursor: pointer; }
  .snap-empty { color: #6c7884; font-style: italic; padding: 6px 0; }
</style>
</head>
<body>

<h1>Problem-solving model -- interactive</h1>
<p class="subtitle">
  Set parameters below, click <strong>Generate</strong>, then explore the run.
  Node <strong>size</strong> = clauses learned. Node <strong>colour</strong>: green = accurate, red = many violations.
  <span style="color:#ffdd44">Yellow dots</span> are clauses being transmitted; <span style="color:#66ddff">blue rings</span> are private observations.
  Click any node to inspect it.
</p>

<details class="form-card" open>
  <summary>Parameters</summary>
  <form id="form" onsubmit="return false;">
    <div class="row">
      <label>N (agents) <input type="number" name="N" value="80" min="4" max="500"></label>
      <label>K (variables) <input type="number" name="K" value="30" min="2" max="200"></label>
      <label>alpha <input type="number" name="alpha" value="2" step="0.1" min="0.1" max="10"></label>
      <label>obs_prob <input type="number" name="obs_prob" value="0.01" step="0.005" min="0" max="1"></label>
      <label>clause_interval <input type="number" name="clause_interval" value="10" min="1" max="1000"></label>
      <label>T (ticks) <input type="number" name="T" value="500" min="20" max="3000"></label>
      <label>seed <input type="number" name="seed" value="42"></label>
      <label style="flex-direction:row;align-items:center;gap:6px;white-space:nowrap;"
             title="If on, a neighbour shares a clause drawn from its whole knowledge base; if the recipient already knows it, nothing is transferred (the elicitation is wasted).">
        <input type="checkbox" name="redundant_comm"> redundant communication
      </label>
      <label style="flex-direction:row;align-items:center;gap:6px;white-space:nowrap;"
             title="On (default): per-agent intake is normalised to ~1 elicitation/tick regardless of topology. Off: each in-edge fires at its raw weight, so higher-degree agents receive more.">
        <input type="checkbox" name="normalize_comm" checked> normalise comm rate
      </label>
    </div>
    <div class="row">
      <label>Network type
        <select name="network" id="network">
          <option value="Random">Random (Erdos-Renyi)</option>
          <option value="Small World">Small World (Watts-Strogatz)</option>
          <option value="Scale Free">Scale Free (Barabasi-Albert)</option>
          <option value="Hierarchical">Hierarchical (layered)</option>
          <option value="Modular">Modular (SBM)</option>
        </select>
      </label>
      <span class="net-params" data-net="Random">
        <label>connect_prob <input type="number" name="connect_prob" value="0.10" step="0.01" min="0.001" max="1"></label>
      </span>
      <span class="net-params" data-net="Small World" style="display:none">
        <label>n_size <input type="number" name="n_size" value="4" min="2" max="20"></label>
        <label>rewire_prob <input type="number" name="rewire_prob" value="0.1" step="0.05" min="0" max="1"></label>
      </span>
      <span class="net-params" data-net="Scale Free" style="display:none">
        <label>min_deg <input type="number" name="min_deg" value="3" min="1" max="20"></label>
      </span>
      <span class="net-params" data-net="Hierarchical" style="display:none">
        <label>nlayers <input type="number" name="nlayers" value="4" min="2" max="10"></label>
        <label>intra_layer_conn <input type="number" name="intra_layer_conn" value="0.20" step="0.05" min="0" max="1"></label>
        <label>inter_layer_conn <input type="number" name="inter_layer_conn" value="0.05" step="0.01" min="0" max="1"></label>
      </span>
      <span class="net-params" data-net="Modular" style="display:none">
        <label>p_in (intra) <input type="number" name="p_in" value="0.20" step="0.05" min="0" max="1"></label>
        <label>p_out (inter) <input type="number" name="p_out" value="0.02" step="0.01" min="0" max="1"></label>
      </span>
      <button type="button" id="generate">Generate</button>
      <span id="loading-msg"></span>
    </div>
  </form>
</details>

<div class="tabbar">
  <button id="tabbtn-live" class="active">Live network</button>
  <button id="tabbtn-perf">Performance</button>
</div>

<div id="tab-live">
<div class="layout">
  <div class="netcol">
    <svg id="netsvg" viewBox="0 0 1000 700" preserveAspectRatio="xMidYMid meet"></svg>
  </div>

  <div class="panel">
    <h3>Controls</h3>
    <button id="play">Pause</button>
    <button id="stepBack1">&minus;1 tick</button>
    <button id="step1">+1 tick</button>
    <button id="restart">Restart</button>
    <label>Speed (ticks/sec): <span id="speed-display">12</span></label>
    <input id="speed" type="range" min="1" max="60" value="12">
    <label>Scrub:</label>
    <input id="tick" type="range" min="0" max="0" value="0">

    <h3>Track a new clause</h3>
    <button id="track-toggle">Track next new clause</button>
    <div id="track-info" style="font-size:11px;color:#9aa3ad;margin-top:6px;"></div>

    <h3>Aggregate stats</h3>
    <div class="stat"><span>Tick</span><span class="v" id="s-tick">-</span></div>
    <div class="stat"><span>avg violations</span><span class="v" id="s-avg">-</span></div>
    <div class="stat"><span>min violations</span><span class="v" id="s-min">-</span></div>
    <div class="stat"><span>mean KB size</span><span class="v" id="s-kb">-</span></div>
    <div class="stat"><span>comm events</span><span class="v" id="s-comm">-</span></div>

    <h3>All clauses (universe now)</h3>
    <div id="universe-count" style="font-size:11px;color:#9aa3ad;margin-bottom:3px;"></div>
    <div id="universe-list" class="nbr-list" style="max-height:150px;"></div>

    <h3>Centrality vs violations</h3>
    <svg id="scatter" viewBox="0 0 260 180" style="width:100%;height:auto;
         background:#0a0c0f;border-radius:6px;"></svg>
    <div style="font-size:10px;color:#7d8794;margin-top:2px;">
      each dot = one agent at the current tick; line = least-squares fit
    </div>

    <h3>Selected agent</h3>
    <div id="agent-info">
      <div class="placeholder">Click an agent on the network to inspect it.</div>
    </div>

    <h3>Legend</h3>
    <div class="legend">
      Violation count (colour):
      <div class="bar"></div>
      <div style="display:flex; justify-content: space-between; font-size: 10px; color: #888;">
        <span>0</span><span id="leg-max">M</span>
      </div>
      <div class="row"><span class="swatch"></span>elicitation packet</div>
      <div class="row"><span class="swatch obs"></span>observation pulse</div>
    </div>
  </div>
</div>
</div><!-- /tab-live -->

<div id="tab-perf" style="display:none;">
  <div class="perf-toolbar">
    <button id="snap-save">📷 Save snapshot of this run</button>
    <span id="snap-msg" style="font-size:12px;color:#9aa3ad;"></span>
  </div>
  <div class="perf-charts">
    <div>
      <h3>Average violations over time</h3>
      <svg id="chart-avgV" viewBox="0 0 760 320"></svg>
      <div class="chart-actions"><button data-chart="avgV">Download PNG</button></div>
    </div>
    <div>
      <h3>Homogeneity over time</h3>
      <svg id="chart-hom" viewBox="0 0 760 320"></svg>
      <div class="chart-actions"><button data-chart="hom">Download PNG</button></div>
    </div>
  </div>
  <h3 style="font-size:12px;color:#7d8794;text-transform:uppercase;letter-spacing:.06em;margin-top:16px;">
    Saved snapshots (tick the boxes to overlay &amp; compare)</h3>
  <div id="snap-list"></div>
</div>

<script>
let DATA = null;
let simTick = 0;
let animMs = 0;            // timeline clock: advances only while playing
let nowMs = 0;            // wall clock: always advances (drives packet flight)
let lastTimestamp = null;
let playing = false;
let ticksPerSec = 12;
let packets = [];
let pulses  = [];
let selectedAgent = null;
let trackMode = false;     // "track next new clause" overlay on/off
let trackData = null;      // {found, clause_str, birth_tick, known_by_tick:[...]}

const W = 1000, H = 700, PAD = 30;
const SVGNS = 'http://www.w3.org/2000/svg';   // used by the scatter-plot drawer
const svg = d3.select('#netsvg').attr('viewBox', `0 0 ${W} ${H}`);
let edgeLayer, packetLayer, obsLayer, nodeLayer, selRingLayer, textLayer;

function pixelPos(nodeId) {
  const p = DATA.positions[nodeId];
  return [PAD + p[0] * (W - 2*PAD), PAD + p[1] * (H - 2*PAD)];
}

function initLayers() {
  svg.selectAll('*').remove();
  edgeLayer    = svg.append('g').attr('class', 'edges');
  packetLayer  = svg.append('g').attr('class', 'packets');
  obsLayer     = svg.append('g').attr('class', 'obs');
  selRingLayer = svg.append('g').attr('class', 'selring');
  nodeLayer    = svg.append('g').attr('class', 'nodes');
  textLayer    = svg.append('g').attr('class', 'labels');
}

function rebuildNetwork() {
  initLayers();

  edgeLayer.selectAll('line').data(DATA.edges).enter().append('line')
    .attr('class', 'edge')
    .attr('x1', d => pixelPos(d[0])[0]).attr('y1', d => pixelPos(d[0])[1])
    .attr('x2', d => pixelPos(d[1])[0]).attr('y2', d => pixelPos(d[1])[1]);

  nodeLayer.selectAll('circle').data(d3.range(DATA.N)).enter().append('circle')
    .attr('class', 'node')
    .attr('cx', d => pixelPos(d)[0])
    .attr('cy', d => pixelPos(d)[1])
    .attr('r', 3)
    .on('click', function(event, d) {
      event.stopPropagation();
      selectedAgent = (selectedAgent === d) ? null : d;
      updateSelectionRing();
      updateAgentPanel();
    });

  // Numeric V label, drawn on top of each node
  textLayer.selectAll('text').data(d3.range(DATA.N)).enter().append('text')
    .attr('class', 'node-label')
    .attr('x', d => pixelPos(d)[0])
    .attr('y', d => pixelPos(d)[1])
    .text('-');

  // Click background to deselect
  svg.on('click', function() {
    selectedAgent = null;
    updateSelectionRing();
    updateAgentPanel();
  });
}

const violationColor = (v) => {
  // Custom inline: 0 -> green, M*0.9 -> red
  const max = DATA.M * 0.9;
  const t = Math.max(0, Math.min(1, v / max));
  return d3.interpolateRdYlGn(1 - t);
};

function updateNodes(t) {
  const agents = DATA.frames[t].a;
  // In track mode, colour by whether the agent holds the tracked clause:
  //   blue = holds it, grey = doesn't. (Colour stays blue even after the
  //   clause is removed from the universe; the info banner notes the removal.)
  // Otherwise colour by violation count as usual.
  let knowsSet = null;
  if (trackMode && trackData && trackData.found) {
    knowsSet = new Set(trackData.known_by_tick[t] || []);
  }
  nodeLayer.selectAll('circle')
    .attr('r', (d, i) => 3 + 20 * (agents[i][0] / DATA.M))
    .attr('fill', (d, i) => {
      if (knowsSet) return knowsSet.has(i) ? '#2b8cff' : '#2a2f37';
      return violationColor(agents[i][1]);
    });
  textLayer.selectAll('text')
    .attr('font-size', (d, i) => {
      const r = 3 + 20 * (agents[i][0] / DATA.M);
      return Math.max(9, Math.min(14, r * 0.95)) + 'px';
    })
    .text((d, i) => agents[i][1]);
}

function updateSelectionRing() {
  selRingLayer.selectAll('*').remove();
  if (selectedAgent === null || !DATA) return;
  const [x, y] = pixelPos(selectedAgent);
  const cur = DATA.frames[simTick].a[selectedAgent];
  const r = 3 + 20 * (cur[0] / DATA.M);
  selRingLayer.append('circle').attr('class', 'sel-ring')
    .attr('cx', x).attr('cy', y).attr('r', r + 4);
}

function spawnEventsForFrame(t) {
  const f = DATA.frames[t];
  for (const [src, tgt] of f.c) packets.push({src, tgt, born: nowMs, dur: 700});
  for (const nid of f.o) pulses.push({node: nid, born: nowMs, dur: 900});
}

function renderPacketsAndPulses() {
  packets = packets.filter(p => nowMs - p.born < p.dur);
  const packSel = packetLayer.selectAll('circle')
    .data(packets, p => p.born + ':' + p.src + '-' + p.tgt);
  packSel.exit().remove();
  packSel.enter().append('circle').attr('class', 'packet').attr('r', 2.5);
  packetLayer.selectAll('circle')
    .attr('cx', p => {
      const u = (nowMs - p.born) / p.dur;
      const [x1, y1] = pixelPos(p.src);
      const [x2, y2] = pixelPos(p.tgt);
      return x1 + u * (x2 - x1);
    })
    .attr('cy', p => {
      const u = (nowMs - p.born) / p.dur;
      const [x1, y1] = pixelPos(p.src);
      const [x2, y2] = pixelPos(p.tgt);
      return y1 + u * (y2 - y1);
    })
    .attr('opacity', p => {
      const u = (nowMs - p.born) / p.dur;
      return 0.95 * (1 - u * u);
    });

  pulses = pulses.filter(p => nowMs - p.born < p.dur);
  const pulseSel = obsLayer.selectAll('circle')
    .data(pulses, p => p.born + ':' + p.node);
  pulseSel.exit().remove();
  pulseSel.enter().append('circle').attr('class', 'obs-pulse');
  obsLayer.selectAll('circle')
    .attr('cx', p => pixelPos(p.node)[0])
    .attr('cy', p => pixelPos(p.node)[1])
    .attr('r',  p => {
      const u = (nowMs - p.born) / p.dur;
      return 4 + 30 * u;
    })
    .attr('opacity', p => {
      const u = (nowMs - p.born) / p.dur;
      return 0.8 * (1 - u);
    });
}

function updateAggStats(t) {
  const agents = DATA.frames[t].a;
  const Vs = agents.map(a => a[1]);
  const KBs = agents.map(a => a[0]);
  const avgV = Vs.reduce((s, x) => s + x, 0) / Vs.length;
  const meanKB = KBs.reduce((s, x) => s + x, 0) / KBs.length;
  document.getElementById('s-tick').textContent = t + 1;
  document.getElementById('s-avg').textContent  = avgV.toFixed(2);
  document.getElementById('s-min').textContent  = Math.min(...Vs);
  document.getElementById('s-kb').textContent   = meanKB.toFixed(1);
  document.getElementById('s-comm').textContent = DATA.frames[t].c.length;
}

// Scatter: violations (x) vs eigen-centrality (y), with least-squares fit.
function updateScatter(t) {
  const sc = document.getElementById('scatter');
  while (sc.firstChild) sc.removeChild(sc.firstChild);
  if (!DATA) return;
  const VB_W = 260, VB_H = 180, mL = 30, mR = 10, mT = 8, mB = 26;
  const plotW = VB_W - mL - mR, plotH = VB_H - mT - mB;
  const xs = DATA.frames[t].a.map(s => s[1]);            // violations 0..M  (x)
  const ys = DATA.agent_info.map(a => a.centr);          // centrality 0..1  (y)
  const xpx = v => mL + (v / DATA.M) * plotW;            // violations on x
  const ypx = c => mT + plotH - c * plotH;               // centrality on y

  const mk = (tag, at) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in at) e.setAttribute(k, at[k]);
    return e;
  };
  // axes
  sc.appendChild(mk('line', {x1: mL, y1: mT, x2: mL, y2: mT + plotH, stroke: '#333', 'stroke-width': 1}));
  sc.appendChild(mk('line', {x1: mL, y1: mT + plotH, x2: mL + plotW, y2: mT + plotH, stroke: '#333', 'stroke-width': 1}));
  // y-axis label (eigen-centrality) + 0/1 ticks
  const lblY = mk('text', {x: 8, y: mT + plotH / 2, fill: '#7d8794', 'font-size': 9,
                           'text-anchor': 'middle', transform: `rotate(-90 8 ${mT + plotH / 2})`});
  lblY.textContent = 'eigen-centrality';
  sc.appendChild(lblY);
  const y0 = mk('text', {x: mL - 4, y: mT + plotH, fill: '#7d8794', 'font-size': 8, 'text-anchor': 'end'}); y0.textContent = '0';
  const y1 = mk('text', {x: mL - 4, y: mT + 6, fill: '#7d8794', 'font-size': 8, 'text-anchor': 'end'}); y1.textContent = '1';
  sc.appendChild(y0); sc.appendChild(y1);
  // x-axis label (violations) + 0/M ticks
  const lblX = mk('text', {x: mL + plotW / 2, y: VB_H - 3, fill: '#7d8794', 'font-size': 9, 'text-anchor': 'middle'});
  lblX.textContent = 'violations';
  sc.appendChild(lblX);
  const x0 = mk('text', {x: mL, y: mT + plotH + 11, fill: '#7d8794', 'font-size': 8, 'text-anchor': 'middle'}); x0.textContent = '0';
  const xM = mk('text', {x: mL + plotW, y: mT + plotH + 11, fill: '#7d8794', 'font-size': 8, 'text-anchor': 'middle'}); xM.textContent = DATA.M;
  sc.appendChild(x0); sc.appendChild(xM);

  // dots
  for (let i = 0; i < xs.length; i++)
    sc.appendChild(mk('circle', {cx: xpx(xs[i]), cy: ypx(ys[i]), r: 2.4,
                                 fill: '#5cd4ff', 'fill-opacity': 0.6}));

  // least-squares fit: centrality (y) on violations (x)
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  if (sxx > 1e-9) {
    const b1 = sxy / sxx, b0 = my - b1 * mx;
    const xa = Math.min(...xs), xb = Math.max(...xs);
    sc.appendChild(mk('line', {x1: xpx(xa), y1: ypx(b0 + b1 * xa),
                               x2: xpx(xb), y2: ypx(b0 + b1 * xb),
                               stroke: '#ff9f43', 'stroke-width': 1.6}));
  }
}

function updateAgentPanel() {
  const panel = document.getElementById('agent-info');
  if (selectedAgent === null || !DATA) {
    panel.innerHTML = '<div class="placeholder">Click an agent on the network to inspect it.</div>';
    return;
  }
  const a = DATA.agent_info[selectedAgent];
  const cur = DATA.frames[simTick].a[selectedAgent];   // [kb_size, V_universe, V_kb]

  // Two series over time: violations vs the universe C, and vs the agent's own KB.
  const sub = DATA.frames.slice(0, simTick + 1).map(f => f.a[selectedAgent]);
  const Vs  = sub.map(f => f[1]);                       // vs universe
  const Ks  = sub.map(f => (f[2] != null ? f[2] : 0));  // vs own KB
  const sparkW = 230, sparkH = 50;
  const xS = (i) => (Vs.length <= 1 ? sparkW / 2 : i * sparkW / (Vs.length - 1));
  const yMax = Math.max(DATA.M, ...Ks, ...Vs) || 1;
  const yS = (v) => sparkH - (v / yMax) * sparkH;
  const toPath = arr => arr.map((v, i) => `${i === 0 ? 'M' : 'L'}${xS(i).toFixed(1)} ${yS(v).toFixed(1)}`).join(' ');
  const pathV = toPath(Vs), pathK = toPath(Ks);

  const bestV = Math.min(...Vs);
  const meanV = (Vs.reduce((s, v) => s + v, 0) / Vs.length).toFixed(2);

  panel.innerHTML = `
    <div class="agent-detail">
      <div class="stat"><span>ID</span><span class="v">${a.id}</span></div>
      <div class="stat"><span>in-degree</span><span class="v">${a.in_deg}</span></div>
      <div class="stat"><span>out-degree</span><span class="v">${a.out_deg}</span></div>
      <div class="stat"><span>eigen-centrality</span><span class="v">${a.centr.toFixed(3)}</span></div>
      <div class="stat"><span>KB size</span><span class="v">${cur[0]} / ${DATA.M}</span></div>
      <div class="stat"><span>violations vs universe C</span><span class="v" style="color:#88c">${cur[1]}</span></div>
      <div class="stat"><span>violations vs own KB</span><span class="v" style="color:#ffb454">${cur[2] != null ? cur[2] : '–'}</span></div>
      <div class="stat"><span>V (best so far, vs C)</span><span class="v">${bestV}</span></div>
      <div class="stat"><span>V (mean so far, vs C)</span><span class="v">${meanV}</span></div>

      <div style="font-size: 10px; color: #888; margin-top: 8px;">
        violations over time (x: 1–${simTick + 1}) —
        <span style="color:#88c">vs universe</span> ·
        <span style="color:#ffb454">vs own KB</span>:
      </div>
      <svg width="${sparkW}" height="${sparkH}" style="display:block; margin-top: 2px;
           background: #0a0c0f; border-radius: 3px;">
        <path d="${pathK}" fill="none" stroke="#ffb454" stroke-width="1.2" opacity="0.9"/>
        <path d="${pathV}" fill="none" stroke="#88c" stroke-width="1.4"/>
      </svg>

      <div class="nbrs">
        <div style="font-size: 10px; color: #888; margin-top: 8px;">
          In-neighbours (${a.in_nbrs.length}):
        </div>
        <div class="nbr-list">${a.in_nbrs.length > 0 ? a.in_nbrs.join(', ') : '(none)'}</div>
        <div style="font-size: 10px; color: #888; margin-top: 8px;">
          Out-neighbours (${a.out_nbrs.length}):
        </div>
        <div class="nbr-list">${a.out_nbrs.length > 0 ? a.out_nbrs.join(', ') : '(none)'}</div>
      </div>

      <div id="agent-kbx"></div>
    </div>
  `;
  renderDetailInto();    // fill in this agent's variable choices + KB at this tick
}

// ---- variable choices (x) and knowledge-base contents, fetched on demand ----
let detailCache = { key: null, html: null };

// d = { x: [0/1,...], kb_ids: [clause-id,...] }. We classify each KB clause as
// current or STALE (no longer in the universe at this tick), and list the
// current-universe clauses the agent hasn't learned (unknown).
function renderKBX(d, tick) {
  const reg = DATA.clause_registry;
  const uniIds = DATA.universe[tick] || [];
  const uniSet = new Set(uniIds);
  const kbSet = new Set(d.kb_ids);

  let cells = '';
  for (let j = 0; j < d.x.length; j++) {
    const on = d.x[j] === 1;
    cells += `<span title="x${j+1} = ${d.x[j]}" style="display:inline-block;`
           + `width:13px;height:13px;margin:1px;border-radius:2px;font-size:7px;`
           + `line-height:13px;text-align:center;color:${on?'#04220f':'#7d8794'};`
           + `background:${on?'#4caf50':'#1c2128'};">${d.x[j]}</span>`;
  }

  // KB clauses, stale ones flagged
  let staleCount = 0;
  const kbList = d.kb_ids.length
    ? d.kb_ids.map(id => {
        const stale = !uniSet.has(id);
        if (stale) staleCount++;
        return stale
          ? `<div style="color:#d9a441;">${reg[id]} <span style="opacity:.7;">(stale)</span></div>`
          : `<div>${reg[id]}</div>`;
      }).join('')
    : '(empty)';

  // current-universe clauses the agent doesn't know
  const unknownIds = uniIds.filter(id => !kbSet.has(id));
  const unknownList = unknownIds.length
    ? unknownIds.map(id => `<div>${reg[id]}</div>`).join('')
    : '(knows every current constraint)';

  return `
    <div style="font-size:10px;color:#7d8794;margin-top:10px;">
      variable choices x (${d.x.length} vars; green = 1):
    </div>
    <div style="margin-top:3px;line-height:0;">${cells}</div>
    <div style="font-size:10px;color:#7d8794;margin-top:10px;">
      ✓ knowledge base (${d.kb_ids.length} clauses${staleCount ? `, <span style="color:#d9a441;">${staleCount} stale</span>` : ''}):
    </div>
    <div class="nbr-list" style="max-height:120px;">${kbList}</div>
    <div style="font-size:10px;color:#7d8794;margin-top:10px;">
      ✗ unknown constraints (${unknownIds.length} of ${uniIds.length} current — not yet learned):
    </div>
    <div class="nbr-list" style="max-height:120px;color:#d98a8a;">${unknownList}</div>
  `;
}

function renderDetailInto() {
  const box = document.getElementById('agent-kbx');
  if (!box || selectedAgent === null || !DATA) return;
  const key = DATA.run_id + ':' + selectedAgent + ':' + simTick;
  if (detailCache.key === key && detailCache.html) { box.innerHTML = detailCache.html; return; }
  box.innerHTML = '<div style="color:#6c7884;font-style:italic;margin-top:8px;">loading…</div>';
  fetchAgentDetail(selectedAgent, simTick, key);
}

async function fetchAgentDetail(agent, tick, key) {
  try {
    const r = await fetch(`/state?run=${DATA.run_id}&tick=${tick}&agent=${agent}`);
    if (!r.ok) return;
    const d = await r.json();
    const html = renderKBX(d, tick);
    detailCache = { key, html };
    // only paint if the user hasn't moved on since the request was issued
    if (selectedAgent === agent && simTick === tick) {
      const box = document.getElementById('agent-kbx');
      if (box) box.innerHTML = html;
    }
  } catch (e) { /* ignore transient fetch errors */ }
}

// ---- "All clauses (universe now)" panel; computed locally from DATA.universe ----
let lastUniverseKey = null;
function updateUniverse(t) {
  if (!DATA || !DATA.universe) return;
  const ids = DATA.universe[t] || [];
  const key = t;                       // universe content is fixed per tick
  if (key === lastUniverseKey) return; // avoid needless DOM churn
  lastUniverseKey = key;
  const reg = DATA.clause_registry;
  const cnt = document.getElementById('universe-count');
  const list = document.getElementById('universe-list');
  if (cnt) cnt.textContent = `${ids.length} constraints in C at tick ${t + 1}`;
  if (list) list.innerHTML = ids.map(id => `<div>${reg[id]}</div>`).join('');
}

// The animation is driven by a setInterval timer (not requestAnimationFrame)
// so it keeps advancing reliably even when the tab is briefly backgrounded
// (browsers pause rAF in hidden tabs). dt is measured from a real clock.
function frame() {
  if (!DATA) return;
  const timestamp = performance.now();
  if (lastTimestamp === null) lastTimestamp = timestamp;
  let dt = timestamp - lastTimestamp;
  lastTimestamp = timestamp;
  if (dt > 250) dt = 250;     // cap a long gap (e.g. tab was hidden) so we don't leap
  nowMs = timestamp;          // wall clock, always advances

  // A bad frame must never kill the loop, so guard the body.
  try {
    if (playing) {
      animMs += dt;
      const targetTick = Math.floor(animMs * ticksPerSec / 1000);
      while (simTick < targetTick && simTick < DATA.T - 1) {
        simTick++;
        spawnEventsForFrame(simTick);
      }
      if (simTick >= DATA.T - 1) {
        playing = false;
        document.getElementById('play').textContent = 'Play';
      }
      updateNodes(simTick);
      updateSelectionRing();
      updateAggStats(simTick);
      updateScatter(simTick);
      updateUniverse(simTick);
      if (trackMode) updateTrackInfo();
      if (perfVisible && simTick !== lastChartTick) drawPerfCharts();
      document.getElementById('tick').value = simTick;
      if (selectedAgent !== null && simTick % 5 === 0) updateAgentPanel();
    }
    renderPacketsAndPulses();
  } catch (e) {
    console.error('frame error:', e);
  }
}

// Render a single tick everywhere (used by +1, scrub, restart).
function renderTick(t) {
  simTick = t;
  updateNodes(t); updateSelectionRing(); updateAggStats(t);
  updateScatter(t); updateUniverse(t); updateAgentPanel(); updateTrackInfo();
  if (perfVisible) drawPerfCharts();
  document.getElementById('tick').value = t;
}

// --- Track-a-clause overlay ---
function updateTrackInfo() {
  const info = document.getElementById('track-info');
  if (!trackMode || !trackData) { info.textContent = ''; return; }
  if (!trackData.found) {
    info.innerHTML = '<span style="color:#caa">No further new clause is created after this tick.</span>';
    return;
  }
  const known = (trackData.known_by_tick[simTick] || []).length;
  const born = trackData.birth_tick;
  const death = trackData.death_tick;        // null if it never dies
  const dead = (death !== null && death !== undefined && simTick >= death);
  let status;
  if (simTick < born) {
    status = `not yet created (born at tick ${born + 1})`;
  } else if (dead) {
    status = `<span style="color:#d9a441;">removed from the universe at tick ${death + 1}</span>; `
           + `<b style="color:#2b8cff">${known}/${DATA.N}</b> agents still hold a stale copy`;
  } else {
    status = `known by <b style="color:#2b8cff">${known}/${DATA.N}</b> agents`;
  }
  const deathNote = (death !== null && death !== undefined)
    ? ` · dies tick ${death + 1}` : ' · never replaced';
  const legend = 'blue = holds this clause · grey = doesn\'t';
  info.innerHTML =
    `Tracking <b style="color:#fff">${trackData.clause_str}</b> `
    + `(born tick ${born + 1}${deathNote}). ` + status
    + `<br><span style="color:#7d8794">${legend}</span>`;
}

async function toggleTrack() {
  const btn = document.getElementById('track-toggle');
  if (!DATA) return;
  if (trackMode) {                       // turn OFF
    trackMode = false; trackData = null;
    btn.textContent = 'Track next new clause';
    document.getElementById('track-info').textContent = '';
    updateNodes(simTick);                // restore violation colouring
    return;
  }
  // turn ON: ask the server for the next clause born at/after the current tick
  const wasPlaying = playing;            // preserve play/pause state across the jump
  btn.disabled = true;
  document.getElementById('track-info').textContent = 'Finding next new clause…';
  try {
    const r = await fetch(`/track?run=${DATA.run_id}&after=${simTick}`);
    const d = await r.json();
    trackData = d; trackMode = true;
    btn.textContent = 'Stop tracking';
    if (d.found) {
      // Jump to just before its birth so you're positioned to watch it spread.
      // Keep playing if it was playing; stay paused if it was paused.
      const start = Math.max(0, d.birth_tick - 1);
      packets = []; pulses = [];
      animMs = start * 1000 / ticksPerSec;
      renderTick(start);
      setPlaying(wasPlaying);
    } else {
      updateTrackInfo();
    }
  } catch (e) {
    document.getElementById('track-info').textContent = 'Track error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

// --- Controls ---
// Single source of truth for play/pause, so the button label and `playing`
// can never drift apart. Starting play at the very end rewinds to the start.
function setPlaying(on) {
  if (!DATA) return;
  if (on && simTick >= DATA.T - 1) {      // at the end -> rewind and play
    simTick = 0; animMs = 0; packets = []; pulses = [];
    renderTick(0);
  }
  playing = on;
  lastTimestamp = null;                   // avoid a dt spike when resuming
  document.getElementById('play').textContent = on ? 'Pause' : 'Play';
}

document.getElementById('track-toggle').addEventListener('click', toggleTrack);

document.getElementById('play').onclick = () => setPlaying(!playing);

document.getElementById('step1').onclick = () => {
  if (!DATA) return;
  // MOVE to the next tick: spawn this tick's clause transfers so you watch
  // them fly across their links, rather than instantly switching state.
  setPlaying(false);
  if (simTick < DATA.T - 1) {
    simTick++;
    spawnEventsForFrame(simTick);            // packets fly even while paused
    animMs = simTick * 1000 / ticksPerSec;   // keep the timeline in sync
    renderTick(simTick);
  }
};

document.getElementById('stepBack1').onclick = () => {
  if (!DATA) return;
  // Step one tick BACK. No packets (we are rewinding, not transferring).
  setPlaying(false);
  if (simTick > 0) {
    simTick--;
    packets = []; pulses = [];
    animMs = simTick * 1000 / ticksPerSec;
    renderTick(simTick);
  }
};

document.getElementById('restart').onclick = () => {
  if (!DATA) return;
  simTick = 0; animMs = 0; packets = []; pulses = [];
  renderTick(0);
  setPlaying(true);
};

document.getElementById('speed').oninput = (e) => {
  ticksPerSec = parseInt(e.target.value);
  document.getElementById('speed-display').textContent = ticksPerSec;
  // Rescale the timeline clock so the CURRENT tick is preserved; otherwise
  // changing speed makes simTick jump (which looked glitchy, esp. while tracking).
  animMs = simTick * 1000 / ticksPerSec;
  lastTimestamp = null;
};

document.getElementById('tick').oninput = (e) => {
  if (!DATA) return;
  setPlaying(false);                        // scrubbing pauses, for predictability
  const t = parseInt(e.target.value);
  animMs = t * 1000 / ticksPerSec;
  packets = []; pulses = [];
  renderTick(t);
};

// --- Network-type field toggling ---
const netSel = document.getElementById('network');
function toggleNetParams() {
  document.querySelectorAll('.net-params').forEach(el => {
    el.style.display = (el.dataset.net === netSel.value) ? 'inline-flex' : 'none';
  });
}
netSel.onchange = () => {
  // Only swap which parameter fields are shown. The user clicks Generate
  // when they're ready -- selecting a type does not regenerate on its own.
  toggleNetParams();
};
toggleNetParams();

// --- Generation: read params, run the sim, render. Called directly by the
// Generate button, the network dropdown, and the on-load auto-run. Does NOT
// rely on form-submit semantics (which can silently reload the page).
async function runGeneration() {
  const form = document.getElementById('form');
  const params = {};
  for (const el of form.querySelectorAll('input, select')) {
    // Only collect visible/active fields. Hidden net-param fields are skipped.
    const wrap = el.closest('.net-params');
    if (wrap && wrap.style.display === 'none') continue;
    params[el.name] = (el.type === 'checkbox') ? el.checked : el.value;
  }
  const btn = document.getElementById('generate');
  const msg = document.getElementById('loading-msg');
  btn.disabled = true;
  msg.style.color = '#ccc';
  // Live elapsed-time counter so a long run (T=3000 is ~45s) visibly
  // progresses and never looks frozen.
  const t0 = performance.now();
  const ticksReq = params.T;
  msg.textContent = `Generating ${ticksReq} ticks… 0s`;
  const timer = setInterval(() => {
    const s = Math.round((performance.now() - t0) / 1000);
    msg.textContent = `Generating ${ticksReq} ticks… ${s}s (working — please wait)`;
  }, 1000);

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    if (!resp.ok) {
      const txt = await resp.text();
      clearInterval(timer);
      msg.textContent = 'ERROR: ' + txt.split('\n')[0];
      msg.style.color = '#f88';
      btn.disabled = false;
      return;
    }
    DATA = await resp.json();
    clearInterval(timer);
    document.getElementById('leg-max').textContent = DATA.M;
    document.getElementById('tick').max = DATA.T - 1;
    simTick = 0; animMs = 0; packets = []; pulses = [];
    selectedAgent = null;
    // reset overlays that belong to the previous run
    trackMode = false; trackData = null; detailCache = { key: null, html: null };
    lastUniverseKey = null;
    document.getElementById('track-toggle').textContent = 'Track next new clause';
    document.getElementById('track-info').textContent = '';
    rebuildNetwork();
    updateNodes(0); updateAggStats(0); updateScatter(0); updateUniverse(0); updateAgentPanel();
    lastChartTick = -1; if (perfVisible) drawPerfCharts();
    // Stay paused at tick 0 after generating; the user presses Play to start.
    playing = false;
    lastTimestamp = null;
    document.getElementById('play').textContent = 'Play';
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    msg.textContent = `Done: ${DATA.T} ticks in ${dt}s — press Play`;
    msg.style.color = '#aaa';
  } catch (err) {
    clearInterval(timer);
    msg.textContent = 'ERROR: ' + err.message;
    msg.style.color = '#f88';
  } finally {
    btn.disabled = false;
  }
}

// Generate button: direct click handler (no form submit involved).
document.getElementById('generate').addEventListener('click', runGeneration);

// ============================================================
// Performance tab: time-series charts + saveable snapshots
// ============================================================
let perfVisible = false;
let lastChartTick = -1;
const SNAP_KEY = 'psm_snapshots_v1';
const SNAP_PALETTE = ['#ff9f43','#5cd4ff','#caa6ff','#7ee787','#ff7b9c','#f3d250','#8aa0ff','#48c9b0'];

// --- tab switching ---
function showTab(which) {
  perfVisible = (which === 'perf');
  document.getElementById('tab-live').style.display  = perfVisible ? 'none' : '';
  document.getElementById('tab-perf').style.display  = perfVisible ? '' : 'none';
  document.getElementById('tabbtn-live').classList.toggle('active', !perfVisible);
  document.getElementById('tabbtn-perf').classList.toggle('active', perfVisible);
  if (perfVisible) { lastChartTick = -1; drawPerfCharts(); renderSnapList(); }
}
document.getElementById('tabbtn-live').addEventListener('click', () => showTab('live'));
document.getElementById('tabbtn-perf').addEventListener('click', () => showTab('perf'));

// --- snapshot persistence (localStorage) ---
function loadSnaps() {
  try { return JSON.parse(localStorage.getItem(SNAP_KEY)) || []; }
  catch (e) { return []; }
}
function saveSnaps(arr) { localStorage.setItem(SNAP_KEY, JSON.stringify(arr)); }

function downsample(arr, maxPts) {
  if (arr.length <= maxPts) return arr.slice();
  const out = [], step = arr.length / maxPts;
  for (let i = 0; i < maxPts; i++) out.push(arr[Math.floor(i * step)]);
  return out;
}

function paramLabel(p) {
  let net = p.network;
  if (net === 'Random') net += ` p=${p.connect_prob}`;
  else if (net === 'Scale Free') net += ` m=${p.min_deg}`;
  else if (net === 'Small World') net += ` n=${p.n_size}`;
  else if (net === 'Hierarchical') net += ` L=${p.nlayers}`;
  else if (net === 'Modular') net += ` pin=${p.p_in}/pout=${p.p_out}`;
  const rc = p.redundant_comm ? ' · redundant-comm' : '';
  const nn = (p.normalize_comm === false) ? ' · no-norm' : '';
  return `${net} · N${p.N} K${p.K} α${p.alpha} obs${p.obs_prob} τ${p.clause_interval}${rc}${nn}`;
}

document.getElementById('snap-save').addEventListener('click', () => {
  if (!DATA || !DATA.series) return;
  const snaps = loadSnaps();
  const color = SNAP_PALETTE[snaps.length % SNAP_PALETTE.length];
  snaps.push({
    id: 'snap_' + DATA.run_id + '_' + snaps.length,
    label: paramLabel(DATA.params),
    color, T: DATA.T, M: DATA.M, on: true,
    avgV: downsample(DATA.series.avgV, 600),
    hom:  downsample(DATA.series.hom, 600),
  });
  saveSnaps(snaps);
  renderSnapList();
  drawPerfCharts();
  const msg = document.getElementById('snap-msg');
  msg.textContent = 'Saved ✓'; setTimeout(() => { msg.textContent = ''; }, 1500);
});

function renderSnapList() {
  const box = document.getElementById('snap-list');
  const snaps = loadSnaps();
  if (!snaps.length) { box.innerHTML = '<div class="snap-empty">No snapshots yet. Click "Save snapshot" to capture this run.</div>'; return; }
  box.innerHTML = snaps.map((s, i) => `
    <div class="snap-row">
      <input type="checkbox" data-i="${i}" ${s.on ? 'checked' : ''}>
      <span class="sw" style="background:${s.color}"></span>
      <span class="lbl">${s.label} <span style="color:#6c7884">(T=${s.T})</span></span>
      <button data-del="${i}">delete</button>
    </div>`).join('');
  box.querySelectorAll('input[type=checkbox]').forEach(cb => cb.addEventListener('change', e => {
    const arr = loadSnaps(); arr[+e.target.dataset.i].on = e.target.checked; saveSnaps(arr); drawPerfCharts();
  }));
  box.querySelectorAll('button[data-del]').forEach(b => b.addEventListener('click', e => {
    const arr = loadSnaps(); arr.splice(+e.target.dataset.del, 1); saveSnaps(arr); renderSnapList(); drawPerfCharts();
  }));
}

// --- chart drawing (raw SVG, no deps) ---
const SVGNS2 = 'http://www.w3.org/2000/svg';
function mkSvg(tag, at) { const e = document.createElementNS(SVGNS2, tag); for (const k in at) e.setAttribute(k, at[k]); return e; }

function drawLineChart(svg, datasets, yMin, yMax, xMax, currentTick, currentColor) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const W = 760, H = 320, mL = 48, mR = 12, mT = 12, mB = 34;
  const pW = W - mL - mR, pH = H - mT - mB;
  const xpx = t => mL + (xMax <= 1 ? 0 : t / xMax * pW);
  const ypx = v => mT + pH - (yMax <= yMin ? 0 : (v - yMin) / (yMax - yMin) * pH);
  // axes
  svg.appendChild(mkSvg('line', {x1: mL, y1: mT, x2: mL, y2: mT + pH, stroke: '#333'}));
  svg.appendChild(mkSvg('line', {x1: mL, y1: mT + pH, x2: mL + pW, y2: mT + pH, stroke: '#333'}));
  // y ticks (min, max)
  const yt0 = mkSvg('text', {x: mL - 5, y: mT + pH, fill: '#7d8794', 'font-size': 10, 'text-anchor': 'end'}); yt0.textContent = (+yMin).toFixed(yMax <= 1 ? 1 : 0); svg.appendChild(yt0);
  const yt1 = mkSvg('text', {x: mL - 5, y: mT + 8, fill: '#7d8794', 'font-size': 10, 'text-anchor': 'end'}); yt1.textContent = (+yMax).toFixed(yMax <= 1 ? 1 : 0); svg.appendChild(yt1);
  // x ticks (0, xMax)
  const xt0 = mkSvg('text', {x: mL, y: mT + pH + 14, fill: '#7d8794', 'font-size': 10, 'text-anchor': 'middle'}); xt0.textContent = '0'; svg.appendChild(xt0);
  const xt1 = mkSvg('text', {x: mL + pW, y: mT + pH + 14, fill: '#7d8794', 'font-size': 10, 'text-anchor': 'middle'}); xt1.textContent = String(xMax); svg.appendChild(xt1);
  const xlab = mkSvg('text', {x: mL + pW / 2, y: H - 4, fill: '#7d8794', 'font-size': 10, 'text-anchor': 'middle'}); xlab.textContent = 'tick'; svg.appendChild(xlab);
  // current-tick marker
  if (currentTick != null && currentTick >= 0) {
    svg.appendChild(mkSvg('line', {x1: xpx(currentTick), y1: mT, x2: xpx(currentTick), y2: mT + pH, stroke: '#ffffff', 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0.5}));
  }
  // dataset lines
  let legendY = mT + 4;
  datasets.forEach(ds => {
    const n = ds.values.length;
    if (!n) return;
    let d = '';
    for (let i = 0; i < n; i++) {
      const t = (n <= 1) ? 0 : i / (n - 1) * ds.T;   // map this series' index onto its own tick range
      d += (i === 0 ? 'M' : 'L') + xpx(t).toFixed(1) + ' ' + ypx(ds.values[i]).toFixed(1) + ' ';
    }
    svg.appendChild(mkSvg('path', {d, fill: 'none', stroke: ds.color, 'stroke-width': ds.width || 1.6, opacity: ds.opacity || 1}));
    // legend entry
    svg.appendChild(mkSvg('rect', {x: mL + 8, y: legendY, width: 10, height: 3, fill: ds.color}));
    const lt = mkSvg('text', {x: mL + 22, y: legendY + 4, fill: '#cdd3da', 'font-size': 9.5}); lt.textContent = ds.label; svg.appendChild(lt);
    legendY += 13;
  });
}

function drawPerfCharts() {
  if (!perfVisible || !DATA || !DATA.series) return;
  const snaps = loadSnaps().filter(s => s.on);
  // datasets for avg violations
  const avgSets = [{ label: 'current run', color: '#ffffff', width: 2, T: DATA.T, values: DATA.series.avgV }];
  const homSets = [{ label: 'current run', color: '#ffffff', width: 2, T: DATA.T, values: DATA.series.hom }];
  snaps.forEach(s => {
    avgSets.push({ label: s.label, color: s.color, T: s.T, values: s.avgV, opacity: 0.9 });
    homSets.push({ label: s.label, color: s.color, T: s.T, values: s.hom, opacity: 0.9 });
  });
  const xMax = Math.max(DATA.T, ...snaps.map(s => s.T));
  const yMaxV = Math.max(DATA.M, ...snaps.map(s => s.M));
  drawLineChart(document.getElementById('chart-avgV'), avgSets, 0, yMaxV, xMax, simTick, '#fff');
  drawLineChart(document.getElementById('chart-hom'),  homSets, 0.5, 1, xMax, simTick, '#fff');
  lastChartTick = simTick;
}

// Download a chart SVG as a PNG.
document.querySelectorAll('.chart-actions button').forEach(b => b.addEventListener('click', () => {
  const svg = document.getElementById(b.dataset.chart === 'avgV' ? 'chart-avgV' : 'chart-hom');
  const xml = new XMLSerializer().serializeToString(svg);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas'); c.width = 760 * 2; c.height = 320 * 2;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#07080a'; ctx.fillRect(0, 0, c.width, c.height);
    ctx.scale(2, 2); ctx.drawImage(img, 0, 0);
    const a = document.createElement('a');
    a.download = (b.dataset.chart === 'avgV' ? 'avg_violations' : 'homogeneity') + '.png';
    a.href = c.toDataURL('image/png'); a.click();
  };
  img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(xml)));
}));

// Drive the animation at ~60fps with a timer (idles until DATA arrives).
setInterval(frame, 1000 / 60);

// Auto-run once on load so the network is visible immediately, no click needed.
window.addEventListener('load', runGeneration);
</script>
</body>
</html>
"""


# ============================================================
# HTTP server
# ============================================================

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Quiet the default per-request log noise.
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            payload = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif path == '/d3.v7.min.js':
            # Serve the vendored D3 copy that sits next to this script, so the
            # viewer works offline / with no CDN dependency.
            d3_path = Path(__file__).resolve().parent / 'd3.v7.min.js'
            try:
                payload = d3_path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/javascript; charset=utf-8')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        elif path == '/state':
            # On-demand: one agent's variable vector + KB at a tick.
            q = parse_qs(urlparse(self.path).query)
            try:
                run_id = int(q['run'][0]); tick = int(q['tick'][0]); agent = int(q['agent'][0])
            except (KeyError, ValueError):
                self.send_response(400); self.end_headers(); return
            data = get_agent_state(run_id, tick, agent)
            if data is None:
                self.send_response(409); self.end_headers(); return   # stale/invalid
            self._send_json(data)
        elif path == '/track':
            # On-demand: discovery + propagation timeline of the next new clause.
            q = parse_qs(urlparse(self.path).query)
            try:
                run_id = int(q['run'][0]); after = int(q['after'][0])
            except (KeyError, ValueError):
                self.send_response(400); self.end_headers(); return
            data = get_track(run_id, after)
            if data is None:
                self.send_response(409); self.end_headers(); return
            self._send_json(data)
        else:
            self.send_response(404)
            self.end_headers()

    def _send_json(self, obj):
        payload = json.dumps(obj, separators=(',', ':')).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if urlparse(self.path).path != '/run':
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length).decode('utf-8')
            params = json.loads(raw)

            # Coerce all incoming values (they arrive as strings from the form).
            p = {
                'N': int(params['N']),
                'K': int(params['K']),
                'alpha': float(params['alpha']),
                'obs_prob': float(params['obs_prob']),
                'clause_interval': int(params['clause_interval']),
                'T': int(params['T']),
                'seed': int(params['seed']),
                'network': params['network'],
                'redundant_comm': bool(params.get('redundant_comm', False)),
                'normalize_comm': bool(params.get('normalize_comm', True)),
            }
            net = p['network']
            if net == 'Random':
                p['connect_prob'] = float(params['connect_prob'])
            elif net == 'Small World':
                p['n_size'] = int(params['n_size'])
                p['rewire_prob'] = float(params['rewire_prob'])
            elif net == 'Scale Free':
                p['min_deg'] = int(params['min_deg'])
            elif net == 'Hierarchical':
                p['nlayers'] = int(params['nlayers'])
                p['intra_layer_conn'] = float(params['intra_layer_conn'])
                p['inter_layer_conn'] = float(params['inter_layer_conn'])
            elif net == 'Modular':
                p['p_in'] = float(params['p_in'])
                p['p_out'] = float(params['p_out'])

            t0 = time.time()
            data = run_simulation(p)
            elapsed = time.time() - t0
            print(f"  run: N={p['N']}, T={p['T']}, network={p['network']}, "
                  f"seed={p['seed']}  ->  {elapsed:.1f}s",
                  flush=True)

            payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        except Exception as e:
            tb = traceback.format_exc()
            print("\n!!! Error during /run:\n" + tb, flush=True)
            payload = tb.encode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


class ReusableThreadingServer(socketserver.ThreadingTCPServer):
    """Threading + SO_REUSEADDR so rapid restarts don't fail and parallel
    requests (e.g. the browser keeping a connection open) don't lock the
    server."""
    allow_reuse_address = True
    daemon_threads = True


def main():
    port = 8765
    # Bind explicitly to 127.0.0.1 (IPv4 loopback) to avoid IPv6 quirks on
    # some Windows setups where 'localhost' resolves first to ::1.
    httpd = ReusableThreadingServer(('127.0.0.1', port), Handler)
    print(f"\nServer running at http://localhost:{port}", flush=True)
    print(f"Press Ctrl+C in this terminal to stop.\n", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()
