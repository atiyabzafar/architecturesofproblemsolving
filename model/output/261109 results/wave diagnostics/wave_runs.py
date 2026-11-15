"""Why does rho_-(tau) rise and fall (a wave) in the Regular network but not in Scale Free?

Instrumented baseline runs (same seeds and seeding as "261109 results runs.py"; the
instrumentation does not change any random draw, checked against main.npz). For every
constraint retired at t = 1000-1199 (first life only), by age tau since retirement and by
agent class (intake lambda = k_in/<k_in>: low <= 0.5, mid, high >= 1.5; same for out-degree):
  hold   agents holding a positive / negative belief
  ev     events: own test falsifies a positive (T_pn); message turns positive into negative
         (M_pn); message brings a negative to an agent without a belief (M_0n); message turns
         negative back into positive (M_np); message brings a positive to an agent without a
         belief (M_0p); own test re-confirms a negative (T_nn); eviction of a negative (E_n)
         and of a positive (E_p) belief
  first  first acquisition of a negative belief since retirement (reach)
  sendn / sendp  transmissions of the negative / positive belief by sender out-class
Per agent over t = 1000-1700: insertions of new beliefs, and the time since the last touch
of each evicted belief (the memory time of the knowledge base).

Networks: even, uneven, dense; uneven_eq (Scale Free links, every agent's intake set to 1);
uneven_same (Scale Free with out-degree = in-degree: those who hear most are heard most).
Run from model/:  PYTHONHASHSEED=0 python wave_runs.py OUTFILE
"""
import importlib.util
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("runner", Path.cwd() / "261109 results runs.py")
R = importlib.util.module_from_spec(spec); sys.modules["runner"] = R; spec.loader.exec_module(R)
L, A, np = R.L, R.A, R.np
A.SPECS["uneven_same"] = dict(A.SPECS["uneven"], out="same_sf")
_orig_build_ideal = A.build_ideal


def _build_same(spec, seed):
    """Scale Free in-degrees with out-degree = in-degree (those who hear most are heard most):
    random stub matching, repaired to a simple graph, redrawn until (weakly, hence strongly,
    since every node is balanced) connected."""
    nx = A.nx
    k = np.asarray(spec["kin"], int); n = len(k)
    for attempt in range(1000):
        rng = np.random.default_rng(seed * 7919 + 17 + 100003 * attempt)
        src = np.repeat(np.arange(n), k); dst = rng.permutation(np.repeat(np.arange(n), k))
        E = len(src)
        for _ in range(200 * E):
            bad = [e for e in range(E) if src[e] == dst[e]]
            pairs = {}
            for e in range(E):
                pairs.setdefault((src[e], dst[e]), []).append(e)
            bad += [e for v in pairs.values() for e in v[1:]]
            if not bad:
                break
            e = bad[0]; f = int(rng.integers(E))
            dst[e], dst[f] = dst[f], dst[e]
        G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(zip(src.tolist(), dst.tolist()))
        if G.number_of_edges() == E and nx.number_of_selfloops(G) == 0 and nx.is_strongly_connected(G):
            perm = rng.permutation(n)
            return nx.relabel_nodes(G, {i: int(perm[i]) for i in range(n)})
    raise RuntimeError("no realisation found")


A.build_ideal = lambda spec, seed: _build_same(spec, seed) if spec.get("out") == "same_sf" else _orig_build_ideal(spec, seed)

T, AGM = 1700, 500
COH0, COH1 = 1000, 1200
NEV = 8
T_PN, M_PN, M_0N, M_NP, M_0P, T_NN, E_N, E_P = range(NEV)

_prev_test = L.LearningByTestingPerson._one_test
_orig_set = L.LearningByTestingPerson.set_belief


def _one_test(self, C_set):
    self._ctx = "test"
    try:
        return _prev_test(self, C_set)
    finally:
        self._ctx = None


def _communicate_signed(self):                      # verbatim copy of the model's, plus logging context
    if not self.in_neighbors:
        return
    comm_scale = self.model.comm_scale
    for nbr_id in self.in_neighbors:
        edge_data = self.model.network[nbr_id][self.unique_id]
        link_weight = edge_data.get("weight", 1.0)
        if random.random() < min(1.0, comm_scale * link_weight):
            nbr = self.model.agent_list[nbr_id]
            if not nbr.order:
                continue
            if getattr(self.model, "recency", True):
                c = next((cc for cc in reversed(nbr.order) if self.sign.get(cc) != nbr.sign[cc]), None)
            else:
                diff = [cc for cc in nbr.order if self.sign.get(cc) != nbr.sign[cc]]
                c = random.choice(diff) if diff else None
            if c is not None:
                self._ctx, self._sender = "msg", nbr_id
                if self.set_belief(c, nbr.sign[c]) in ("new_pos", "neg2pos"):
                    self.local_update_around(c)
                self._ctx = None


def _enforce_cap(self):                             # verbatim logic of the model's, plus logging
    cap = self.belief_cap()
    m = self.model
    while len(self.order) > cap:
        old = self.order.pop(0)
        s = self.sign.pop(old, None)
        if s == 1:
            try:
                self.kb.remove(old)
            except ValueError:
                pass
        if m._measure:
            self._evn += 1; self._evlag += m.steps - self._touch.get(old, m.steps)
        if old in m._alive:
            age = m.steps - m._born[old]
            if age <= AGM:
                m._ev[age, E_N if s == -1 else E_P, self._cin] += 1


def set_belief(self, clause, s):
    m = self.model
    tr = _orig_set(self, clause, s)
    self._touch[clause] = m.steps
    if m._measure and tr in ("new_pos", "new_neg"):
        self._ins += 1
    if clause in m._alive:
        age = m.steps - m._born[clause]
        if age <= AGM:
            ctx = getattr(self, "_ctx", None)
            e = None
            if ctx == "test":
                e = T_PN if tr == "pos2neg" else T_NN if (tr == "confirm" and s == -1) else None
            elif ctx == "msg":
                e = {"pos2neg": M_PN, "new_neg": M_0N, "neg2pos": M_NP, "new_pos": M_0P}.get(tr)
                if e in (M_PN, M_0N):
                    m._sendn[age, m.agent_list[self._sender]._cout] += 1
                elif e in (M_NP, M_0P):
                    m._sendp[age, m.agent_list[self._sender]._cout] += 1
            if e is not None:
                m._ev[age, e, self._cin] += 1
            if s == -1 and self.unique_id not in m._ever[clause]:
                m._ever[clause].add(self.unique_id); m._first[age, self._cin] += 1
    return tr


L.LearningByTestingPerson._one_test = _one_test
L.LearningByTestingPerson._communicate_signed = _communicate_signed
L.LearningByTestingPerson._enforce_cap = _enforce_cap
L.LearningByTestingPerson.set_belief = set_belief
_prev_cap = L.LearningByTestingPerson.belief_cap
L.LearningByTestingPerson.belief_cap = lambda self: getattr(self, "_capi", None) or _prev_cap(self)
BASE_NET = {"uneven_eq": "uneven", "even_memspread": "even", "uneven_memeq": "uneven"}


def cls(lam):
    return 0 if lam <= 0.5 + 1e-9 else 2 if lam >= 1.5 - 1e-9 else 1


def run(job):
    name, seed = job
    m = R.build(BASE_NET.get(name, name), seed, T)
    n = m.N
    kin = np.array([m.network.in_degree(i) for i in range(n)], float)
    kout = np.array([m.network.out_degree(i) for i in range(n)], float)
    if name == "uneven_eq":                          # every agent pulls one message per period on average
        for u, v in m.network.edges():
            m.network[u][v]["weight"] = 1.0 / (kin[v] * m.comm_scale)
    for i, a in enumerate(m.agent_list):
        a._cin, a._cout = cls(kin[i] / kin.mean()), cls(kout[i] / kout.mean())
        a._touch = {c: 0 for c in a.order}; a._ins = a._evn = a._evlag = 0
    if name == "even_memspread":                     # Regular links and intake; memory capacities spread as in Scale Free
        perm = np.random.default_rng(seed).permutation(n)
        for r, i in enumerate(perm):
            a = m.agent_list[int(i)]; a._cin = 2 if r < 16 else 1 if r < 48 else 0
            a._capi = {2: 34, 1: 97, 0: 160}[a._cin]
    if name == "uneven_memeq":                       # Scale Free; capacities set so that every agent has about the same memory time
        for a in m.agent_list:
            a._capi = {2: 195, 1: 111, 0: 68}[a._cin]
    m._alive, m._born, m._ever, m._measure = set(), {}, {}, False
    m._ev = np.zeros((AGM + 1, NEV, 3)); m._first = np.zeros((AGM + 1, 3))
    m._sendn = np.zeros((AGM + 1, 3)); m._sendp = np.zeros((AGM + 1, 3))
    hold = np.zeros((AGM + 1, 2, 3)); hold_out = np.zeros((AGM + 1, 2, 3)); cacc = np.zeros(AGM + 1)
    avgV = np.zeros(T + 1); avgV[0] = m.avg_true_V
    born, seen = {}, 0
    for t in range(1, T + 1):
        m._measure = t > COH0
        m.step()
        avgV[t] = m.avg_true_V
        Cs = set(m.C); log = m.retire_log
        while seen < len(log):
            tick, old, new, gone = log[seen]; seen += 1
            if gone and COH0 <= tick < COH1 and old not in born:
                born[old] = tick
                m._ever[old] = {a.unique_id for a in m.agent_list if a.sign.get(old) == -1}
                for a in m.agent_list:
                    if a.sign.get(old) == -1:
                        m._first[0, a._cin] += 1
        for c in [c for c in born if born[c] is not None and c in Cs]:
            born[c] = None
        m._born = {c: b for c, b in born.items() if b is not None and t - b <= AGM}
        m._alive = set(m._born)
        for c in m._alive:
            age = t - m._born[c]; cacc[age] += 1
            for a in m.agent_list:
                sg = a.sign.get(c)
                if sg is not None:
                    k = 0 if sg == 1 else 1
                    hold[age, k, a._cin] += 1; hold_out[age, k, a._cout] += 1
    per = dict(kin=kin, kout=kout, ins=np.array([a._ins for a in m.agent_list]) / (T - COH0),
               evn=np.array([a._evn for a in m.agent_list]), evlag=np.array([a._evlag for a in m.agent_list]))
    ncl_in = np.bincount([a._cin for a in m.agent_list], minlength=3)
    ncl_out = np.bincount([a._cout for a in m.agent_list], minlength=3)
    return name, seed, dict(hold=hold, hold_out=hold_out, cacc=cacc, ev=m._ev, first=m._first, sendn=m._sendn,
                            sendp=m._sendp, avgV=avgV, ncl_in=ncl_in, ncl_out=ncl_out, **per)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    nets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["even", "uneven", "dense", "uneven_eq", "uneven_same"]
    jobs = [(nm, s) for s in range(1, 41) for nm in nets]
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, (nm, s, o) in enumerate(ex.map(run, jobs), 1):
            for q, v in o.items():
                res[f"{nm}|{s}|{q}"] = v
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out, **res)
    print("wrote", out, f"{time.time() - t0:.0f}s")
