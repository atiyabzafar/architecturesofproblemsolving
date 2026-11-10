"""Simulations for Section 3 (Results) of the npj draft.

Same model, settings and network construction as "261001 analysis.py" (it is
imported for make_model). Every run lasts T = 2000 periods and uses seeds 1-40.
Three batches:

MAIN: all 7 networks (Regular, Scale Free, Scale Free dense and the four
  generators of the Supplementary Information), default model. It records:
  - each agent's violations, averaged over 20 windows of 50 periods (t = 1001-2000),
    to see who performs best and whether it persists;
  - each agent's beliefs, sampled every 10 periods over t = 1501-2000:
    current rules believed, outdated "rule exists" beliefs, "rule gone" beliefs
    (and how many of those are wrong), and the current rules the agent violates
    among those it believes and those it does not;
  - network averages per period: violations, best agent, homogeneity.

ABLATION (Section 3.3): Regular and Scale Free, seven versions of the belief process:
    full      the paper's model: "rule gone" beliefs, freshest-first sharing, memory 120
    norec     sharing picks a random disagreement instead of the freshest (recency off)
    noneg120  no "rule gone" beliefs: a falsified belief is deleted, nothing negative is
              stored or sent; memory 120
    noneg     as noneg120, with memory 60
    noneg24   as noneg120, with memory 24, about the number of "rule exists" beliefs the
              full model holds
    neither   recency off and no "rule gone" beliefs (memory 60)
    old       the earlier version of the model (model.py Person.step): passive observation
              with probability 0.01, no testing, positive beliefs only, first-in-first-out
              memory of 60, random sharing
  It records, per period, the network's violations and, every 10 periods, the
  average agent's true, outdated and "rule gone" beliefs. It also follows every
  rule retired at t = 1000-1199 for its first life, until it re-enters the rule
  set or the run ends, recording the share of agents that still believe it or
  disbelieve it at each age. At t = 2000 it records how long ago the rules
  behind each outdated belief were retired.

NOCOMM: no communication (comm_scale = 0): agents learn only by testing. Network
  violations per period.

The activation order of agents (Mesa's own random stream, unseeded in the
model) is seeded here, so these runs are reproducible.
Run from the model folder:
  PYTHONHASHSEED=0 python "261109 results runs.py" OUTDIR [main|abl|nocomm|all]
Writes main.npz, ablation.npz and nocomm.npz to OUTDIR.
"""
import importlib.util
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("analysis_261001", Path.cwd() / "261001 analysis.py")
A = importlib.util.module_from_spec(spec)
sys.modules["analysis_261001"] = A
spec.loader.exec_module(A)
import model_learningbytesting as L  # noqa: E402  (same module object A uses)

np = A.np
WORKERS = 8
NETS = ["even", "uneven", "dense", "random", "smallworld", "scalefree", "layered"]
VARIANTS = {   # recency, belief process, memory cap, learning by testing, observation probability
    "full": (True, "normal", None, True, None),
    "norec": (False, "normal", None, True, None),
    "noneg120": (True, "delete", 120, True, None),
    "noneg": (True, "delete", 60, True, None),
    "noneg24": (True, "delete", 24, True, None),
    "neither": (False, "delete", 60, True, None),
    "old": (False, "normal", None, False, 0.01),
}
T = 2000
SEEDS = range(1, 41)
COH0, COH1 = 1000, 1200

# --- variants of the belief process, without touching the model files
_orig_test = L.LearningByTestingPerson._one_test
_orig_cap = L.LearningByTestingPerson.belief_cap


def _one_test(self, C_set):
    if not getattr(self.model, "delete_mode", False):
        return _orig_test(self, C_set)
    # Same random draws as the original for the new-clause test.
    if random.random() < 0.5:
        cand = self.model.random_clause()
        tries = 0
        while cand in self.sign and tries < 10:
            cand = self.model.random_clause(); tries += 1
        if cand not in self.sign and cand in C_set:
            self.set_belief(cand, 1)
            self.local_update_around(cand)
    elif self.order:
        clause = random.choice(self.order)         # deletion design: a falsified belief is deleted
        if clause in C_set:
            self.set_belief(clause, 1)
        else:
            self.order.remove(clause)
            self.sign.pop(clause, None)
            if clause in self.kb:
                self.kb.remove(clause)


def _cap(self):
    c = getattr(self.model, "cap_override", None)
    return c if c is not None else _orig_cap(self)


L.LearningByTestingPerson._one_test = _one_test
L.LearningByTestingPerson.belief_cap = _cap


def build(name, seed, T, variant="full"):
    m = A.make_model(name, seed, T)
    m.random.seed(1_000_003 * seed + 17)          # agents' activation order (Mesa's stream)
    rec, mode, cap, lbt, obs = VARIANTS[variant]
    m.recency, m.cap_override, m.learning_by_testing = rec, cap, lbt
    m.delete_mode = mode == "delete"
    if obs is not None:
        m.obs_prob = obs
    return m


def run_main(name, seed):
    m = build(name, seed, T)
    n = m.N
    kin = np.array([m.network.in_degree(i) for i in range(n)], float)
    kout = np.array([m.network.out_degree(i) for i in range(n)], float)
    avgV = np.zeros(T + 1); minV = np.zeros(T + 1); hom = np.zeros(T + 1)
    avgV[0], minV[0], hom[0] = m.avg_true_V, m.min_true_V, m.homogeneity
    win = np.zeros((20, n))                           # per-agent V, 50-period windows, t=1001..2000
    Q = ("cov", "true", "stale", "neg", "wneg", "vb", "vu")
    f = {q: np.zeros(n) for q in Q}; ns = 0
    for t in range(1, T + 1):
        m.step()
        avgV[t], minV[t], hom[t] = m.avg_true_V, m.min_true_V, m.homogeneity
        if t > 1000:
            win[(t - 1001) // 50] += [a.true_violations for a in m.agent_list]
        if t > 1500 and t % 10 == 0:
            ns += 1
            C = m.C; Cs = set(C)
            for i, a in enumerate(m.agent_list):
                P = set(a.kb)
                bel = [c in P for c in C]
                viol = [a.clause_violated(c, a.x) for c in C]
                negs = [c for c, s in a.sign.items() if s == -1]
                tr = sum(1 for c in P if c in Cs)
                f["cov"][i] += sum(bel)
                f["true"][i] += tr
                f["stale"][i] += len(P) - tr
                f["neg"][i] += len(negs)
                f["wneg"][i] += sum(1 for c in negs if c in Cs)
                f["vb"][i] += sum(v for b, v in zip(bel, viol) if b)
                f["vu"][i] += sum(v for b, v in zip(bel, viol) if not b)
    out = {q: f[q] / ns for q in Q}
    out.update(kin=kin, kout=kout, win=win / 50, avgV=avgV, minV=minV, hom=hom)
    return ("main", name, seed), out


def run_abl(name, variant, seed):
    m = build(name, seed, T, variant)
    n = m.N
    avgV = np.zeros(T + 1); avgV[0] = m.avg_true_V
    S = T // 10 + 1
    ts_true = np.zeros(S); ts_stale = np.zeros(S); ts_neg = np.zeros(S)
    AG = T - COH0 + 1
    h_acc = np.zeros(AG); n_acc = np.zeros(AG); c_acc = np.zeros(AG)
    born = {}                                      # tracked rule -> tick it was retired (first life)
    last_gone = {}                                 # clause -> last tick it left the rule set
    seen_log = 0
    for t in range(1, T + 1):
        m.step()
        avgV[t] = m.avg_true_V
        Cs = set(m.C)
        log = m.retire_log
        while seen_log < len(log):
            tick, old, new, gone = log[seen_log]; seen_log += 1
            if gone:
                last_gone[old] = tick
                if COH0 <= tick < COH1 and old not in born:
                    born[old] = tick
        # first lives only: a tracked rule that re-enters the rule set is dropped for good
        for c in [c for c in born if c in Cs]:
            born[c] = None
        alive = {c for c, t0 in born.items() if t0 is not None}
        if alive:
            hb = dict.fromkeys(alive, 0); nb = dict.fromkeys(alive, 0)
            for a in m.agent_list:
                if not m.learning_by_testing:          # old model: positive beliefs only, in a.kb
                    for c in alive.intersection(a.kb):
                        hb[c] += 1
                    continue
                sg = a.sign
                for c in alive.intersection(sg):
                    if sg[c] == 1:
                        hb[c] += 1
                    else:
                        nb[c] += 1
            for c in alive:
                age = t - born[c]
                h_acc[age] += hb[c] / n; n_acc[age] += nb[c] / n; c_acc[age] += 1
        if t % 10 == 0:
            k = t // 10; tr = st = ng = 0
            for a in m.agent_list:
                x = sum(1 for c in a.kb if c in Cs)
                tr += x; st += len(a.kb) - x
                ng += sum(1 for s in a.sign.values() if s == -1)
            ts_true[k], ts_stale[k], ts_neg[k] = tr / n, st / n, ng / n
    ages = []                                      # how long ago each outdated belief's rule was retired
    Cs = set(m.C)
    for a in m.agent_list:
        ages += [T - last_gone[c] for c in a.kb if c not in Cs and c in last_gone]
    age_hist = np.bincount(np.array(ages, int), minlength=T + 1)[: T + 1] if ages else np.zeros(T + 1, int)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dict(coh_h=h_acc / c_acc, coh_n=n_acc / c_acc, coh_c=c_acc)
    out.update(avgV=avgV, ts_true=ts_true, ts_stale=ts_stale, ts_neg=ts_neg, age_hist=age_hist)
    return ("abl", f"{name}|{variant}", seed), out


def run_nocomm(seed):
    m = build("even", seed, T)
    m.comm_scale = 0.0                             # no link ever fires: agents learn only by testing
    avgV = np.zeros(T + 1); avgV[0] = m.avg_true_V
    for t in range(1, T + 1):
        m.step()
        avgV[t] = m.avg_true_V
    return ("nocomm", "none", seed), {"avgV": avgV}


def run(job):
    kind = job[0]
    return run_main(*job[1:]) if kind == "main" else run_abl(*job[1:]) if kind == "abl" else run_nocomm(*job[1:])


FILES = {"main": "main.npz", "abl": "ablation.npz", "nocomm": "nocomm.npz"}

if __name__ == "__main__":
    outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    jobs = []
    if which in ("all", "abl"):
        jobs += [("abl", nm, v, s) for s in SEEDS for v in VARIANTS for nm in ("even", "uneven")]
    if which in ("all", "nocomm"):
        jobs += [("nocomm", s) for s in SEEDS]
    if which in ("all", "main"):
        jobs += [("main", nm, s) for s in SEEDS for nm in NETS]
    t0 = time.time(); res = {k: {} for k in FILES}
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, ((kind, key, seed), out) in enumerate(ex.map(run, jobs), 1):
            for q, v in out.items():
                res[kind][f"{key}|{seed}|{q}"] = v
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    for kind, fname in FILES.items():
        if res[kind]:
            np.savez_compressed(outdir / fname, **res[kind]); print("wrote", outdir / fname, flush=True)
    print(f"done: {len(jobs)} runs on {WORKERS} workers, {time.time() - t0:.0f}s")
