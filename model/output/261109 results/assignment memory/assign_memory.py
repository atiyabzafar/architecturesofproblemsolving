"""Does an agent's assignment remember constraints that are no longer in its knowledge base?
Every 10 periods over t = 1501-2000, each constraint is classified relative to each agent, and the share
violated by the agent's assignment is computed (average of the AND and XOR shares, so categories with a
different mix of operators are comparable):
  current constraints:  believed now | held as positive earlier but not now (by periods since it was lost) | never held
  retired constraints:  outdated belief held now | held earlier but not now (by periods since lost)
  reference:            random constraints never held and not in C (drawn with a separate generator)
Same seeds and seeding as "261109 results runs.py"; the simulation's own random stream is untouched."""
import importlib.util, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("runner", Path.cwd() / "261109 results runs.py")
R = importlib.util.module_from_spec(spec); sys.modules["runner"] = R; spec.loader.exec_module(R)
L, np = R.L, R.np
BINS = [(0, 50), (51, 150), (151, 400), (401, 10 ** 9)]
CATS = ["cur_believed", "cur_never"] + [f"cur_lost_{a}" for a, b in BINS] + ["old_believed"] + [f"old_lost_{a}" for a, b in BINS] + ["random"]
_prev_step = L.LearningByTestingPerson.step

def step(self):
    before = set(self.kb)
    _prev_step(self)
    lost = before - set(self.kb)
    if lost:
        la = self.__dict__.setdefault("_lost_at", {})
        for c in lost:
            la[c] = self.model.steps
L.LearningByTestingPerson.step = step

def run(job):
    n, v, s = job
    m = R.build(n, s, 2000, v); rng = np.random.default_rng(10_000 + s)
    viol = {c: np.zeros(2) for c in CATS}; cnt = {c: np.zeros(2) for c in CATS}
    V = []
    for t in range(1, 2001):
        m.step()
        if t > 1500: V.append(m.avg_true_V)
        if t > 1500 and t % 10 == 0:
            Cs = set(m.C)
            for a in m.agent_list:
                kb = set(a.kb); la = getattr(a, "_lost_at", {})
                def add(cat, c):
                    o = 0 if c[0] == "AND" else 1
                    viol[cat][o] += a.clause_violated(c, a.x); cnt[cat][o] += 1
                def lostcat(prefix, c):
                    d = t - la[c]
                    for lo, hi in BINS:
                        if lo <= d <= hi: return f"{prefix}_lost_{lo}"
                for c in Cs:
                    add("cur_believed" if c in kb else lostcat("cur", c) if c in la else "cur_never", c)
                for c in kb - Cs:
                    add("old_believed", c)
                for c in la:
                    if c not in Cs and c not in kb:
                        add(lostcat("old", c), c)
                k = 0
                while k < 60:
                    i, j = sorted(rng.choice(np.arange(1, 31), 2, replace=False).tolist())
                    c = ("AND" if rng.random() < 0.5 else "XOR", (i, j))
                    if c in Cs or c in kb or c in la: continue
                    add("random", c); k += 1
    return n, v, s, {c: (viol[c], cnt[c]) for c in CATS}, float(np.mean(V))

if __name__ == "__main__":
    jobs = [(n, v, s) for s in range(1, 11) for v in ("full", "noneg24") for n in ("even", "uneven")]
    ref = np.load("output/261109 results/ablation.npz"); acc = {}
    with ProcessPoolExecutor(8) as ex:
        for n, v, s, res, V in ex.map(run, jobs):
            a = acc.setdefault((n, v), {c: [np.zeros(2), np.zeros(2)] for c in CATS} | {"same": True})
            a["same"] &= abs(V - ref[f"{n}|{v}|{s}|avgV"][1501:].mean()) < 1e-9
            for c in CATS: a[c][0] += res[c][0]; a[c][1] += res[c][1]
    for (n, v), a in sorted(acc.items()):
        print(f"{n} {v} (trajectories identical to stored runs: {a['same']})")
        tot = sum(a[c][1].sum() for c in CATS if c.startswith("cur"))
        for c in CATS:
            vv, cc = a[c]
            if cc.sum() == 0: continue
            rate = np.mean(vv / np.maximum(cc, 1))
            extra = f" | share of current constraints {cc.sum()/tot:.0%}" if c.startswith("cur") else f" | per agent-sample {cc.sum()/(100*50*10):.1f}"
            print(f"   {c:16s} violated {rate:.0%}{extra}")
