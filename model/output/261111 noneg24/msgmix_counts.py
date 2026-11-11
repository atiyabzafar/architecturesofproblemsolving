"""What do messages carry, with and without negative beliefs? Periods 1501-2000, same seeds/seeding as the runner.
Every delivered message is classified by what it does to the receiver:
  cur_new  adds a current constraint the receiver did not hold        cur_flip  turns a negative into a positive, constraint current
  out_new  adds an outdated (retired) positive belief                 out_flip  turns a negative into a positive, constraint retired
  neg_ok   brings a correct negative belief (constraint absent)       neg_bad   brings a negative belief about a current constraint
and 'fired' counts link activations, 'empty' those with no disagreement."""
import importlib.util, os, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("runner", Path.cwd() / "261109 results runs.py")
R = importlib.util.module_from_spec(spec); sys.modules["runner"] = R; spec.loader.exec_module(R)
L, np = R.L, R.np
K = ["cur_new", "cur_flip", "out_new", "out_flip", "neg_ok", "neg_bad", "fired", "empty"]

def _communicate_signed(self):                      # verbatim copy of the model's, plus counting
    if not self.in_neighbors:
        return
    m = self.model; on = getattr(m, "_count", False)
    comm_scale = m.comm_scale
    for nbr_id in self.in_neighbors:
        edge_data = m.network[nbr_id][self.unique_id]
        link_weight = edge_data.get("weight", 1.0)
        if random.random() < min(1.0, comm_scale * link_weight):
            nbr = m.agent_list[nbr_id]
            if on: m._cnt["fired"] += 1
            if not nbr.order:
                if on: m._cnt["empty"] += 1
                continue
            if getattr(m, "recency", True):
                c = next((cc for cc in reversed(nbr.order) if self.sign.get(cc) != nbr.sign[cc]), None)
            else:
                diff = [cc for cc in nbr.order if self.sign.get(cc) != nbr.sign[cc]]
                c = random.choice(diff) if diff else None
            if c is None:
                if on: m._cnt["empty"] += 1
                continue
            if on:
                cur = c in m._C_set; s = nbr.sign[c]; had = self.sign.get(c)
                k = (("cur_new" if had is None else "cur_flip") if cur else ("out_new" if had is None else "out_flip")) if s == 1 \
                    else ("neg_ok" if not cur else "neg_bad")
                m._cnt[k] += 1
            if self.set_belief(c, nbr.sign[c]) in ("new_pos", "neg2pos"):
                self.local_update_around(c)

L.LearningByTestingPerson._communicate_signed = _communicate_signed

def run(job):
    n, v, s = job
    m = R.build(n, s, 2000, v); m._cnt = dict.fromkeys(K, 0); V = []
    for t in range(1, 2001):
        m._count = t > 1500
        m.step()
        if t > 1500: V.append(m.avg_true_V)
    return n, v, s, m._cnt, float(np.mean(V))

if __name__ == "__main__":
    jobs = [(n, v, s) for s in range(1, 41) for v in ("full", "noneg24") for n in ("uneven",)]
    ref = np.load("output/261109 results/ablation.npz")
    acc = {}
    with ProcessPoolExecutor(8) as ex:
        for n, v, s, c, V in ex.map(run, jobs):
            same = abs(V - ref[f"{n}|{v}|{s}|avgV"][1501:].mean()) < 1e-9
            a = acc.setdefault((n, v), dict.fromkeys(K, 0) | {"V": 0.0, "same": True})
            for k in K: a[k] += c[k]
            a["V"] += V / 40; a["same"] &= same
    import json
    json.dump({f"{n}|{v}": a for (n, v), a in acc.items()}, open(Path(__file__).resolve().parent / "messages_counts.json", "w"), indent=1, default=lambda o: o.item())
    print("wrote messages_counts.json")
