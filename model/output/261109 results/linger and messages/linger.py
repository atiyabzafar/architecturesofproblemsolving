"""For each constraint retired at t = 1000-1199 (first life), how many periods some agent still holds a
positive belief about it, and, for each agent holding it at retirement, how many periods it keeps it.
Same seeds and seeding as "261109 results runs.py"; runs extended to T = 3000 so long-lived beliefs are observed."""
import importlib.util, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("runner", Path.cwd() / "261109 results runs.py")
R = importlib.util.module_from_spec(spec); sys.modules["runner"] = R; spec.loader.exec_module(R)
np = R.np
T, C0, C1 = 3000, 1000, 1200

def run(job):
    name, v, seed = job
    m = R.build(name, seed, T, v)
    born, seen, cons, agents = {}, 0, {}, []
    for t in range(1, T + 1):
        m.step()
        Cs = set(m.C); log = m.retire_log
        while seen < len(log):
            tick, old, new, gone = log[seen]; seen += 1
            if gone and C0 <= tick < C1 and old not in born:
                born[old] = tick
                holders = {a.unique_id for a in m.agent_list if a.sign.get(old) == 1}
                cons[old] = dict(born=tick, last=0 if holders else None, end=None, open={i: None for i in holders})
        for c, d in cons.items():
            if d["end"] is not None:
                continue
            age = t - d["born"]
            if c in Cs:                                   # re-entered: first life over, observation ends
                d["end"] = age; continue
            hold = {a.unique_id for a in m.agent_list if a.sign.get(c) == 1}
            if hold:
                d["last"] = age
            for i, lost in d["open"].items():          # agents that held it at retirement: first age without it
                if lost is None and i not in hold:
                    d["open"][i] = age
    for c, d in cons.items():
        if d["end"] is None:
            d["end"] = T - d["born"]
    # per constraint: linger = last age with a believer + 1 (0 if none at retirement); censored if still believed at end
    lin = np.array([(0 if d["last"] is None else d["last"] + 1, d["end"],
                     d["last"] is not None and d["last"] >= d["end"] - 1) for d in cons.values()], float)
    ag = np.array([(lost if lost is not None else d["end"], lost is None) for d in cons.values() for lost in d["open"].values()], float)
    return name, v, seed, lin, ag

if __name__ == "__main__":
    out = Path(sys.argv[1])
    jobs = [(n, v, s) for s in range(1, 41) for v in ("full", "noneg120", "noneg", "noneg24") for n in ("even", "uneven")]
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(8) as ex:
        for i, (n, v, s, lin, ag) in enumerate(ex.map(run, jobs), 1):
            res[f"{n}|{v}|{s}|lin"] = lin; res[f"{n}|{v}|{s}|ag"] = ag
            if i % 40 == 0: print(f"  {i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(out, **res); print("wrote", out, f"{time.time()-t0:.0f}s")
