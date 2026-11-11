"""The no-negative-beliefs model with knowledge-base capacity 24 ("noneg24" in "261109 results runs.py"),
run through the runner's own main and no-communication batches: Regular, Scale Free and Scale Free dense,
seeds 1-40, T = 2000. Writes main.npz and nocomm.npz next to this file. Run from model/:
  PYTHONHASHSEED=0 python "output/261111 noneg24/runs_noneg24.py"
"""
import importlib.util, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, os.getcwd())
spec = importlib.util.spec_from_file_location("runner", Path.cwd() / "261109 results runs.py")
R = importlib.util.module_from_spec(spec); sys.modules["runner"] = R; spec.loader.exec_module(R)
_build = R.build
R.build = lambda name, seed, T, variant="full": _build(name, seed, T, "noneg24")   # every batch uses the variant
OUT = Path(__file__).resolve().parent
SEEDS = range(1, 41)

if __name__ == "__main__":
    jobs = [("main", n, s) for s in SEEDS for n in ("even", "uneven", "dense")] + [("nocomm", s) for s in SEEDS]
    t0 = time.time(); res = {"main": {}, "nocomm": {}}
    with ProcessPoolExecutor(8) as ex:
        for (kind, key, seed), out in ex.map(R.run, jobs):
            for q, v in out.items():
                res[kind][f"{key}|{seed}|{q}"] = v
    for kind in res:
        R.np.savez_compressed(OUT / f"{kind}.npz", **res[kind])
    print(f"wrote main.npz, nocomm.npz ({len(jobs)} runs, {time.time() - t0:.0f}s)")
