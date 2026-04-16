"""
Scale Free vs Random: parameter sweeps at T=1000.

For each parameter, we vary it along a short grid while holding everything
else at the baseline. For every cell we run 3 seeds on each of two networks
(Scale Free, Random-dense) and compute, at step 1000:

    gap_avg = SF_avg_V  - Random_avg_V
    gap_min = SF_min_V  - Random_min_V

Positive gap = Scale Free has more violations (worse) than Random.

Baseline:
    N = 80, K = 30, alpha = 2, obs_prob = 0.01, clause_interval = 10,
    XOR_prop = 0.5 (the current AND/XOR split), local-search passes = 1,
    Scale Free: min_deg = 3
    Random:    connect_prob = 0.20
    T = 1000, 3 seeds (42, 43, 44)

Sweeps:
    1. alpha                (problem hardness via density of constraints)
    2. xor_prop             (problem hardness via clause-type mix)
    3. obs_prob             (private-channel throughput)
    4. N                    (population size)
    5. density              (network density, matched pairs)
    6. clause_interval      (exogenous clause-change rate)
    7. local_search_passes  (how hard each agent optimises per arrival)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import pandas as pd
from model import ProblemSolvingModel

# ============================================================
# Baseline configuration
# ============================================================
BASE = dict(
    N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
    xor_prop=0.5, ls_passes=1,
)
SEEDS = [42, 43, 44]
T = 1000

SF_DEFAULT   = dict(type_network="Scale Free", min_deg=3)
RAND_DEFAULT = dict(type_network="Random",     connect_prob=0.20)


# ============================================================
# Subclass: configurable XOR proportion + optional extra LS passes
# ============================================================
class ConfigModel(ProblemSolvingModel):
    """Extends the model with two knobs set before __init__:
       - xor_prop:   probability a generated clause is XOR (rest are AND)
       - ls_passes:  how many times an agent runs local_update_around
                     over the *whole* KB after each new clause is learned.
    """
    def __init__(self, *args, xor_prop=0.5, ls_passes=1, **kwargs):
        self.xor_prop = xor_prop
        self.ls_passes = ls_passes
        super().__init__(*args, **kwargs)

    def random_clause(self):
        indices = random.sample(range(1, self.K + 1), 2)
        op = "XOR" if random.random() < self.xor_prop else "AND"
        return self.canonicalise_clause((op, tuple(indices)))


# ============================================================
# Runner
# ============================================================
def run_one(net_kw, seed, overrides):
    """Run a single (network, seed, param) configuration to step T; return
    (avg_V, min_V) at the end."""
    kwargs = {
        **{k: v for k, v in BASE.items() if k not in ("xor_prop", "ls_passes")},
        **net_kw,
        "setup_source": "generate",
        "R": T,
        "seed": seed,
    }
    kwargs.update(overrides.get("model", {}))

    model_kwargs = dict(
        xor_prop=overrides.get("xor_prop", BASE["xor_prop"]),
        ls_passes=overrides.get("ls_passes", BASE["ls_passes"]),
    )
    m = ConfigModel(**kwargs, **model_kwargs)

    # Extra LS passes (if any) wrap the standard step by re-running
    # local_update_around over a sample of the KB each tick. Cheap proxy
    # for "harder optimisation".
    extra_passes = model_kwargs["ls_passes"] - 1
    for _ in range(T):
        m.step()
        if extra_passes > 0 and m.agents:
            for a in m.agents:
                if not a.kb:
                    continue
                for _ in range(extra_passes):
                    cl = random.choice(a.kb)
                    a.local_update_around(cl)
    return m.avg_true_V, m.min_true_V


def sweep(param_name, values, overrides_for):
    """Run a sweep: one row per parameter value, two networks, multiple seeds.
    Returns a tidy DataFrame.
    """
    rows = []
    total = len(values) * 2 * len(SEEDS)
    i = 0
    for v in values:
        for net_label, net_kw_fn in [("Scale Free", sf_kw_for),
                                     ("Random",     rand_kw_for)]:
            net_kw = net_kw_fn(v, param_name)
            for seed in SEEDS:
                i += 1
                overrides = overrides_for(v)
                avgV, minV = run_one(net_kw, seed, overrides)
                rows.append({
                    param_name: v,
                    "network":   net_label,
                    "seed":      seed,
                    "avg_V":     avgV,
                    "min_V":     minV,
                })
                print(f"  [{i:3d}/{total}] {param_name}={v}  "
                      f"{net_label:<11} seed={seed}  "
                      f"avgV={avgV:6.2f}  minV={minV:6.2f}")
    return pd.DataFrame(rows)


def summarize(df, param):
    """Mean over seeds, wide by network, with gaps."""
    agg = (df.groupby([param, "network"])
             .agg(avg_V=("avg_V", "mean"),
                  min_V=("min_V", "mean"))
             .reset_index())
    sf  = agg[agg.network == "Scale Free"].set_index(param)
    rnd = agg[agg.network == "Random"].set_index(param)
    out = pd.DataFrame({
        "SF_avg":    sf["avg_V"],
        "Rand_avg":  rnd["avg_V"],
        "gap_avg":   sf["avg_V"] - rnd["avg_V"],
        "SF_min":    sf["min_V"],
        "Rand_min":  rnd["min_V"],
        "gap_min":   sf["min_V"] - rnd["min_V"],
    }).round(2)
    return out


# ============================================================
# Network-parameter constructors per sweep
# ============================================================
def sf_kw_for(value, param):
    kw = dict(SF_DEFAULT)
    if param == "density":
        kw["min_deg"] = value["sf_min_deg"]
    return kw

def rand_kw_for(value, param):
    kw = dict(RAND_DEFAULT)
    if param == "density":
        kw["connect_prob"] = value["rand_connect_prob"]
    return kw


# ============================================================
# Overrides per sweep
# ============================================================
def ovr_alpha(v):              return {"model": {"alpha": v}}
def ovr_xor_prop(v):           return {"xor_prop": v}
def ovr_obs_prob(v):           return {"model": {"obs_prob": v}}
def ovr_N(v):                  return {"model": {"N": v}}
def ovr_density(v):            return {}
def ovr_clause_interval(v):    return {"model": {"clause_interval": v}}
def ovr_ls_passes(v):          return {"ls_passes": v}


# ============================================================
# Main: run all sweeps and print tables
# ============================================================
if __name__ == "__main__":
    all_results = {}

    print("\n" + "=" * 70)
    print("SWEEP 1: alpha (constraint density)")
    print("=" * 70)
    df = sweep("alpha", [1, 2, 4, 8], ovr_alpha)
    all_results["alpha"] = summarize(df, "alpha")

    print("\n" + "=" * 70)
    print("SWEEP 2: xor_prop (clause-type mix)")
    print("=" * 70)
    df = sweep("xor_prop", [0.0, 0.25, 0.5, 0.75, 1.0], ovr_xor_prop)
    all_results["xor_prop"] = summarize(df, "xor_prop")

    print("\n" + "=" * 70)
    print("SWEEP 3: obs_prob (private-channel rate)")
    print("=" * 70)
    df = sweep("obs_prob", [0.001, 0.005, 0.01, 0.05, 0.1], ovr_obs_prob)
    all_results["obs_prob"] = summarize(df, "obs_prob")

    print("\n" + "=" * 70)
    print("SWEEP 4: N (population size)")
    print("=" * 70)
    df = sweep("N", [40, 80, 160, 320], ovr_N)
    all_results["N"] = summarize(df, "N")

    print("\n" + "=" * 70)
    print("SWEEP 5: density (matched Random/SF pairs)")
    print("=" * 70)
    density_pairs = [
        {"label": "low",       "rand_connect_prob": 0.10, "sf_min_deg": 2},
        {"label": "med-low",   "rand_connect_prob": 0.20, "sf_min_deg": 3},
        {"label": "med-high",  "rand_connect_prob": 0.30, "sf_min_deg": 5},
        {"label": "high",      "rand_connect_prob": 0.40, "sf_min_deg": 8},
    ]
    # sweep function can't take dicts as keys cleanly; inline it here
    rows = []
    total = len(density_pairs) * 2 * len(SEEDS)
    i = 0
    for pair in density_pairs:
        for net_label, net_kw in [
            ("Scale Free", dict(type_network="Scale Free", min_deg=pair["sf_min_deg"])),
            ("Random",     dict(type_network="Random",     connect_prob=pair["rand_connect_prob"])),
        ]:
            for seed in SEEDS:
                i += 1
                avgV, minV = run_one(net_kw, seed, {})
                rows.append({
                    "density": pair["label"],
                    "network": net_label,
                    "seed":    seed,
                    "avg_V":   avgV,
                    "min_V":   minV,
                })
                print(f"  [{i:3d}/{total}] density={pair['label']:<9} "
                      f"{net_label:<11} seed={seed}  "
                      f"avgV={avgV:6.2f}  minV={minV:6.2f}")
    df = pd.DataFrame(rows)
    # keep the ordered label set
    label_order = [p["label"] for p in density_pairs]
    df["density"] = pd.Categorical(df["density"], categories=label_order, ordered=True)
    all_results["density"] = summarize(df, "density")

    print("\n" + "=" * 70)
    print("SWEEP 6: clause_interval (exogenous change rate)")
    print("=" * 70)
    df = sweep("clause_interval", [1, 5, 10, 25, 100], ovr_clause_interval)
    all_results["clause_interval"] = summarize(df, "clause_interval")

    print("\n" + "=" * 70)
    print("SWEEP 7: local-search passes (optimisation intensity)")
    print("=" * 70)
    df = sweep("ls_passes", [1, 2, 5], ovr_ls_passes)
    all_results["ls_passes"] = summarize(df, "ls_passes")

    # -----------------------------------------
    # Print summary tables at the end
    # -----------------------------------------
    print("\n\n" + "#" * 70)
    print("# FINAL TABLES  (SF means Scale Free, Rand means Random dense)")
    print("# gap = SF - Random.  Positive gap = SF worse than Random.")
    print("#" * 70)

    for name, tbl in all_results.items():
        print(f"\n--- Sweep: {name} ---")
        print(tbl.to_string())

    # Also save CSVs
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    for name, tbl in all_results.items():
        tbl.to_csv(out_dir / f"sweep_{name}.csv")
    print(f"\nCSV outputs saved to {out_dir}")
