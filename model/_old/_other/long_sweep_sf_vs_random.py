"""
Long-run parameter sweep: Scale Free vs Random networks.

For each of six parameters we vary its value along a grid while holding
everything else at the baseline, run T=10000 ticks per configuration with
5 random seeds, and report the average and minimum violation gap between
Scale Free and Random at the final tick.

What "gap" means:
    gap_avg = (avg violations on Scale Free) - (avg violations on Random)
    gap_min = (min violations on Scale Free) - (min violations on Random)
    Positive gap -> Scale Free is worse (more unsatisfied clauses).
    Negative gap -> Scale Free is better.

Outputs (all in output/long sweep/):
    long_sweep_results.xlsx             -- single spreadsheet containing:
                                             * one tab per parameter sweep
                                             * an "all_summaries" tab
                                             * a "raw" tab with every single run
                                             * ts_default  -- time-series tab at baseline
                                             * ts_combined -- time-series tab with
                                                 alpha=8, xor_prop=1, obs_prob=0.01,
                                                 N=300, density=med-high,
                                                 clause_interval=10
    sweep_<param>_raw.png               -- raw violations vs parameter (SF vs Random)
    sweep_<param>_gap.png               -- SF-Random gap vs parameter
    overview_avg.png                    -- 6-panel summary of raw averages
    overview_gap.png                    -- 6-panel summary of all gaps
    timeseries_default_avg.png          -- avg violations over time, baseline
    timeseries_default_min.png          -- min violations over time, baseline
    timeseries_combined_avg.png         -- avg violations over time, combined config
    timeseries_combined_min.png         -- min violations over time, combined config
    timeseries_low_comm_avg.png         -- avg violations over time, low-comm (rate=0.1)
    timeseries_low_comm_min.png         -- min violations over time, low-comm (rate=0.1)

Error bands are the standard error of the mean (std / sqrt(n_seeds)).

This will take a while to run.
"""

# ============================================================
# Step 1: Imports and configuration
# ============================================================

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Locate model.py one level up (this script lives in _other/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ProblemSolvingModel


# ---- time horizon and seeds --------------------------------
T     = 10_000                        # simulation length (ticks)
SEEDS = [42, 43, 44, 45, 46]          # five seeds per cell


# ---- baseline parameters (all sweeps hold everything else at these) ----
BASE = dict(
    N=80,                 # number of agents
    K=80,                 # number of boolean variables
    alpha=2,              # ratio of clauses to variables (M = alpha*K)
    obs_prob=0.01,        # prob an agent observes the environment this tick
    clause_interval=10,   # ticks between replacements of one universal clause
    xor_prop=0.5,         # fraction of clauses that are XOR (rest are AND)
)

# ---- default network settings for each network type --------
SF_DEFAULT   = dict(type_network="Scale Free", min_deg=3)
RAND_DEFAULT = dict(type_network="Random",     connect_prob=0.20)


# ============================================================
# Step 2: Model subclass that exposes the xor_prop knob
# ============================================================
# The stock ProblemSolvingModel hardcodes a 50/50 AND/XOR split when
# generating clauses. To sweep xor_prop we subclass and override just
# the clause generator. Everything else is inherited unchanged.
#
# We store xor_prop BEFORE calling super().__init__ because the parent
# constructor calls generate_clauses() during setup, which in turn
# calls random_clause().
# ============================================================

class ConfigModel(ProblemSolvingModel):
    """ProblemSolvingModel with two extra knobs:
       - xor_prop:  fraction of generated clauses that are XOR (default 0.5)
       - comm_rate: multiplicative scaling on self.comm_scale (default 1.0).
                    The stock model normalises so the average per-agent
                    successful elicitation rate is ~1/tick. Setting
                    comm_rate=0.1 rescales that average to ~0.1/tick.
    """

    def __init__(self, *args, xor_prop=0.5, comm_rate=1.0, **kwargs):
        self.xor_prop = xor_prop
        super().__init__(*args, **kwargs)
        # super().__init__ -> setup_network -> compute_comm_scale, which
        # has just set self.comm_scale = 1 / avg_in_strength. Rescale it.
        self.comm_scale *= comm_rate

    def random_clause(self):
        """Pick 2 distinct variable indices; emit XOR with prob xor_prop, AND otherwise."""
        indices = random.sample(range(1, self.K + 1), 2)
        op = "XOR" if random.random() < self.xor_prop else "AND"
        return self.canonicalise_clause((op, tuple(indices)))


# ============================================================
# Step 3: Run a single configuration to completion
# ============================================================
# One call = one (parameter value, network, seed) triple taken to tick T.
# Returns the final-tick average and minimum violations across agents.
# ============================================================

def run_one(network_kwargs, seed, overrides):
    """Build a ConfigModel, step it T times, return (avg_V, min_V) at tick T.

    Args:
        network_kwargs : dict of network-topology args, e.g. {"type_network": "Random", "connect_prob": 0.2}
        seed           : integer RNG seed for this run
        overrides      : dict that may contain:
                            "model":     dict of kwargs to pass to ProblemSolvingModel
                            "xor_prop":  float, the XOR fraction for this run
                            "comm_rate": float, multiplicative scaling on comm_scale
                                         (default 1.0; <1 means agents communicate less)
    """
    # Start from the baseline, drop the subclass-only knob, let overrides win.
    kwargs = {k: v for k, v in BASE.items() if k != "xor_prop"}
    kwargs.update(network_kwargs)
    kwargs.update({
        "setup_source": "generate",   # generate the universal clause set from scratch
        "R": T,                       # total expected rounds (used internally)
        "seed": seed,
    })
    kwargs.update(overrides.get("model", {}))   # per-sweep model-arg overrides

    xor_prop  = overrides.get("xor_prop",  BASE["xor_prop"])
    comm_rate = overrides.get("comm_rate", 1.0)

    m = ConfigModel(**kwargs, xor_prop=xor_prop, comm_rate=comm_rate)
    for _ in range(T):
        m.step()

    return m.avg_true_V, m.min_true_V


# ============================================================
# Step 4: Generic sweep runner
# ============================================================
# Given a parameter name, its values, and an "overrides" builder, run every
# (value, network, seed) triple and collect raw results.
# ============================================================

def run_sweep(param_name, param_values, overrides_for, network_kw_for, label=None):
    """Run a sweep along one parameter and return a long-format DataFrame.

    One row per (parameter value, network, seed).
    """
    label = label or param_name
    total = len(param_values) * 2 * len(SEEDS)
    rows = []
    i = 0

    print(f"\n=== Sweep: {label} ===", flush=True)
    t0 = time.time()

    for v in param_values:
        for net_label, net_kw in [("Scale Free", network_kw_for(v, "SF")),
                                  ("Random",    network_kw_for(v, "Random"))]:
            for seed in SEEDS:
                i += 1
                overrides = overrides_for(v)
                avg_V, min_V = run_one(net_kw, seed, overrides)
                rows.append({
                    param_name: _display(v),
                    "network":  net_label,
                    "seed":     seed,
                    "avg_V":    avg_V,
                    "min_V":    min_V,
                })
                elapsed = time.time() - t0
                eta     = elapsed / i * (total - i)
                print(f"  [{i:3d}/{total}] {label}={str(_display(v)):<10}  "
                      f"{net_label:<11}  seed={seed}  "
                      f"avgV={avg_V:6.2f}  minV={min_V:6.2f}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)",
                      flush=True)

    return pd.DataFrame(rows)


def _display(v):
    """Printable form for either a scalar or a density-pair dict."""
    return v["label"] if isinstance(v, dict) else v


# ============================================================
# Step 5: Summarise a sweep DataFrame across seeds
# ============================================================
# Collapse the 5 seeds to mean / std / SE for each (param, network), then
# reshape so the two networks sit side-by-side with gaps computed.
#
# Columns in the output:
#   <NET>_avg      : mean final avg violations across seeds
#   <NET>_avg_std  : across-seed standard deviation
#   <NET>_avg_sem  : standard error of the mean = std / sqrt(n_seeds)
#   <NET>_min      : mean final min violations across seeds
#   <NET>_min_std  : across-seed standard deviation
#   <NET>_min_sem  : standard error of the mean
#   gap_avg, gap_min : SF - Random (positive = SF worse)
# ============================================================

def summarize(df, param_name):
    """Return a wide-form summary with SF/Random means, stds, SEs, and gaps."""
    n_seeds = len(SEEDS)
    sqrt_n  = np.sqrt(n_seeds)

    agg = (df.groupby([param_name, "network"])
             .agg(avg_V_mean=("avg_V", "mean"),
                  avg_V_std =("avg_V", "std"),
                  min_V_mean=("min_V", "mean"),
                  min_V_std =("min_V", "std"))
             .reset_index())

    sf  = agg[agg.network == "Scale Free"].set_index(param_name)
    rnd = agg[agg.network == "Random"   ].set_index(param_name)

    # Preserve order (Categorical from the caller, otherwise numeric)
    index_order = df[param_name].drop_duplicates().tolist()

    out = pd.DataFrame({
        "SF_avg":       sf["avg_V_mean"],
        "SF_avg_std":   sf["avg_V_std"],
        "SF_avg_sem":   sf["avg_V_std"] / sqrt_n,
        "Rand_avg":     rnd["avg_V_mean"],
        "Rand_avg_std": rnd["avg_V_std"],
        "Rand_avg_sem": rnd["avg_V_std"] / sqrt_n,
        "gap_avg":      sf["avg_V_mean"] - rnd["avg_V_mean"],
        "SF_min":       sf["min_V_mean"],
        "SF_min_std":   sf["min_V_std"],
        "SF_min_sem":   sf["min_V_std"] / sqrt_n,
        "Rand_min":     rnd["min_V_mean"],
        "Rand_min_std": rnd["min_V_std"],
        "Rand_min_sem": rnd["min_V_std"] / sqrt_n,
        "gap_min":      sf["min_V_mean"] - rnd["min_V_mean"],
    }).round(4)

    out = out.loc[index_order]  # keep the original parameter ordering
    return out


# ============================================================
# Step 6: Per-sweep override / network-kw builders
# ============================================================
# For most sweeps the override just goes into model kwargs. For xor_prop
# it sets the subclass-only attribute. For the density sweep the network
# kwargs themselves depend on the sweep value.
# ============================================================

# ---- model-kwargs overrides per sweep ----------------------
def ovr_alpha(v):           return {"model": {"alpha": v}}
def ovr_xor_prop(v):        return {"xor_prop": v}
def ovr_obs_prob(v):        return {"model": {"obs_prob": v}}
def ovr_N(v):               return {"model": {"N": v}}
def ovr_density(v):         return {}   # density is expressed via network kwargs
def ovr_clause_interval(v): return {"model": {"clause_interval": v}}


# ---- network kwargs per sweep ------------------------------
def net_kw_default(v, which):
    """Default: Scale Free / Random at their baseline densities."""
    return dict(SF_DEFAULT) if which == "SF" else dict(RAND_DEFAULT)


def net_kw_density(v, which):
    """Density sweep: the value itself encodes both SF min_deg and Random p."""
    if which == "SF":
        return dict(type_network="Scale Free", min_deg=v["sf_min_deg"])
    else:
        return dict(type_network="Random", connect_prob=v["rand_connect_prob"])


# ============================================================
# Step 7: Plotting
# ============================================================
# For each sweep we save TWO independent files (each a single-panel figure):
#   sweep_<param>_raw.png  -- SF_avg and Rand_avg vs parameter, +/-1 SE shaded
#   sweep_<param>_gap.png  -- gap_avg and gap_min vs parameter, with zero line
# We also save TWO 6-panel overviews (one for averages, one for gaps).
# SE bands = mean +/- standard error (std / sqrt(n_seeds)).
#
# Colour conventions (consistent across raw / gap / overview plots):
#   Averages: Scale Free = blue (C0),    Random = orange (C1)
#   Gaps:     gap_avg    = tab:purple,   gap_min = tab:green
# ============================================================

# Color constants so averages and gaps are visually distinct everywhere.
COLOR_SF       = "tab:blue"
COLOR_RAND     = "tab:orange"
COLOR_GAP_AVG  = "tab:purple"
COLOR_GAP_MIN  = "tab:green"

def _xaxis_setup(ax, summary):
    """Shared X-axis handling: numeric on a linear axis, categorical by index.
    Categorical labels (e.g. density: low/med-low/med-high/high) are rendered
    horizontally."""
    x_raw = summary.index.tolist()
    numeric = all(isinstance(x, (int, float, np.integer, np.floating)) for x in x_raw)
    x = x_raw if numeric else list(range(len(x_raw)))
    if not numeric:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x_raw], rotation=0)
    return x


def plot_sweep_raw(summary, param_name, outfile, title=None):
    """Save a single-panel plot: raw SF_avg and Rand_avg vs parameter,
    with +/- 1 SE shaded around each line."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = _xaxis_setup(ax, summary)

    # Scale Free (blue)
    ax.plot(x, summary["SF_avg"], marker="o", color=COLOR_SF, label="Scale Free")
    ax.fill_between(x,
                    summary["SF_avg"] - summary["SF_avg_sem"],
                    summary["SF_avg"] + summary["SF_avg_sem"],
                    alpha=0.2, color=COLOR_SF)

    # Random (orange)
    ax.plot(x, summary["Rand_avg"], marker="s", color=COLOR_RAND, label="Random")
    ax.fill_between(x,
                    summary["Rand_avg"] - summary["Rand_avg_sem"],
                    summary["Rand_avg"] + summary["Rand_avg_sem"],
                    alpha=0.2, color=COLOR_RAND)

    ax.set_xlabel(param_name)
    ax.set_ylabel(f"avg violations at T={T}")
    ax.set_title(title or f"Raw avg violations vs {param_name}  (shaded = +/- 1 SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_sweep_gap(summary, param_name, outfile, title=None):
    """Save a single-panel plot: gap_avg and gap_min vs parameter, with zero
    reference line. Gap = SF - Random (positive = SF worse).
    Uses distinct colours (purple/green) to separate visually from raw plots."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = _xaxis_setup(ax, summary)

    ax.axhline(0, color="k", lw=0.5)
    ax.plot(x, summary["gap_avg"], marker="o", color=COLOR_GAP_AVG,
            label="gap_avg (SF - Random)")
    ax.plot(x, summary["gap_min"], marker="s", color=COLOR_GAP_MIN,
            label="gap_min (SF - Random)")

    ax.set_xlabel(param_name)
    ax.set_ylabel("violation gap")
    ax.set_title(title or f"SF - Random gap vs {param_name}  (positive = SF worse)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_overview_avg(summaries, outfile):
    """6-panel overview: one subplot per sweep, showing raw SF_avg and Rand_avg
    with SE shading. Colours match the per-sweep raw plots (blue / orange)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (pname, summary) in zip(axes.flatten(), summaries.items()):
        x = _xaxis_setup(ax, summary)

        # Scale Free
        ax.plot(x, summary["SF_avg"], marker="o", color=COLOR_SF, label="Scale Free")
        ax.fill_between(x,
                        summary["SF_avg"] - summary["SF_avg_sem"],
                        summary["SF_avg"] + summary["SF_avg_sem"],
                        alpha=0.2, color=COLOR_SF)

        # Random
        ax.plot(x, summary["Rand_avg"], marker="s", color=COLOR_RAND, label="Random")
        ax.fill_between(x,
                        summary["Rand_avg"] - summary["Rand_avg_sem"],
                        summary["Rand_avg"] + summary["Rand_avg_sem"],
                        alpha=0.2, color=COLOR_RAND)

        ax.set_title(pname)
        ax.set_xlabel(pname)
        ax.set_ylabel("avg violations")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Raw avg violations across all six sweeps  "
                 f"(T={T}, {len(SEEDS)} seeds, shaded = +/- 1 SE)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_overview_gap(summaries, outfile):
    """6-panel overview: one subplot per sweep, showing the gaps only.
    Colours match the per-sweep gap plots (purple / green)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (pname, summary) in zip(axes.flatten(), summaries.items()):
        x = _xaxis_setup(ax, summary)
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(x, summary["gap_avg"], marker="o", color=COLOR_GAP_AVG, label="gap_avg")
        ax.plot(x, summary["gap_min"], marker="s", color=COLOR_GAP_MIN, label="gap_min")
        ax.set_title(pname)
        ax.set_xlabel(pname)
        ax.set_ylabel("SF - Random")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Scale Free - Random violation gaps (T={T}, {len(SEEDS)} seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


# ============================================================
# Step 7b: Time-series machinery
# ============================================================
# For a handful of named configurations we want the full trajectory of
# avg violations and min violations over all T ticks (not just the final
# snapshot). We run each (config, network) pair across all seeds, then
# report the per-tick mean and standard error of the mean across seeds.
# ============================================================

def run_one_timeseries(network_kwargs, seed, overrides):
    """Like run_one, but records (avg_V, min_V) at every tick.

    Returns two length-T numpy arrays."""
    kwargs = {k: v for k, v in BASE.items() if k != "xor_prop"}
    kwargs.update(network_kwargs)
    kwargs.update({"setup_source": "generate", "R": T, "seed": seed})
    kwargs.update(overrides.get("model", {}))
    xor_prop  = overrides.get("xor_prop",  BASE["xor_prop"])
    comm_rate = overrides.get("comm_rate", 1.0)

    m = ConfigModel(**kwargs, xor_prop=xor_prop, comm_rate=comm_rate)
    avg_hist = np.zeros(T, dtype=float)
    min_hist = np.zeros(T, dtype=float)
    for t in range(T):
        m.step()
        avg_hist[t] = m.avg_true_V
        min_hist[t] = m.min_true_V
    return avg_hist, min_hist


def run_config_timeseries(config_name, sf_network_kw, rand_network_kw, overrides):
    """Run SEEDS seeds on both SF and Random for one config, and return a
    DataFrame with per-tick means and SEs across seeds."""
    n_seeds = len(SEEDS)
    sqrt_n  = np.sqrt(n_seeds)

    # Buffers: (n_seeds, T) matrices -- one row per seed.
    sf_avg = np.zeros((n_seeds, T))
    sf_min = np.zeros((n_seeds, T))
    rd_avg = np.zeros((n_seeds, T))
    rd_min = np.zeros((n_seeds, T))

    print(f"\n=== Time series: {config_name} ===", flush=True)
    t0 = time.time()

    for i, seed in enumerate(SEEDS):
        # Scale Free run
        sf_avg[i], sf_min[i] = run_one_timeseries(sf_network_kw, seed, overrides)
        # Random run
        rd_avg[i], rd_min[i] = run_one_timeseries(rand_network_kw, seed, overrides)
        elapsed = (time.time() - t0) / 60
        eta = elapsed / (i + 1) * (n_seeds - i - 1)
        print(f"  seed {seed}: SF final avgV={sf_avg[i, -1]:.2f}  "
              f"Rand final avgV={rd_avg[i, -1]:.2f}  "
              f"(elapsed {elapsed:.1f}m, eta {eta:.1f}m)", flush=True)

    df = pd.DataFrame({
        "tick":          np.arange(1, T + 1),
        "SF_avg_mean":   sf_avg.mean(axis=0),
        "SF_avg_sem":    sf_avg.std(axis=0, ddof=1) / sqrt_n,
        "Rand_avg_mean": rd_avg.mean(axis=0),
        "Rand_avg_sem":  rd_avg.std(axis=0, ddof=1) / sqrt_n,
        "SF_min_mean":   sf_min.mean(axis=0),
        "SF_min_sem":    sf_min.std(axis=0, ddof=1) / sqrt_n,
        "Rand_min_mean": rd_min.mean(axis=0),
        "Rand_min_sem":  rd_min.std(axis=0, ddof=1) / sqrt_n,
    })
    return df


def plot_timeseries_one(ts_df, metric_prefix, config_name, outfile, ylabel):
    """Plot one metric (avg or min) over time for SF and Random, with SE
    shaded bands. `metric_prefix` is either 'avg' or 'min'."""
    t = ts_df["tick"].values
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    # Scale Free (blue)
    mean = ts_df[f"SF_{metric_prefix}_mean"].values
    sem  = ts_df[f"SF_{metric_prefix}_sem"].values
    ax.plot(t, mean, label="Scale Free", color=COLOR_SF)
    ax.fill_between(t, mean - sem, mean + sem, alpha=0.25, color=COLOR_SF)

    # Random (orange)
    mean = ts_df[f"Rand_{metric_prefix}_mean"].values
    sem  = ts_df[f"Rand_{metric_prefix}_sem"].values
    ax.plot(t, mean, label="Random", color=COLOR_RAND)
    ax.fill_between(t, mean - sem, mean + sem, alpha=0.25, color=COLOR_RAND)

    ax.set_xlabel("tick")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{config_name}: {ylabel} over time  (shaded = +/- 1 SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


# ============================================================
# Step 8: Main -- runs all six sweeps, saves CSV+xlsx+plots
# ============================================================

def main():
    # ---- 8a. Prepare output folders --------------------------
    # Everything -- the xlsx AND all PNG plots -- goes straight into
    # output/long sweep/. No "plots" subfolder.
    out_dir   = Path(__file__).parent / "output" / "long sweep"
    plots_dir = out_dir                                 # plots live alongside the xlsx
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved under: {out_dir}", flush=True)

    # ---- 8b. Define the six parameter sweeps ----------------
    # Each entry: (name, values, overrides-builder, network-kw-builder)
    sweeps = [
        ("alpha",
         [1, 2, 4, 8],
         ovr_alpha,
         net_kw_default),

        ("xor_prop",
         [0.0, 0.25, 0.5, 0.75, 1.0],
         ovr_xor_prop,
         net_kw_default),

        ("obs_prob",
         [0.001, 0.005, 0.01, 0.05, 0.1],
         ovr_obs_prob,
         net_kw_default),

        ("N",
         [40, 80, 160, 320],
         ovr_N,
         net_kw_default),

        ("density",
         [   # matched SF / Random densities, from sparse to dense
            {"label": "low",      "sf_min_deg": 2, "rand_connect_prob": 0.10},
            {"label": "med-low",  "sf_min_deg": 3, "rand_connect_prob": 0.20},
            {"label": "med-high", "sf_min_deg": 5, "rand_connect_prob": 0.30},
            {"label": "high",     "sf_min_deg": 8, "rand_connect_prob": 0.40},
         ],
         ovr_density,
         net_kw_density),

        ("clause_interval",
         [1, 5, 10, 25, 100],
         ovr_clause_interval,
         net_kw_default),
    ]

    # ---- 8c. Define the time-series configurations ----------
    # These produce the "classic" avg/min violations over time plots.
    # Two configs:
    #   "default"  -- all parameters at baseline
    #   "combined" -- single run with ALL of:
    #                   alpha=8, xor_prop=1, obs_prob=0.01, N=300,
    #                   density=med-high, clause_interval=10
    # (obs_prob and clause_interval already equal their default values, but
    # we pass them explicitly so the combined config is self-documenting.)
    # Each entry: (label, overrides, SF-network-kwargs, Random-network-kwargs)
    MED_HIGH_SF   = dict(type_network="Scale Free", min_deg=5)
    MED_HIGH_RAND = dict(type_network="Random",     connect_prob=0.30)

    timeseries_configs = [
        ("default",  {},                                   SF_DEFAULT,   RAND_DEFAULT),
        ("combined", {"xor_prop": 1.0,
                      "model": {"alpha":           8,
                                "N":               300,
                                "obs_prob":        0.01,
                                "clause_interval": 10}},   MED_HIGH_SF,  MED_HIGH_RAND),
        # Default parameters (incl. obs_prob=0.01) but average per-agent
        # successful elicitation rate scaled down to ~0.1/tick instead of ~1.
        # Tests whether topology effects re-emerge under low-communication.
        ("low_comm", {"comm_rate": 0.1},                   SF_DEFAULT,   RAND_DEFAULT),
    ]

    # ---- 8d. Run each parameter sweep -----------------------
    raw_frames = {}   # param_name -> raw DataFrame (one row per run)
    summaries  = {}   # param_name -> summary DataFrame (one row per param value)

    overall_t0 = time.time()

    for (name, values, ovr_fn, net_fn) in sweeps:
        df_raw = run_sweep(name, values, ovr_fn, net_fn, label=name)

        # Preserve display order for density (categorical labels)
        if name == "density":
            order = [v["label"] for v in values]
            df_raw[name] = pd.Categorical(df_raw[name], categories=order, ordered=True)

        raw_frames[name] = df_raw
        summaries[name]  = summarize(df_raw, name)

    sweeps_minutes = (time.time() - overall_t0) / 60
    print(f"\nAll sweeps finished in {sweeps_minutes:.1f} minutes.", flush=True)

    # ---- 8e. Run the time-series configurations -------------
    ts_frames = {}   # config_label -> DataFrame with per-tick means and SEs
    ts_t0 = time.time()

    for (label, overrides, sf_kw, rand_kw) in timeseries_configs:
        ts_frames[label] = run_config_timeseries(label, sf_kw, rand_kw, overrides)

    ts_minutes = (time.time() - ts_t0) / 60
    print(f"\nAll time-series runs finished in {ts_minutes:.1f} minutes.", flush=True)

    # ---- 8f. Write ONE consolidated Excel spreadsheet -------
    # Tabs:
    #   <param>               -- per-parameter summary for each of the 6 sweeps
    #   all_summaries         -- long-format table of all six summaries
    #   raw                   -- every single sweep run (seed level)
    #   ts_<config>           -- per-tick means and SEs for each time-series config
    xlsx_path = out_dir / "long_sweep_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:

        # (i) per-parameter summary sheets
        for name, summary in summaries.items():
            summary.to_excel(writer, sheet_name=name[:31])  # Excel sheet-name limit = 31

        # (ii) combined summary sheet
        combined = []
        for name, summary in summaries.items():
            tidy = summary.reset_index().rename(columns={summary.index.name or "index": "value"})
            tidy.insert(0, "parameter", name)
            combined.append(tidy)
        pd.concat(combined, ignore_index=True).to_excel(
            writer, sheet_name="all_summaries", index=False)

        # (iii) raw per-run sheet
        raw_all = []
        for name, df in raw_frames.items():
            d = df.copy()
            d["parameter"] = name
            d = d.rename(columns={name: "value"})
            raw_all.append(d[["parameter", "value", "network", "seed", "avg_V", "min_V"]])
        pd.concat(raw_all, ignore_index=True).to_excel(
            writer, sheet_name="raw", index=False)

        # (iv) one time-series tab per config
        for label, ts_df in ts_frames.items():
            # sheet names must not contain some characters ( [ ] : * ? / \ )
            safe = label.replace("/", "_").replace(":", "_")
            ts_df.to_excel(writer, sheet_name=f"ts_{safe}"[:31], index=False)

    print(f"-> saved {xlsx_path}", flush=True)

    # ---- 8g. Save individual per-sweep plots ----------------
    for name, summary in summaries.items():
        plot_sweep_raw(summary, name,
                       outfile=plots_dir / f"sweep_{name}_raw.png",
                       title=f"{name}: raw avg violations  (T={T}, {len(SEEDS)} seeds)")
        plot_sweep_gap(summary, name,
                       outfile=plots_dir / f"sweep_{name}_gap.png",
                       title=f"{name}: SF - Random gap  (T={T}, {len(SEEDS)} seeds)")
        print(f"-> saved sweep_{name}_raw.png and _gap.png", flush=True)

    plot_overview_avg(summaries, outfile=plots_dir / "overview_avg.png")
    plot_overview_gap(summaries, outfile=plots_dir / "overview_gap.png")
    print("-> saved overview_avg.png and overview_gap.png", flush=True)

    # ---- 8h. Save individual time-series plots --------------
    for label, ts_df in ts_frames.items():
        plot_timeseries_one(ts_df, "avg", label,
                            outfile=plots_dir / f"timeseries_{label}_avg.png",
                            ylabel="avg violations")
        plot_timeseries_one(ts_df, "min", label,
                            outfile=plots_dir / f"timeseries_{label}_min.png",
                            ylabel="min violations")
        print(f"-> saved timeseries_{label}_avg.png and _min.png", flush=True)

    # ---- 8i. Print the final tables to stdout ---------------
    print("\n" + "#" * 70)
    print("# FINAL SUMMARY (positive gap = Scale Free is worse than Random)")
    print("#" * 70)
    for name, summary in summaries.items():
        print(f"\n--- {name} ---")
        print(summary[["SF_avg", "Rand_avg", "gap_avg",
                       "SF_min", "Rand_min", "gap_min"]].to_string())

    total_minutes = sweeps_minutes + ts_minutes
    print(f"\nSweeps: {sweeps_minutes:.1f} min  "
          f"Time-series: {ts_minutes:.1f} min  "
          f"Total: {total_minutes:.1f} min")


if __name__ == "__main__":
    main()
