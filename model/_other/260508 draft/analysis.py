"""
Analysis for "Problem solving and organisation structure".

This script runs the agent-based model on five canonical network topologies
and computes both steady-state and transient performance metrics for each.

Topologies (all built with N=80 nodes):
    Random          Erdos-Renyi
    Small World     Watts-Strogatz
    Scale Free      Barabasi-Albert (preferential attachment)
    Hierarchical    Layered (corporate-like) with intra/inter-layer links
    Modular         Stochastic block model with 4 communities

Steady-state metrics (averaged over the post-burn-in window):
    avg_V_long      time-averaged mean violations across agents
    min_V_long      time-averaged best-agent violations
    gini_V_long     across-agent inequality of violation counts
    H_long          homogeneity (majority-share agreement on each variable)
    vol_V           within-run standard deviation of avg_V (stability)

Transient metrics (computed from the trajectory leaving initial conditions):
    t_conv          ticks until avg_V first enters within 5% of its long-run mean
    slope_V_max     the steepest negative slope of avg_V during the first 1000 ticks
    H_burnin        homogeneity averaged over the first 1000 ticks

Optional shock-recovery experiment (`run_shock_recovery=True`):
    At tick T_shock the universal clause set is perturbed by replacing a fraction
    `shock_fraction` of clauses at once. Recovery time is the number of post-shock
    ticks until avg_V returns to within 5% of its pre-shock mean.

Outputs (under "260508 draft/output/"):
    topology_results.xlsx       per-network steady-state and transient summaries +
                                downsampled time-series tables
    figures/timeseries_avg_V.png    avg violations over time, all networks overlaid
    figures/timeseries_min_V.png
    figures/timeseries_gini_V.png
    figures/timeseries_H.png
    figures/steady_state_summary.png  bar chart of steady-state metrics
    figures/transient_summary.png     bar chart of transient metrics
    figures/shock_recovery.png         (only if shock experiment is run)

Usage:
    py -u analysis.py                       # default settings, no shock
    py -u analysis.py --shock               # include shock experiment
    py -u analysis.py --quick               # short smoke test
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Find model.py at the grandparent folder (this script is at _other/260508 draft/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model import ProblemSolvingModel


# ============================================================
# Configuration
# ============================================================

# Default run parameters (override by passing --quick for a smoke test)
DEFAULT_T_TOTAL  = 10_000      # total ticks per run
DEFAULT_BURN_IN  = 5_000       # ticks discarded as transient when computing "long" metrics
DEFAULT_SEEDS    = [42, 43, 44, 45, 46]

# Model parameters held fixed across topologies
N         = 80                 # number of agents
K         = 30                 # number of binary decision variables
ALPHA     = 2                  # clause density (M = alpha * K)
OBS_PROB  = 0.01               # private-observation probability per tick
CINT      = 10                 # ticks between universal-clause replacements
TRANSIENT_WINDOW = 1000        # window over which transient metrics are computed

# Plot colours -- one per topology
COLORS = {
    "Random":       "tab:blue",
    "Small World":  "tab:orange",
    "Scale Free":   "tab:green",
    "Hierarchical": "tab:red",
    "Modular":      "tab:purple",
}


# ============================================================
# Network builders
# ============================================================
# Each builder returns a directed nx.DiGraph with N nodes whose edges all
# carry weight=1.0 (so the model's comm_scale normalisation works on
# in-degree, not weighted strength). For undirected generators we orient
# each edge randomly with 50/50 probability.

def _to_asymmetric_directed(G_undirected, rng):
    """Convert undirected graph to directed by random edge orientation."""
    G = nx.DiGraph()
    G.add_nodes_from(G_undirected.nodes())
    for u, v in G_undirected.edges():
        if u == v:
            continue
        if rng.random() < 0.5:
            G.add_edge(u, v, weight=1.0)
        else:
            G.add_edge(v, u, weight=1.0)
    return G


def network_random(seed, n=N):
    """Erdos-Renyi directed graph with target ~620 edges (mean in-degree ~7.75)."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    p = 0.10
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < p:
                G.add_edge(i, j, weight=1.0)
    return G


def network_small_world(seed, n=N):
    """Watts-Strogatz with k=8 nearest neighbours, rewire prob 0.1, then random direction."""
    rng = np.random.default_rng(seed)
    G_und = nx.watts_strogatz_graph(n, k=8, p=0.1, seed=seed)
    return _to_asymmetric_directed(G_und, rng)


def network_scale_free(seed, n=N):
    """Barabasi-Albert with min_deg=4, then random direction.
    BA gives undirected mean degree ~2*min_deg=8, matching the random-graph density."""
    rng = np.random.default_rng(seed)
    G_und = nx.barabasi_albert_graph(n, m=4, seed=seed)
    return _to_asymmetric_directed(G_und, rng)


def network_hierarchical(seed, n=N):
    """Layered network: 4 layers of ~equal size, intra-layer prob 0.20,
    inter-layer prob 0.05 (downstream + upstream, no jumps across layers).
    Intended to mimic a corporate org chart with cross-functional links."""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    n_layers = 4
    layer_size = n // n_layers
    layers = [list(range(i * layer_size, (i + 1) * layer_size)) for i in range(n_layers)]
    # Intra-layer
    for layer in layers:
        for i in layer:
            for j in layer:
                if i != j and rng.random() < 0.20:
                    G.add_edge(i, j, weight=1.0)
    # Inter-layer (between any two distinct layers)
    for a, layer_a in enumerate(layers):
        for b, layer_b in enumerate(layers):
            if a != b:
                for i in layer_a:
                    for j in layer_b:
                        if rng.random() < 0.05:
                            G.add_edge(i, j, weight=1.0)
    return G


def network_modular(seed, n=N):
    """Stochastic block model with 4 communities of equal size, p_in=0.20, p_out=0.02.
    Then orient each undirected edge randomly. Intended to mimic
    department-organised firms or politically segmented networks."""
    rng = np.random.default_rng(seed)
    sizes = [n // 4] * 4
    p_in, p_out = 0.20, 0.02
    p_matrix = [[p_in if i == j else p_out for j in range(4)] for i in range(4)]
    G_und = nx.stochastic_block_model(sizes, p_matrix, seed=seed)
    return _to_asymmetric_directed(G_und, rng)


NETWORK_BUILDERS = [
    ("Random",        network_random),
    ("Small World",   network_small_world),
    ("Scale Free",    network_scale_free),
    ("Hierarchical",  network_hierarchical),
    ("Modular",       network_modular),
]


# ============================================================
# Metric helpers
# ============================================================

def gini(values):
    """Gini coefficient of a 1D array of non-negative values.
    0 = perfectly equal; 1 = one value holds everything."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n


# ============================================================
# Run a single (topology, seed) pair and return per-tick time series
# ============================================================

def run_one(network_builder, seed, T):
    """Build the network, instantiate the model, step T times, return per-tick metrics.

    Returns a dict with arrays of length T:
        avg_V, min_V, gini_V, H
    """
    G = network_builder(seed)
    m = ProblemSolvingModel(
        N=N, K=K, alpha=ALPHA,
        obs_prob=OBS_PROB, clause_interval=CINT,
        setup_source="graph", input_graph=G,
        R=T, seed=seed,
    )
    avg_V  = np.zeros(T)
    min_V  = np.zeros(T)
    gini_V = np.zeros(T)
    H      = np.zeros(T)
    for t in range(T):
        m.step()
        violations = np.array([a.true_violations for a in m.agents], dtype=float)
        avg_V[t]  = m.avg_true_V
        min_V[t]  = m.min_true_V
        gini_V[t] = gini(violations)
        H[t]      = m.homogeneity
    return {"avg_V": avg_V, "min_V": min_V, "gini_V": gini_V, "H": H}


# ============================================================
# Steady-state and transient summaries
# ============================================================

def steady_state_summary(per_seed_data, burn_in):
    """Average post-burn-in metrics across seeds, with SE."""
    rows = []
    for net_name, runs in per_seed_data.items():
        n_seeds = len(runs)
        sqrt_n  = np.sqrt(n_seeds)

        # Stack across seeds: shape (n_seeds, T)
        stacks = {k: np.stack([r[k] for r in runs]) for k in ("avg_V", "min_V", "gini_V", "H")}

        row = {"network": net_name}
        for metric_name in ("avg_V", "min_V", "gini_V", "H"):
            tail = stacks[metric_name][:, burn_in:]
            per_seed_means = tail.mean(axis=1)
            row[f"{metric_name}_long_mean"] = float(per_seed_means.mean())
            row[f"{metric_name}_long_sem"]  = float(per_seed_means.std(ddof=1) / sqrt_n)

        # Stability: within-run std of avg_V over the post-burn-in window,
        # averaged across seeds.
        within_run_std = stacks["avg_V"][:, burn_in:].std(axis=1, ddof=1)
        row["vol_V_mean"] = float(within_run_std.mean())
        row["vol_V_sem"]  = float(within_run_std.std(ddof=1) / sqrt_n)

        rows.append(row)
    return pd.DataFrame(rows)


def transient_summary(per_seed_data, burn_in, transient_window=TRANSIENT_WINDOW):
    """Convergence speed, peak slope, early-window homogeneity."""
    rows = []
    for net_name, runs in per_seed_data.items():
        n_seeds = len(runs)
        sqrt_n  = np.sqrt(n_seeds)

        stacks = {k: np.stack([r[k] for r in runs]) for k in ("avg_V", "min_V", "H")}
        T_total = stacks["avg_V"].shape[1]

        # Long-run mean of avg_V (shared threshold for all seeds of this network)
        long_mean = stacks["avg_V"][:, burn_in:].mean()
        threshold = long_mean * 1.05    # within 5% of the long-run mean

        # t_conv: ticks until avg_V first dips below threshold
        t_conv_per_seed = []
        for run in stacks["avg_V"]:
            below = np.where(run <= threshold)[0]
            t_conv = int(below[0]) + 1 if len(below) > 0 else T_total
            t_conv_per_seed.append(t_conv)

        # slope_V_max: most-negative average slope over a 50-tick window,
        # taken inside the first transient_window ticks
        slope_per_seed = []
        win = 50
        for run in stacks["avg_V"]:
            limit = min(transient_window, T_total)
            if limit > win:
                slopes = (run[win:limit] - run[:limit - win]) / win
                slope_per_seed.append(float(slopes.min()))
            else:
                slope_per_seed.append(0.0)

        # H_burnin: homogeneity averaged over first transient_window ticks
        h_burn = stacks["H"][:, :transient_window].mean(axis=1)

        rows.append({
            "network":              net_name,
            "t_conv_mean":          float(np.mean(t_conv_per_seed)),
            "t_conv_sem":           float(np.std(t_conv_per_seed, ddof=1) / sqrt_n),
            "slope_V_max_mean":     float(np.mean(slope_per_seed)),
            "slope_V_max_sem":      float(np.std(slope_per_seed, ddof=1) / sqrt_n),
            "H_burnin_mean":        float(h_burn.mean()),
            "H_burnin_sem":         float(h_burn.std(ddof=1) / sqrt_n),
        })
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def plot_timeseries(per_seed_data, metric, ylabel, outfile, T_total):
    """Overlaid time series, one curve per network with SE band."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = np.arange(1, T_total + 1)
    for net_name, runs in per_seed_data.items():
        stack = np.stack([r[metric] for r in runs])
        mean = stack.mean(axis=0)
        sem  = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        ax.plot(t, mean, color=COLORS.get(net_name, "k"), label=net_name, lw=1.4)
        ax.fill_between(t, mean - sem, mean + sem,
                        color=COLORS.get(net_name, "k"), alpha=0.18)
    ax.set_xlabel("tick")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_steady_state_summary(ss_df, outfile):
    """Five-panel bar chart: one panel per metric, one bar per network."""
    metrics = [
        ("avg_V_long",   "Avg violations (long-run)"),
        ("min_V_long",   "Min violations (long-run)"),
        ("gini_V_long",  "Gini of violations"),
        ("H_long",       "Homogeneity"),
        ("vol_V",        "Within-run std of avg_V (stability)"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    nets = ss_df["network"].tolist()
    colors = [COLORS.get(n, "tab:gray") for n in nets]
    for ax, (key, label) in zip(axes, metrics):
        means = ss_df[f"{key}_mean"]
        sems  = ss_df[f"{key}_sem"]
        ax.bar(range(len(nets)), means, yerr=sems, color=colors,
               edgecolor="k", linewidth=0.6, capsize=3)
        ax.set_xticks(range(len(nets)))
        ax.set_xticklabels(nets, rotation=20, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_transient_summary(tr_df, outfile):
    """Three-panel bar chart of transient metrics."""
    metrics = [
        ("t_conv",       "Convergence time (ticks)"),
        ("slope_V_max",  "Peak descent slope (avg_V/tick, more negative = faster)"),
        ("H_burnin",     "Homogeneity over first 1000 ticks"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    nets = tr_df["network"].tolist()
    colors = [COLORS.get(n, "tab:gray") for n in nets]
    for ax, (key, label) in zip(axes, metrics):
        means = tr_df[f"{key}_mean"]
        sems  = tr_df[f"{key}_sem"]
        ax.bar(range(len(nets)), means, yerr=sems, color=colors,
               edgecolor="k", linewidth=0.6, capsize=3)
        ax.set_xticks(range(len(nets)))
        ax.set_xticklabels(nets, rotation=20, fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


# ============================================================
# Optional shock-recovery experiment
# ============================================================
# At T_shock we replace a fraction of the universal clause set with fresh
# random clauses. We measure how many ticks each network needs to return
# to within 5% of its pre-shock avg_V level.

def run_one_with_shock(network_builder, seed, T, T_shock, shock_fraction=0.3):
    """Run T ticks; at tick T_shock replace shock_fraction of clauses."""
    G = network_builder(seed)
    m = ProblemSolvingModel(
        N=N, K=K, alpha=ALPHA,
        obs_prob=OBS_PROB, clause_interval=CINT,
        setup_source="graph", input_graph=G,
        R=T, seed=seed,
    )
    avg_V = np.zeros(T)
    rng_local = np.random.default_rng(seed + 9999)
    for t in range(T):
        m.step()
        avg_V[t] = m.avg_true_V
        if t == T_shock - 1:
            # Apply shock: replace shock_fraction of clauses with fresh ones
            n_replace = int(shock_fraction * m.M)
            indices = rng_local.choice(m.M, size=n_replace, replace=False)
            for idx in indices:
                m.C[int(idx)] = m.random_clause()
    return avg_V


def shock_recovery_experiment(seeds, T_total, T_shock, shock_fraction=0.3):
    """For each topology, run the shock experiment across seeds, and
    compute pre-shock baseline and post-shock recovery time."""
    print(f"\n=== Shock recovery experiment ===", flush=True)
    print(f"  Pre-shock window: ticks 1..{T_shock} "
          f"(baseline = mean of last 1000 pre-shock ticks)", flush=True)
    print(f"  Shock at tick {T_shock}: replace {int(shock_fraction*100)}% of clauses",
          flush=True)
    print(f"  Recovery window: ticks {T_shock+1}..{T_total}\n", flush=True)

    per_seed = {name: [] for name, _ in NETWORK_BUILDERS}
    rows = []
    for net_name, builder in NETWORK_BUILDERS:
        recovery_times = []
        peak_jumps     = []
        for seed in seeds:
            avg_V = run_one_with_shock(builder, seed, T_total, T_shock, shock_fraction)
            per_seed[net_name].append(avg_V)
            # Pre-shock baseline: mean over last 1000 ticks before shock
            baseline = avg_V[max(0, T_shock - 1000):T_shock].mean()
            # Peak post-shock value (within 100 ticks)
            post = avg_V[T_shock: T_shock + 100]
            peak_jumps.append(float(post.max() - baseline))
            # Recovery time: ticks until avg_V dips within 5% of baseline
            threshold = baseline * 1.05
            below = np.where(avg_V[T_shock:] <= threshold)[0]
            rec = int(below[0]) + 1 if len(below) > 0 else T_total - T_shock
            recovery_times.append(rec)
            print(f"  {net_name:<14} seed={seed}  baseline={baseline:5.2f}  "
                  f"peak_jump={post.max() - baseline:+5.2f}  recovery={rec}t",
                  flush=True)
        n = len(seeds)
        rows.append({
            "network":             net_name,
            "recovery_time_mean":  float(np.mean(recovery_times)),
            "recovery_time_sem":   float(np.std(recovery_times, ddof=1) / np.sqrt(n)),
            "peak_jump_mean":      float(np.mean(peak_jumps)),
            "peak_jump_sem":       float(np.std(peak_jumps, ddof=1) / np.sqrt(n)),
        })
    return pd.DataFrame(rows), per_seed


def plot_shock_recovery(per_seed_data, T_total, T_shock, outfile):
    """Time series of avg_V around the shock event."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(1, T_total + 1)
    for net_name, runs in per_seed_data.items():
        stack = np.stack(runs)
        mean = stack.mean(axis=0)
        sem  = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        ax.plot(t, mean, color=COLORS.get(net_name, "k"), label=net_name, lw=1.4)
        ax.fill_between(t, mean - sem, mean + sem,
                        color=COLORS.get(net_name, "k"), alpha=0.18)
    ax.axvline(T_shock, color="k", ls="--", lw=1, alpha=0.5, label="shock")
    ax.set_xlabel("tick")
    ax.set_ylabel("avg violations")
    ax.set_title("Shock recovery: 30% of clauses replaced at the shock tick")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: T=500, 2 seeds")
    parser.add_argument("--shock", action="store_true",
                        help="Run shock-recovery experiment in addition to baseline")
    args = parser.parse_args()

    if args.quick:
        T_total, burn_in, seeds = 500, 200, [42, 43]
        print("QUICK SMOKE TEST: T=500, 2 seeds, burn_in=200", flush=True)
    else:
        T_total, burn_in, seeds = DEFAULT_T_TOTAL, DEFAULT_BURN_IN, DEFAULT_SEEDS
        print(f"FULL RUN: T={T_total}, {len(seeds)} seeds, burn_in={burn_in}",
              flush=True)

    # Output folders
    out_dir = Path(__file__).parent / "output"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Run all (topology, seed) pairs ----
    overall_t0 = time.time()
    per_seed_data = {name: [] for name, _ in NETWORK_BUILDERS}
    total = len(NETWORK_BUILDERS) * len(seeds)
    i = 0
    for net_name, builder in NETWORK_BUILDERS:
        for seed in seeds:
            i += 1
            t0 = time.time()
            d = run_one(builder, seed, T_total)
            per_seed_data[net_name].append(d)
            elapsed = (time.time() - overall_t0) / 60
            eta     = elapsed / i * (total - i)
            print(f"  [{i:2d}/{total}] {net_name:<14} seed={seed}  "
                  f"final avg_V={d['avg_V'][-1]:6.2f}  "
                  f"min_V={d['min_V'][-1]:5.0f}  "
                  f"H={d['H'][-1]:.3f}  "
                  f"({time.time() - t0:.1f}s, total {elapsed:.1f}m, eta {eta:.1f}m)",
                  flush=True)

    # ---- 2. Compute summaries ----
    ss_df = steady_state_summary(per_seed_data, burn_in)
    tr_df = transient_summary(per_seed_data, burn_in, TRANSIENT_WINDOW)

    print("\n=== Steady-state summary ===")
    print(ss_df.round(3).to_string(index=False))
    print("\n=== Transient summary ===")
    print(tr_df.round(3).to_string(index=False))

    # ---- 3. Save xlsx ----
    xlsx = out_dir / "topology_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        ss_df.to_excel(w, sheet_name="steady_state", index=False)
        tr_df.to_excel(w, sheet_name="transient", index=False)
        # Downsampled time series (every 50 ticks) per network
        step = max(1, T_total // 200)
        ticks = np.arange(step, T_total + 1, step)
        for net_name, runs in per_seed_data.items():
            rows = []
            stack = {k: np.stack([r[k] for r in runs]) for k in
                     ("avg_V", "min_V", "gini_V", "H")}
            for t in ticks:
                row = {"tick": int(t)}
                for k in ("avg_V", "min_V", "gini_V", "H"):
                    arr = stack[k][:, t - 1]
                    row[f"{k}_mean"] = float(arr.mean())
                    row[f"{k}_sem"]  = float(arr.std(ddof=1) / np.sqrt(len(seeds)))
                rows.append(row)
            sheet = f"ts_{net_name.replace(' ', '_')}"[:31]
            pd.DataFrame(rows).to_excel(w, sheet_name=sheet, index=False)
    print(f"\n-> saved {xlsx}", flush=True)

    # ---- 4. Plots ----
    plot_timeseries(per_seed_data, "avg_V",  "avg violations",
                    fig_dir / "timeseries_avg_V.png", T_total)
    plot_timeseries(per_seed_data, "min_V",  "min violations",
                    fig_dir / "timeseries_min_V.png", T_total)
    plot_timeseries(per_seed_data, "gini_V", "Gini of violations",
                    fig_dir / "timeseries_gini_V.png", T_total)
    plot_timeseries(per_seed_data, "H",      "homogeneity",
                    fig_dir / "timeseries_H.png", T_total)
    plot_steady_state_summary(ss_df, fig_dir / "steady_state_summary.png")
    plot_transient_summary   (tr_df, fig_dir / "transient_summary.png")
    print(f"-> saved 6 figures in {fig_dir}", flush=True)

    # ---- 5. Optional shock-recovery experiment ----
    if args.shock:
        T_shock = T_total // 2
        sr_df, sr_data = shock_recovery_experiment(seeds, T_total, T_shock, 0.3)
        print("\n=== Shock recovery summary ===")
        print(sr_df.round(3).to_string(index=False))
        with pd.ExcelWriter(xlsx, engine="openpyxl",
                            mode="a", if_sheet_exists="replace") as w:
            sr_df.to_excel(w, sheet_name="shock_recovery", index=False)
        plot_shock_recovery(sr_data, T_total, T_shock,
                            fig_dir / "shock_recovery.png")
        print(f"-> saved shock_recovery sheet and figure", flush=True)

    total_min = (time.time() - overall_t0) / 60
    print(f"\nTotal wall time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
