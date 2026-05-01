"""
compare_scalefree_vs_random.py
-------------------------------
Compare Scale-Free vs Random network using built-in generators.
Mesa 3+ compatible (requires updated model.py).

Network density is matched:
  Scale-Free : min_deg=2  → E[edges] ≈ 2*(N-2) directed edges
  Random     : connect_prob set so E[edges] ≈ same

Outputs (in output/):
  scalefree_vs_random.csv              – full per-step per-seed data
  scalefree_vs_random_overview.png     – 3-panel overview
  scalefree_vs_random_publication.png  – 2-panel publication figure
"""

from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from model import ProblemSolvingModel   # Mesa 3+ model

os.makedirs("output", exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────

N               = 100
K               = 50
ALPHA           = 2
OBS_PROB        = 0.01
CLAUSE_INTERVAL = 10
T               = 2000    # steps per run
R               = 2000    # model horizon
N_SEEDS         = 5
SEEDS           = range(42, 42 + N_SEEDS)
N_PROCESSES     = 10

MIN_DEG      = 2
# Match edge density: E[SF edges] ≈ 2*(N-2); E[ER edges] = p*N*(N-1)
CONNECT_PROB = round(2 * (N - MIN_DEG) / (N * (N - 1)), 4)   # ≈ 0.0198

NETWORK_CONFIGS = {
    "Scale Free": dict(type_network="Scale Free", min_deg=MIN_DEG),
    "Random":     dict(type_network="Random",     connect_prob=CONNECT_PROB),
}

PALETTE = {
    "Scale Free": "#01696f",   # teal
    "Random":     "#964219",   # brown
}

METRICS      = ["avg_violations", "min_violations", "homogeneity"]
CSV_OUT      = "output/scalefree_vs_random.csv"
PLOT_OVERVIEW= "output/scalefree_vs_random_overview.png"
PLOT_PUB     = "output/scalefree_vs_random_publication.png"


# ── Bug fix: ensure all nodes exist in network before caching neighbors ──────
# Applied inside model.py's setup_*_network methods by calling:
#   self.network.add_nodes_from(range(self.N))
# at the top of each generator.  If your model.py does not have this line,
# add it as the first statement in setup_random_network() and
# setup_scale_free_network() to avoid NetworkXError on isolated nodes.


# ── Worker ───────────────────────────────────────────────────────────────────

def run_simulation(params):
    label, net_kwargs, seed = params
    try:
        model = ProblemSolvingModel(
            N=N, K=K, alpha=ALPHA,
            obs_prob=OBS_PROB,
            clause_interval=CLAUSE_INTERVAL,
            R=R,
            setup_source="generate",
            seed=seed,
            **net_kwargs,
        )
        rows = []
        for _ in range(T):
            model.step()
            rows.append({
                "step":           model.steps,
                "avg_violations": model.avg_true_V,
                "min_violations": model.min_true_V,
                "homogeneity":    model.homogeneity,
                "network":        label,
                "seed":           seed,
            })
        return rows
    except Exception as e:
        print(f"[ERROR] {label} seed={seed}: {e}")
        return []


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    params = [
        (label, kwargs, seed)
        for label, kwargs in NETWORK_CONFIGS.items()
        for seed in SEEDS
    ]

    print("=" * 60)
    print("Scale-Free vs Random ")
    print(f"N={N}, K={K}, α={ALPHA}, obs_prob={OBS_PROB}")
    print(f"clause_interval={CLAUSE_INTERVAL}, T={T}, {N_SEEDS} seeds")
    print(f"Scale-Free min_deg={MIN_DEG}  |  "
          f"Random connect_prob={CONNECT_PROB} (density-matched)")
    print("=" * 60)

    with Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(run_simulation, params),
            total=len(params),
            desc="Simulating",
        ))

    df = pd.DataFrame([r for seed_rows in results for r in seed_rows])
    df.to_csv(CSV_OUT, index=False)
    print(f"\nData → {CSV_OUT}  ({len(df):,} rows)")

    # ── Per-step statistics ───────────────────────────────────────────────
    stats = (
        df.groupby(["network", "step"])[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = (
        ["network", "step"] +
        [f"{col}_{agg}" for col, agg in stats.columns[2:]]
    )

    # ── Plot 1: three-panel overview ──────────────────────────────────────
    panel_labels = {
        "avg_violations": "Avg Violations",
        "min_violations": "Min Violations",
        "homogeneity":    "Homogeneity",
    }
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, metric in zip(axes, METRICS):
        m_col, s_col = f"{metric}_mean", f"{metric}_std"
        for net in NETWORK_CONFIGS:
            sub   = stats[stats["network"] == net]
            color = PALETTE[net]
            ax.plot(sub["step"], sub[m_col],
                    label=net, color=color, linewidth=1.8)
            ax.fill_between(sub["step"],
                            sub[m_col] - sub[s_col],
                            sub[m_col] + sub[s_col],
                            alpha=0.2, color=color)
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel(panel_labels[metric], fontsize=12)
        ax.set_title(panel_labels[metric], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Scale-Free vs Random  |  N={N}, K={K}, α={ALPHA}, "
        f"T={T}, {N_SEEDS} seeds",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(PLOT_OVERVIEW, dpi=300, bbox_inches="tight")
    print(f"Overview plot → {PLOT_OVERVIEW}")
    plt.close(fig)

    # ── Plot 2: publication style ─────────────────────────────────────────
    try:
        import scienceplots
        plt.style.use(["science", "nature"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    legend_handles = {}

    for net in NETWORK_CONFIGS:
        sub   = stats[stats["network"] == net]
        color = PALETTE[net]

        # Left: avg violations (solid) + min violations (dashed)
        ax = axes2[0]
        line, = ax.plot(sub["step"], sub["avg_violations_mean"],
                        color=color, linewidth=1.8, label=net)
        legend_handles[net] = line
        ax.fill_between(sub["step"],
                        sub["avg_violations_mean"] - sub["avg_violations_std"],
                        sub["avg_violations_mean"] + sub["avg_violations_std"],
                        alpha=0.2, color=color)
        ax.plot(sub["step"], sub["min_violations_mean"],
                color=color, linewidth=1.8, linestyle="--")
        ax.fill_between(sub["step"],
                        sub["min_violations_mean"] - sub["min_violations_std"],
                        sub["min_violations_mean"] + sub["min_violations_std"],
                        alpha=0.15, color=color)

        # Right: homogeneity
        axes2[1].plot(sub["step"], sub["homogeneity_mean"],
                      color=color, linewidth=1.8, label=net)
        axes2[1].fill_between(sub["step"],
                              sub["homogeneity_mean"] - sub["homogeneity_std"],
                              sub["homogeneity_mean"] + sub["homogeneity_std"],
                              alpha=0.2, color=color)

    axes2[0].set_xlabel("Time Steps")
    axes2[0].set_ylabel("Violations")
    axes2[0].set_title("Avg Violations (solid)  &  Min Violations (dashed)")
    axes2[0].grid(True)
    axes2[1].set_xlabel("Time Steps")
    axes2[1].set_ylabel("Homogeneity")
    axes2[1].set_title("Homogeneity")
    axes2[1].grid(True)

    fig2.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=2, frameon=True,
    )
    plt.suptitle(
        f"Scale-Free vs Random  |  N={N}, K={K}, α={ALPHA}, "
        f"T={T}, {N_SEEDS} seeds"
    )
    plt.tight_layout()
    fig2.savefig(PLOT_PUB, dpi=300, bbox_inches="tight")
    print(f"Publication plot → {PLOT_PUB}")
    plt.close(fig2)

    # ── Terminal summary ──────────────────────────────────────────────────
    final = df[df["step"] == df["step"].max()]
    print("\n=== Final-step statistics (mean ± std across seeds) ===")
    print(
        final.groupby("network")[METRICS]
        .agg(["mean", "std"])
        .to_string()
    )
    print(f"\nAnalytical prediction: avg_violations → αK/2 = {ALPHA*K/2}")