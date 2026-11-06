"""
Long-run simulation comparing two network datasets: CMC and MCL.
T = 10000 steps, R = 10000, 50 seeds each.
"""

from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Locate model.py one level up (this script lives in _other/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import ProblemSolvingModel

# ============================================================
# Configuration
# ============================================================

DATASETS = {
    "CMC": "data/fwdscialogdata/networks/Collab_values_CMC_associated.graphml",
    "MCL": "data/fwdscialogdata/networks/Collab_values_MCL_associated.graphml",
}

T               = 10000
R               = 10000
K               = 50
ALPHA           = 2
OBS_PROB        = 0.01
CLAUSE_INTERVAL = 1
N_SEEDS         = 50
SEEDS           = range(42, 42 + N_SEEDS)
N_PROCESSES     = 10

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_OUT      = os.path.join(OUTPUT_DIR, "longrun_CMC_MCL_10000.csv")
PLOT_RAW_OUT = os.path.join(OUTPUT_DIR, "longrun_CMC_MCL_10000_metrics.png")
PLOT_PUB_OUT = os.path.join(OUTPUT_DIR, "longrun_CMC_MCL_10000_publication.png")

METRICS = ["avg_violations", "min_violations", "homogeneity"]

# Two visually distinct, accessible colours
PALETTE = {
    "CMC": "#01696f",   # teal
    "MCL": "#964219",   # terra brown
}


# ============================================================
# Simulation worker
# ============================================================

def run_simulation(params):
    """Run a single long-time simulation for one dataset/seed pair."""
    label, filepath, seed = params

    try:
        model = ProblemSolvingModel(
            K=K,
            alpha=ALPHA,
            obs_prob=OBS_PROB,
            clause_interval=CLAUSE_INTERVAL,
            R=R,
            setup_source="dataset",
            file_path=filepath,
            seed=seed,
        )

        metrics = []
        for _ in range(T):
            model.step()
            metrics.append({
                "step":           model.steps,
                "avg_violations": model.avg_true_V,
                "min_violations": model.min_true_V,
                "homogeneity":    model.homogeneity,
                "network":        label,
                "seed":           seed,
            })

        return metrics

    except Exception as e:
        print(f"[ERROR] network={label}, seed={seed}: {e}")
        return []


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    params = [
        (label, filepath, seed)
        for label, filepath in DATASETS.items()
        for seed in SEEDS
    ]

    print(f"Comparing datasets: {list(DATASETS.keys())}")
    print(f"T={T}, R={R}, K={K}, alpha={ALPHA}, obs_prob={OBS_PROB}, "
          f"clause_interval={CLAUSE_INTERVAL}")
    print(f"Seeds per dataset: {N_SEEDS}   |   Parallel processes: {N_PROCESSES}")

    with Pool(processes=N_PROCESSES) as pool:
        results = list(tqdm(
            pool.imap(run_simulation, params),
            total=len(params),
            desc="Simulating",
        ))

    # Flatten and save
    all_rows = [row for seed_rows in results for row in seed_rows]
    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nData saved → {CSV_OUT}")

    # ----------------------------------------------------------
    # Per-step statistics per network
    # ----------------------------------------------------------
    stats = (
        df.groupby(["network", "step"])[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats.columns = ["network", "step"] + [
        f"{col}_{agg}" for col, agg in stats.columns[2:]
    ]

    # ----------------------------------------------------------
    # Plot 1 – three-panel overview
    # ----------------------------------------------------------
    panel_labels = {
        "avg_violations": "Avg Violations",
        "min_violations": "Min Violations",
        "homogeneity":    "Homogeneity",
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax, metric in zip(axes, METRICS):
        mean_col = f"{metric}_mean"
        std_col  = f"{metric}_std"

        for net in DATASETS:
            sub   = stats[stats["network"] == net]
            color = PALETTE[net]
            ax.plot(sub["step"], sub[mean_col],
                    label=net, color=color, linewidth=1.5)
            ax.fill_between(
                sub["step"],
                sub[mean_col] - sub[std_col],
                sub[mean_col] + sub[std_col],
                alpha=0.2, color=color,
            )

        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel(panel_labels[metric], fontsize=12)
        ax.set_title(panel_labels[metric], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Long-Run Dynamics — CMC vs MCL  (T={T}, R={R}, {N_SEEDS} seeds each)",
        fontsize=14,
    )
    plt.tight_layout()
    fig.savefig(PLOT_RAW_OUT, dpi=300, bbox_inches="tight")
    print(f"Overview plot saved → {PLOT_RAW_OUT}")
    plt.close(fig)

    # ----------------------------------------------------------
    # Plot 2 – publication style (2 panels)
    # ----------------------------------------------------------
    try:
        import scienceplots
        plt.style.use(["science", "nature"])
    except ImportError:
        plt.style.use("seaborn-v0_8-whitegrid")

    # Track colours across subplots for a consistent shared legend
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    legend_handles = {}

    for net in DATASETS:
        sub   = stats[stats["network"] == net]
        color = PALETTE[net]

        # --- Left panel: avg violations (solid) + min violations (dashed) ---
        ax = axes2[0]
        line, = ax.plot(sub["step"], sub["avg_violations_mean"],
                        color=color, linewidth=1.5, label=net)
        legend_handles[net] = line          # store for shared legend
        ax.fill_between(
            sub["step"],
            sub["avg_violations_mean"] - sub["avg_violations_std"],
            sub["avg_violations_mean"] + sub["avg_violations_std"],
            alpha=0.2, color=color,
        )
        ax.plot(sub["step"], sub["min_violations_mean"],
                color=color, linewidth=1.5, linestyle="--")
        ax.fill_between(
            sub["step"],
            sub["min_violations_mean"] - sub["min_violations_std"],
            sub["min_violations_mean"] + sub["min_violations_std"],
            alpha=0.15, color=color,
        )

        # --- Right panel: homogeneity ---
        ax = axes2[1]
        ax.plot(sub["step"], sub["homogeneity_mean"],
                color=color, linewidth=1.5, label=net)
        ax.fill_between(
            sub["step"],
            sub["homogeneity_mean"] - sub["homogeneity_std"],
            sub["homogeneity_mean"] + sub["homogeneity_std"],
            alpha=0.2, color=color,
        )

    axes2[0].set_xlabel("Time Steps")
    axes2[0].set_ylabel("Violations")
    axes2[0].set_title(r"Avg Violations (solid) \& Min Violations (dashed)")
    axes2[0].grid(True)

    axes2[1].set_xlabel("Time Steps")
    axes2[1].set_ylabel("Homogeneity")
    axes2[1].set_title("Homogeneity")
    axes2[1].grid(True)

    # Shared legend below both panels
    fig2.legend(
        handles=list(legend_handles.values()),
        labels=list(legend_handles.keys()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(DATASETS),
        frameon=True,
    )

    plt.suptitle(
        f"Long-Run Behaviour — CMC vs MCL  (T={T}, R={R}, {N_SEEDS} seeds each)"
    )
    plt.tight_layout()
    fig2.savefig(PLOT_PUB_OUT, dpi=300, bbox_inches="tight")
    print(f"Publication plot saved → {PLOT_PUB_OUT}")
    plt.close(fig2)

    # ----------------------------------------------------------
    # Terminal summary at final step
    # ----------------------------------------------------------
    final = df[df["step"] == df["step"].max()]
    print("\n=== Final-step summary (mean ± std across seeds) ===")
    print(
        final.groupby("network")[METRICS]
        .agg(["mean", "std"])
        .to_string()
    )
