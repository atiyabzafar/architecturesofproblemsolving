"""
better_random_comparison.py
---------------------------
An optimized and expanded comparison script for generated network topologies.
Matches average in-degree across Random, Scale Free, Small World, and Hierarchical networks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from tqdm import tqdm

from model_2026_04_21 import ProblemSolvingModel

# --- Configuration ---
N = 100
K = 50
ALPHA = 2.0
OBS_PROB = 0.01
CLAUSE_INTERVAL = 10
T_STEPS = 10000
N_SEEDS = 15  # Increased for better statistical significance
SEEDS = range(100, 100 + N_SEEDS)
N_PROCS = min(os.cpu_count(), 10)

# --- Density Matching Calculations ---
# Baseline: Scale Free with min_deg=3
SF_MIN_DEG = 3
# Theoretical Edge Count: SF_MIN_DEG * (N - SF_MIN_DEG)
TARGET_EDGES = SF_MIN_DEG * (N - SF_MIN_DEG)

# Match Random p: p * N * (N-1) = TARGET_EDGES
RAND_P = round(TARGET_EDGES / (N * (N - 1)), 4)

# Match Small World neighbor size: n_size * N = TARGET_EDGES
SW_K = int(round(TARGET_EDGES / N))

CONFIGS = {
    "Random":       {"type_network": "Random", "connect_prob": RAND_P},
    "Scale Free":   {"type_network": "Scale Free", "min_deg": SF_MIN_DEG},
    "Small World":  {"type_network": "Small World", "n_size": SW_K, "rewire_prob": 0.1},
    "Hierarchical": {"type_network": "Hierarchical", "nlayers": 3, "intra_layer_connectance": 0.4, "inter_layer_connectance": 0.05}
}

OUTPUT_DIR = "output/better_random_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_simulation(params):
    label, kwargs, seed = params
    try:
        model = ProblemSolvingModel(
            N=N, K=K, alpha=ALPHA,
            obs_prob=OBS_PROB,
            clause_interval=CLAUSE_INTERVAL,
            R=T_STEPS,
            setup_source="generate",
            seed=seed,
            **kwargs
        )
        
        # Track realized edges for verification
        edge_count = model.network.number_of_edges()
        
        results = []
        for _ in range(T_STEPS):
            model.step()
            results.append({
                "Step": model.steps,
                "Avg Violations": model.avg_true_V,
                "Min Violations": model.min_true_V,
                "Homogeneity": model.homogeneity,
                "Network": label,
                "Seed": seed,
                "Realized Edges": edge_count
            })
        return results
    except Exception as e:
        print(f"Error in {label} (Seed {seed}): {e}")
        return []

if __name__ == "__main__":
    print("="*50)
    print(f"Comprehensive Network Comparison (N={N}, K={K}, α={ALPHA})")
    print(f"Targeting ~{TARGET_EDGES} edges across all topologies")
    print("="*50)

    job_params = [
        (label, kwargs, seed)
        for label, kwargs in CONFIGS.items()
        for seed in SEEDS
    ]

    with Pool(processes=N_PROCS) as pool:
        all_results = list(tqdm(
            pool.imap_unordered(run_simulation, job_params),
            total=len(job_params),
            desc="Simulating Topologies"
        ))

    # Flatten and convert to DataFrame
    df = pd.DataFrame([row for sublist in all_results for row in sublist])
    
    # Save Data
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSimulation complete. Data saved to {csv_path}")

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Violations Evolution
    sns.lineplot(
        data=df, x="Step", y="Avg Violations", hue="Network", 
        ax=axes[0], palette="viridis", linewidth=2
    )
    axes[0].set_title("Problem Solving Performance (Avg Violations)", fontsize=14)
    axes[0].set_ylabel("Average Unsatisfied Constraints")

    # Plot 2: Homogeneity Evolution
    sns.lineplot(
        data=df, x="Step", y="Homogeneity", hue="Network", 
        ax=axes[1], palette="viridis", linewidth=2
    )
    axes[1].set_title("Collective Agreement (Homogeneity)", fontsize=14)
    axes[1].set_ylabel("Agreement Level (0-1)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "performance_over_time.png"), dpi=300)
    
    # Final Statistics Printout
    print("\n" + "-"*30)
    print("Final Equilibrium Statistics (Mean ± Std)")
    print("-"*30)
    final_stats = df[df["Step"] == T_STEPS].groupby("Network").agg({
        "Avg Violations": ["mean", "std"],
        "Homogeneity": ["mean", "std"],
        "Realized Edges": "mean"
    })
    print(final_stats)

    plt.show()
