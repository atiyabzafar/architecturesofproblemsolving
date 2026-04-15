"""
Long-run comparison of two network datasets.
Based on network_comparison.py — structure and plotting style preserved.
T = 10000 steps, R = 10000, clause_interval = 10, 50 seeds each.
"""

from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from model import ProblemSolvingModel

os.makedirs("output", exist_ok=True)

T=10000 # things were hard-coded somewhere before, so I'm placing this knob here for convenience


# ============================================================
# Simulation worker  (same signature as network_comparison.py)
# ============================================================

def run_network_simulation(params):
    """Run a single simulation for a network dataset."""
    filepath, seed = params

    try:
        model = ProblemSolvingModel(
            K=50,
            alpha=2,
            obs_prob=0.01,
            clause_interval=10,   # kept from network_comparison.py
            R=T,
            setup_source="dataset",
            file_path=filepath,
            seed=seed,
        )

        metrics = []
        for _ in range(T):
            model.step()
            metrics.append({
                'step':           model.steps,
                'avg_violations': model.avg_true_V,
                'min_violations': model.min_true_V,
                'homogeneity':    model.homogeneity,
                'network':        filepath.split('/')[-1],
                'seed':           seed,
            })

        return metrics

    except Exception as e:
        print(f"Error with {filepath}: {str(e)}")
        return []


# ============================================================
# Setup parameters  — two datasets from network_comparison.py
# ============================================================

graphfiles = [
    "data/congress.graphml",
    "data/EmailManufacturing-copy.xml",
]

network_names = {
    "congress.graphml":             "Congress Twitter",
    "EmailManufacturing-copy.xml":  "Email Manufacturing",
}

seeds  = range(42, 42 + 5)   # 50 seeds per network
params = [(f, s) for f in graphfiles for s in seeds]


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    # --- Run simulations in parallel ---
    print(f"Running parallel simulations (T={T}, R={R}) ...")
    with Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(run_network_simulation, params),
                            total=len(params)))

    # Flatten results
    all_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_results)

    # Replace filenames with readable names
    df['network'] = df['network'].map(network_names)

    # Save raw data
   df.to_csv(f"output/longrun_network_comparison_{T}.csv", index=False)
    print(f"Data saved → output/longrun_network_comparison_{T}.csv")

    # ----------------------------------------------------------
    # Plot 1 – three-panel overview  (style from network_comparison.py)
    # ----------------------------------------------------------
    metrics = ['avg_violations', 'min_violations', 'homogeneity']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, metric in enumerate(metrics):
        stats = df.groupby(['network', 'step'])[metric] \
                  .agg(['mean', 'std']).reset_index()

        for network in df['network'].unique():
            network_data = stats[stats['network'] == network]

            axes[idx].plot(network_data['step'],
                           network_data['mean'],
                           label=network)

            axes[idx].fill_between(network_data['step'],
                                   network_data['mean'] - network_data['std'],
                                   network_data['mean'] + network_data['std'],
                                   alpha=0.2)

        axes[idx].set_xlabel('Time Steps')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].grid(True)
        if idx == 0:
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f'Network Comparison: Evolution of Metrics Over Time  (T={T})')
    plt.tight_layout()

    print("Saving plots...")
    fig.savefig(f"output/longrun_network_comparison_{T}_overview.png",
                dpi=300, bbox_inches='tight')
    fig.savefig(f"output/longrun_network_comparison_{T}_overview.pdf",
                bbox_inches='tight')
    plt.close(fig)

    # Print summary statistics
    print("\nFinal timestep statistics:")
    final_stats = df[df['step'] == df['step'].max()].groupby('network').agg({
        'avg_violations': ['mean', 'std'],
        'min_violations': ['mean', 'std'],
        'homogeneity':    ['mean', 'std'],
    })
    print(final_stats)

    # ----------------------------------------------------------
    # Plot 2 – publication style  (style from network_comparison.py)
    # ----------------------------------------------------------
    import scienceplots
    plt.style.use(['science', 'nature'])

    metrics_pub = ['avg_violations', 'homogeneity']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    network_colors = {}   # keep colours consistent across panels

    for idx, metric in enumerate(metrics_pub):
        stats = df.groupby(['network', 'step'])[metric] \
                  .agg(['mean', 'std']).reset_index()

        for network in df['network'].unique():
            network_data = stats[stats['network'] == network]

            line = axes[idx].plot(network_data['step'],
                                  network_data['mean'],
                                  label=network,
                                  color=network_colors.get(network))

            if network not in network_colors:
                network_colors[network] = line[0].get_color()

            axes[idx].fill_between(network_data['step'],
                                   network_data['mean'] - network_data['std'],
                                   network_data['mean'] + network_data['std'],
                                   alpha=0.2,
                                   color=network_colors[network])

        # Overlay min_violations as dashed on the violations panel
        if idx == 0:
            stats_min = df.groupby(['network', 'step'])['min_violations'] \
                          .agg(['mean', 'std']).reset_index()
            for network in df['network'].unique():
                nd = stats_min[stats_min['network'] == network]
                axes[idx].plot(nd['step'], nd['mean'],
                               color=network_colors[network],
                               linestyle='--')
                axes[idx].fill_between(nd['step'],
                                       nd['mean'] - nd['std'],
                                       nd['mean'] + nd['std'],
                                       alpha=0.2,
                                       color=network_colors[network])

        axes[idx].set_xlabel('Time Steps')
        axes[idx].grid(True)

        if idx == 0:
            axes[idx].set_title(
                r'Avg Violations (Solid) \& Min Violations (Dashed)',
                fontsize=12, pad=10)
            axes[idx].set_ylabel('Violations')
        else:
            axes[idx].set_title('Homogeneity', fontsize=12, pad=10)
            axes[idx].set_ylabel('Homogeneity')

    # Shared legend below both plots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, -0.05),
               ncol=len(df['network'].unique()) // 2 or 1)

    plt.suptitle(f'Network Comparison: Evolution of Metrics Over Time  (T={T})')
    plt.tight_layout()

    fig.savefig(f"output/longrun_network_comparison_{T}_publication.png",
                dpi=300, bbox_inches='tight')
    fig.savefig(f"output/longrun_network_comparison_{T}_publication.pdf",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("\nAll outputs saved to output/")
