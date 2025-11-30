from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model_new_1 import ProblemSolvingModel


def run_network_simulation(params):
    """Run a single simulation for a network dataset"""
    filepath, seed = params
    
    try:
        model = ProblemSolvingModel(
            K=50,
            alpha=2,
            obs_prob=0.01,
            clause_interval=1,
            R=2000,
            setup_source="dataset",
            file_path=filepath,
            seed=seed
        )
        
        # Store metrics at each timestep
        metrics = []
        for _ in range(2000):
            model.step()
            metrics.append({
                'step': model.schedule.steps,
                'avg_violations': model.avg_true_V,
                'min_violations': model.min_true_V,
                'homogeneity': model.homogeneity,
                'network': filepath.split('/')[-1],
                'seed': seed
            })
        
        return metrics
        
    except Exception as e:
        print(f"Error with {filepath}: {str(e)}")
        return []

# Setup parameters
graphfiles = [
    "data/fwdscialogdata/networks/Collab_values_CMC_associated.graphml",
    "data/fwdscialogdata/networks/Collab_values_MCL_associated.graphml", 
    "data/fwdscialogdata/networks/Collab_values_TDA_associated.graphml",
    "data/fwdscialogdata/networks/Collab_values_AES_associated.graphml",
]

network_names = {
    "Collab_values_CMC_associated.graphml" : "CMC",
    "Collab_values_MCL_associated.graphml" : "MCL", 
    "Collab_values_TDA_associated.graphml" : "TDA",
    "Collab_values_AES_associated.graphml" : "AES",
}

seeds = range(42, 42+50)  # 50 seeds per network
params = [(f, s) for f in graphfiles for s in seeds]

if __name__ == '__main__':
    # Run simulations in parallel
    print("Running parallel simulations...")
    with Pool(processes=10) as pool:
        results = list(tqdm(pool.imap(run_network_simulation, params), total=len(params)))
    
    # Flatten results
    all_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(all_results)
    
    # Replace filenames with readable names
    df['network'] = df['network'].map(network_names)
    
    # Create plots
    metrics = ['avg_violations', 'min_violations', 'homogeneity']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, metric in enumerate(metrics):
        # Calculate mean and std per timestep for each network
        stats = df.groupby(['network', 'step'])[metric].agg(['mean', 'std']).reset_index()
        
        # Plot each network
        for network in df['network'].unique():
            network_data = stats[stats['network'] == network]
            
            # Plot mean line
            axes[idx].plot(network_data['step'], 
                         network_data['mean'], 
                         label=network)
            
            # Add error bands
            axes[idx].fill_between(network_data['step'],
                                 network_data['mean'] - network_data['std'],
                                 network_data['mean'] + network_data['std'],
                                 alpha=0.2)
        
        # Customize plot
        axes[idx].set_xlabel('Time Steps')
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].grid(True)
        if idx == 0:  # Only show legend on first plot
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('Conference Comparison: Evolution of Metrics Over Time')
    plt.tight_layout()
    
    # Save plots
    print("Saving plots...")
    fig.savefig("output/50_conference_comparison_2000_01_1associated.png", dpi=300, bbox_inches='tight')
    fig.savefig("output/50_conference_comparison_2000_01_1associated.pdf",  bbox_inches='tight')
    
    # Print summary statistics
    print("\nFinal timestep statistics:")
    final_stats = df[df['step'] == df['step'].max()].groupby('network').agg({
        'avg_violations': ['mean', 'std'],
        'min_violations': ['mean', 'std'],
        'homogeneity': ['mean', 'std']
    })
    print(final_stats)
    
    # Save data
    df.to_csv("output/50_conference_simulation_results_2000_01_1associated.csv", index=False)

#df=pd.read_csv("output/50_conference_simulation_results_2000_01_10associated.csv")

# Create plots
metrics = ['avg_violations', 'homogeneity']
import scienceplots
plt.style.use(['science','nature'])
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Store colors for each network
network_colors = {}

for idx, metric in enumerate(metrics):
    # Calculate mean and std per timestep for each network
    stats = df.groupby(['network', 'step'])[metric].agg(['mean', 'std']).reset_index()
    
    # Plot each network
    for network in df['network'].unique():
        network_data = stats[stats['network'] == network]
        
        # Plot mean line and store color on first iteration
        line = axes[idx].plot(network_data['step'], 
                        network_data['mean'], 
                        label=network,
                        color=network_colors.get(network))  # Use stored color if available
        
        # Store the color for this network (from first subplot)
        if network not in network_colors:
            network_colors[network] = line[0].get_color()
        
        # Add error bands with matching color
        axes[idx].fill_between(network_data['step'],
                                network_data['mean'] - network_data['std'],
                                network_data['mean'] + network_data['std'],
                                alpha=0.2,
                                color=network_colors[network])
    
    if idx == 0:
        stats = df.groupby(['network', 'step'])['min_violations'].agg(['mean', 'std']).reset_index()
        for network in df['network'].unique():
            network_data = stats[stats['network'] == network]
            
            # Plot mean line with same color - NO label (won't appear in legend)
            axes[idx].plot(network_data['step'], 
                            network_data['mean'], 
                            color=network_colors[network],  # Same color!
                            linestyle='--')  # Different linestyle for clarity
            
            # Add error bands with matching color
            axes[idx].fill_between(network_data['step'],
                                    network_data['mean'] - network_data['std'],
                                    network_data['mean'] + network_data['std'],
                                    alpha=0.2,
                                    color=network_colors[network])
    
    # Customize plot
    axes[idx].set_xlabel('Time Steps')
    
    # Set custom titles
    if idx == 0:
        axes[idx].set_title('Avg Violations (Solid) \& Min Violations (Dashed)', fontsize=12, pad=10)
        axes[idx].set_ylabel('Violations')
    else:
        axes[idx].set_title('Homogeneity', fontsize=12, pad=10)
        axes[idx].set_ylabel('Homogeneity')
    
    axes[idx].grid(True)

# Create a single legend below both plots (only network names, no "(min)" labels)
handles, labels = axes[0].get_legend_handles_labels()  # Get from first plot (only has network names)
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(df['network'].unique())/2)

plt.suptitle('Network Comparison: Evolution of Metrics Over Time')
plt.tight_layout()
#plt.subplots_adjust(bottom=0.15)  # Make room for the legend below

# Save plots
#print("Saving plots...")

fig.savefig("output/50_conference_comparison_all_2000_01_1associated.png", dpi=300, bbox_inches='tight')
fig.savefig("output/50_conference_comparison_all_2000_01_1associated.pdf", dpi=300, bbox_inches='tight')

#fig.savefig("output/network_comparison_three_01_1.eps", format='eps', bbox_inches='tight')
#fig

# Print summary statistics
print("\nFinal timestep statistics:")
final_stats = df[df['step'] == df['step'].max()].groupby('network').agg({
    'avg_violations': ['mean', 'std'],
    'min_violations': ['mean', 'std'],
    'homogeneity': ['mean', 'std']
})
print(final_stats)
    
# Save data
#df.to_csv("output/newmodel_network_comparison_results_all_01_10.csv", index=False)
