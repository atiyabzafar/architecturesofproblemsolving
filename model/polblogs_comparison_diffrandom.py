from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from model_new_1 import ProblemSolvingModel
"""

def run_network_simulation(params):
    #Run a single simulation for a network dataset
    filepath, seed = params
    
    try:
        model = ProblemSolvingModel(
            K=50,
            alpha=2.0,
            obs_prob=0.01,
            clause_interval=10,
            R=500,
            setup_source="dataset",
            file_path=filepath,
            seed=seed
        )
        
        # Store metrics at each timestep
        metrics = []
        for _ in range(500):
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
    "data/PolBlogsGiant.xml",
    "data/PolBlogsGiant_random_trad_0.1.graphml",
    "data/PolBlogsGiant_random_0.1.xml",
]

network_names = {
    "PolBlogsGiant.xml": "Political Blogs",
    "PolBlogsGiant_random_trad_0.1.graphml" : "Political Blogs Randomised",
    "PolBlogsGiant_random_0.1.xml" : "Political Blogs Randomised (hierarchy)",
}

seeds = range(42, 42+10)  # 10 seeds per network
params = [(f, s) for f in graphfiles for s in seeds]

# Run simulations in parallel
print("Running parallel simulations...")
with Pool(processes=20) as pool:
    results = list(tqdm(pool.imap(run_network_simulation, params), total=len(params)))

# Flatten results
all_results = [item for sublist in results for item in sublist]
df = pd.DataFrame(all_results)
"""
df=pd.read_csv("output/diffrandomPolitical_Blogs_01_10.csv")
graphfiles = [
    "data/PolBlogsGiant.xml",
    "data/PolBlogsGiant_random_trad_0.1.graphml",
    "data/PolBlogsGiant_random_0.1.xml",
]

network_names = {
    "PolBlogsGiant.xml": "Political Blogs",
    "PolBlogsGiant_random_trad_0.1.graphml" : "Political Blogs Randomised",
    "PolBlogsGiant_random_0.1.xml" : "Political Blogs Randomised (hierarchy)",
}
# Replace filenames with readable names
df['network'] = df['network'].map(network_names)



import scienceplots
plt.style.use(['science','nature'])

# Create plots
metrics = ['avg_violations', 'homogeneity']
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
fig.savefig("output/diffrandomPolitical_Blogs_01_10.png", dpi=300, bbox_inches='tight')
fig.savefig("output/diffrandomPolitical_Blogs_01_10.pdf", dpi=300, bbox_inches='tight')

#fig.savefig("output/network_comparison_three_01_1.eps", format='eps', bbox_inches='tight')
fig

# Print summary statistics
print("\nFinal timestep statistics:")
final_stats = df[df['step'] == df['step'].max()].groupby('network').agg({
    'avg_violations': ['mean', 'std'],
    'min_violations': ['mean', 'std'],
    'homogeneity': ['mean', 'std']
})
print(final_stats)
    
# Save data
#df.to_csv("output/diffrandomPolitical_Blogs_01_10.csv", index=False)
