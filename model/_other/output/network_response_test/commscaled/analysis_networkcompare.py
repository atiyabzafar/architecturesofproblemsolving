import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
# Define the directory where your CSVs are stored
DATA_DIR = Path("./") # Change this if your CSVs are in a subfolder
file_mapping = {
    "Oscillatory (f=4)": "full_trajectories_oscillatory_4.csv",
    "Oscillatory (f=0.75)": "full_trajectories_oscillatory_75.csv",
    "Binary (p=0.05)": "full_trajectories_binary.csv",
    "Binary (p=0.5)": "full_trajectories_binary5.csv",
    "Binary (p=0.25)": "full_trajectories_binary25.csv",
    "Basic": "full_trajectories_basic.csv"
}

all_data = []

# --- 1. DATA LOADING ---
for name, filename in file_mapping.items():
    path = DATA_DIR / filename
    if path.exists():
        df = pd.read_csv(path)
        # Filter late window (steps > 5000)
        late = df[df["step"] > 5000].copy()
        late["Experiment"] = name
        all_data.append(late)
    else:
        print(f"Warning: File {filename} not found.")

if not all_data:
    print("Error: No files were loaded. Check DATA_DIR path.")
else:
    full_df = pd.concat(all_data)

    # --- 2. AGGREGATION ---
    # Group by Experiment and Network to calculate stats
    summary = full_df.groupby(["Experiment", "network"]).agg(
        mean_v=("avg_violations", "mean"),
        std_v=("avg_violations", "std"),
        mean_h=("homogeneity", "mean"),
        std_h=("homogeneity", "std")
    ).reset_index()

    # --- 3. PLOTTING ---
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Violations Plot
    sns.pointplot(data=summary, x="Experiment", y="mean_v", hue="network", 
                  dodge=0.3, join=False, markers=["o", "s"], capsize=.1, ax=axes[0])
    
    # Add manual error bars
    for i, (net, grp) in enumerate(summary.groupby("network")):
        axes[0].errorbar(x=np.arange(len(grp)) + (0.15 if net == "Scale Free" else -0.15), 
                         y=grp["mean_v"], yerr=grp["std_v"], fmt='none', ecolor='black', capsize=4, alpha=0.3)

    axes[0].set_title("Late-time Average Violations")
    axes[0].set_ylabel("Avg Violations")
    axes[0].tick_params(axis='x', rotation=45)

    # Homogeneity Plot
    sns.pointplot(data=summary, x="Experiment", y="mean_h", hue="network", 
                  dodge=0.3, join=False, markers=["o", "s"], capsize=.1, ax=axes[1])
    
    # Add manual error bars
    for i, (net, grp) in enumerate(summary.groupby("network")):
        axes[1].errorbar(x=np.arange(len(grp)) + (0.15 if net == "Scale Free" else -0.15), 
                         y=grp["mean_h"], yerr=grp["std_h"], fmt='none', ecolor='black', capsize=4, alpha=0.3)

    axes[1].set_title("Late-time Homogeneity")
    axes[1].set_ylabel("Homogeneity")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("network_comparison_summary.png")
    print("Plot saved as 'network_comparison_summary.png'")