import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIGURATION ---
# UPDATE THIS PATH to the directory containing your CSV files
BASE_PATH = Path("./") 

# Map labels to filenames
file_mapping = {
    "Oscillatory (f=4)": "full_trajectories_oscillatory_4.csv",
    "Oscillatory (f=0.75)": "full_trajectories_oscillatory_75.csv",
    "Binary (p=0.05)": "full_trajectories_binary.csv",
    "Binary (p=0.5)": "full_trajectories_binary5.csv",
    "Binary (p=0.25)": "full_trajectories_binary25.csv",
    "Basic": "full_trajectories_basic.csv"
}

# Analysis Settings
T_STEPS = 8000
LATE_WINDOW = 3000
START_STEP = T_STEPS - LATE_WINDOW

summary_data = []

# --- DATA PROCESSING ---
print(f"Looking for files in: {BASE_PATH.resolve()}")

for name, filename in file_mapping.items():
    path = BASE_PATH / filename
    if path.exists():
        print(f"Processing: {filename}")
        df = pd.read_csv(path)
        
        # Filter for late window
        late = df[df["step"] > START_STEP]
        
        # Calculate stats
        mean_v = late["avg_violations"].mean()
        mean_h = late["homogeneity"].mean()
        
        summary_data.append({
            "Experiment": name,
            "Avg Violations": mean_v,
            "Homogeneity": mean_h
        })
    else:
        print(f"Warning: File not found at {path}")

# Create DataFrame
summary_df = pd.DataFrame(summary_data)

if summary_df.empty:
    print("Error: No data loaded. Please check the BASE_PATH variable.")
else:
    print("\nCalculated Summary:")
    print(summary_df)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Avg Violations
    sns.scatterplot(data=summary_df, x="Experiment", y="Avg Violations", ax=axes[0], s=200, color='teal')
    axes[0].set_title(f"Late-time Average Violations (Steps > {START_STEP})")
    axes[0].tick_params(axis='x', rotation=45)

    # Plot Homogeneity
    sns.scatterplot(data=summary_df, x="Experiment", y="Homogeneity", ax=axes[1], s=200, color='brown')
    axes[1].set_title(f"Late-time Homogeneity (Steps > {START_STEP})")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("late_comparison_scatter.png")
    print("\nSuccess: Plot saved as 'late_comparison_scatter.png'")