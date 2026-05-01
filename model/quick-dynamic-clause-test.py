# quick_dynamic_clause_test.py
import os
import math
import random
import importlib.util
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# Load the attached model file with the hyphen in its filename
# ============================================================
MODEL_PATH = "model_2026_04_24.py"

spec = importlib.util.spec_from_file_location("dynamic_model", MODEL_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ProblemSolvingModel = mod.ProblemSolvingModel

# ============================================================
# Patch the two issues in the attached file for this test only
# ============================================================
def clause_probability(self):
    self.ANDbias = (1.0 - np.sin(self.sinfreq * self.steps)) / 2.0
    self.XORbias = 1.0 - self.ANDbias

def random_clause_biased(self):
    indices = random.sample(range(1, self.K + 1), 2)
    self.clause_probability()  # <- actually call it
    operator = "AND" if random.random() < self.ANDbias else "XOR"  # <- use ANDbias for AND
    clause = (operator, tuple(indices))
    return self.canonicalise_clause(clause)

ProblemSolvingModel.clause_probability = clause_probability
ProblemSolvingModel.random_clause_biased = random_clause_biased

# ============================================================
# Configuration
# ============================================================
N = 100
K = 50
ALPHA = 2.0
OBS_PROB = 0.01
CLAUSE_INTERVAL = 1
T_STEPS = 10000
N_SEEDS = 10
SEEDS = range(100, 100 + 5)
N_PROCS = min(os.cpu_count() or 2, 16)
kb_fraction=1.0

# Match random density to scale-free baseline, following your earlier comparison style
SF_MIN_DEG = 3
TARGET_EDGES = SF_MIN_DEG * (N - SF_MIN_DEG)
RAND_P = TARGET_EDGES / (N * (N - 1))

CONFIGS = {
    "Random": {
        "type_network": "Random",
        "connect_prob": RAND_P,
    },
    "Scale Free": {
        "type_network": "Scale Free",
        "min_deg": SF_MIN_DEG,
    },
}

OUTDIR = Path("output/quick_dynamic_clause_test")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Worker
# ============================================================
def run_simulation(args):
    label, kwargs, seed = args
    try:
        model = ProblemSolvingModel(
            N=N,
            K=K,
            alpha=ALPHA,
            obs_prob=OBS_PROB,
            clause_interval=CLAUSE_INTERVAL,
            R=T_STEPS,
            setup_source="generate",
            seed=seed,
            kb_fraction=kb_fraction,
            **kwargs
        )

        rows = []
        for _ in range(T_STEPS):
            model.step()
            step = int(model.steps)
            and_bias = (1.0 - np.sin(model.sinfreq * step)) / 2.0

            rows.append({
                "step": step,
                "network": label,
                "seed": seed,
                "avg_violations": float(model.avg_true_V),
                "min_violations": float(model.min_true_V),
                "homogeneity": float(model.homogeneity),
                "AND_bias": float(and_bias),
                "XOR_bias": float(1.0 - and_bias),
            })
        return rows

    except Exception as e:
        print(f"[ERROR] {label} seed={seed}: {e}")
        return []

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("Dynamic clause-pool test: Random vs Scale Free")
    print(f"N={N}, K={K}, alpha={ALPHA}, obs_prob={OBS_PROB}, clause_interval={CLAUSE_INTERVAL}, kb_fraction={kb_fraction}")
    print(f"T={T_STEPS}, seeds={N_SEEDS}, SF min_deg={SF_MIN_DEG}, Random p={RAND_P:.4f}")
    print("=" * 72)

    jobs = [
        (label, kwargs, seed)
        for label, kwargs in CONFIGS.items()
        for seed in SEEDS
    ]

    with Pool(processes=N_PROCS) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_simulation, jobs),
            total=len(jobs),
            desc="Simulating"
        ))

    df = pd.DataFrame([r for sub in results for r in sub])
    if df.empty:
        raise RuntimeError("No simulation rows produced.")

    # Save full trajectories
    csv_path = OUTDIR / "dynamic_clause_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full data to: {csv_path}")

    # --------------------------------------------------------
    # Late-window diagnostics
    # --------------------------------------------------------
    LATE_WINDOW = 2000
    late = df[df["step"] > T_STEPS - LATE_WINDOW].copy()

    def slope_of_last_window(g, ycol):
        x = g["step"].to_numpy()
        y = g[ycol].to_numpy()
        if len(x) < 2:
            return np.nan
        return np.polyfit(x, y, 1)[0]

    summary_rows = []
    for (network, seed), g in late.groupby(["network", "seed"]):
        summary_rows.append({
            "network": network,
            "seed": seed,
            "late_mean_avgV": g["avg_violations"].mean(),
            "late_std_avgV": g["avg_violations"].std(),
            "late_slope_avgV": slope_of_last_window(g, "avg_violations"),
            "late_mean_hom": g["homogeneity"].mean(),
            "late_std_hom": g["homogeneity"].std(),
            "late_slope_hom": slope_of_last_window(g, "homogeneity"),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "late_window_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    print("\nLate-window summary by network")
    print(summary.groupby("network").agg(["mean", "std"]))

    # --------------------------------------------------------
    # Phase-binned response
    # --------------------------------------------------------
    late["phase"] = np.mod(0.75 * late["step"], 2 * np.pi)
    n_bins = 24
    late["phase_bin"] = pd.cut(late["phase"], bins=np.linspace(0, 2*np.pi, n_bins + 1), include_lowest=True)

    phase_df = (
        late.groupby(["network", "phase_bin"], observed=False)
        .agg(
            phase_center=("phase", "mean"),
            avg_violations=("avg_violations", "mean"),
            homogeneity=("homogeneity", "mean"),
            AND_bias=("AND_bias", "mean"),
        )
        .reset_index()
    )
    phase_path = OUTDIR / "phase_binned_response.csv"
    phase_df.to_csv(phase_path, index=False)
    print(f"Saved phase-binned response to: {phase_path}")

    # --------------------------------------------------------
    # Plot 1: long-run trajectories
    # --------------------------------------------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    traj = (
        df.groupby(["network", "step"])
        .agg(
            avg_violations=("avg_violations", "mean"),
            homogeneity=("homogeneity", "mean"),
            AND_bias=("AND_bias", "mean"),
        )
        .reset_index()
    )

    sns.lineplot(data=traj, x="step", y="avg_violations", hue="network", ax=axes[0], linewidth=2)
    ax2 = axes[0].twinx()
    ax2.plot(traj[traj["network"] == "Random"]["step"],
             traj[traj["network"] == "Random"]["AND_bias"],
             color="black", alpha=0.25, linestyle="--", label="AND bias")
    axes[0].set_title("Average violations under oscillating clause pool")
    axes[0].set_ylabel("Avg violations")

    sns.lineplot(data=traj, x="step", y="homogeneity", hue="network", ax=axes[1], linewidth=2, legend=False)
    axes[1].set_title("Homogeneity under oscillating clause pool")
    axes[1].set_ylabel("Homogeneity")
    axes[1].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig(OUTDIR / "dynamic_clause_timeseries.png", dpi=300)
    plt.close()

    # --------------------------------------------------------
    # Plot 2: phase-binned late-time response
    # --------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    sns.lineplot(data=phase_df, x="phase_center", y="avg_violations", hue="network", ax=axes[0], linewidth=2)
    ax2 = axes[0].twinx()
    phase_sorted = phase_df[phase_df["network"] == "Random"].sort_values("phase_center")
    ax2.plot(phase_sorted["phase_center"], phase_sorted["AND_bias"],
             color="black", alpha=0.3, linestyle="--")
    axes[0].set_title("Late-time phase response: avg violations")
    axes[0].set_ylabel("Avg violations")

    sns.lineplot(data=phase_df, x="phase_center", y="homogeneity", hue="network", ax=axes[1], linewidth=2, legend=False)
    axes[1].set_title("Late-time phase response: homogeneity")
    axes[1].set_ylabel("Homogeneity")
    axes[1].set_xlabel("Phase")

    plt.tight_layout()
    plt.savefig(OUTDIR / "dynamic_clause_phase_response.png", dpi=300)
    plt.close()

    print("\nDone.")
    print("Interpretation guide:")
    print("- If late_slope_avgV is near 0 and late_std_avgV is tiny, you have near-fixed-point behavior.")
    print("- If late_slope_avgV is near 0 but late_std_avgV stays nonzero and the phase plot is structured,")
    print("  you have a periodic driven regime rather than convergence to one constant value.")
    print("- Compare late_mean_avgV across networks to test whether they still share the same cycle-average.")