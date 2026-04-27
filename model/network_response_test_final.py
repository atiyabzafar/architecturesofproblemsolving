"""
network_response_test.py
========================
A controlled test of how network topology affects collective adaptation
to an oscillating clause environment.

Scientific question
-------------------
Does network topology change (a) the *amplitude* of violation oscillations,
(b) the *phase lag* of the response relative to the AND-bias forcing, and
(c) steady-state *homogeneity* — when agents only have partial direct access
to the global clause pool?

Design
------
* Two network types: Scale-Free (SF) vs density-matched Random (RD).
* local_obs_fraction sweep: [0.1, 0.3, 0.5, 1.0]
  - At 1.0 → every agent sees all of C directly (old behaviour, no topology effect expected)
  - At lower values → agents must acquire foreign clauses via communication,
    so topology starts to matter.
* commscale = 1.0 throughout (NOT re-normalised), so denser networks
  genuinely deliver more clauses per step.
* Static control condition (clause_interval=0) alongside dynamic (clause_interval=1)
  to separate oscillation from topology effect.
* Metrics extracted:
  - violation_mean, violation_std  (cycle-average and amplitude proxy)
  - response_amplitude             (half peak-to-peak of phase-binned violations)
  - phase_lag                      (phase of violation minimum relative to AND-bias minimum)
  - homogeneity_mean
  - kb_fill_rate                   (mean fraction of kb capacity actually used, as sanity check)

Usage
-----
    python network_response_test.py

Outputs (all in output/network_response_test/)
-----------------------------------------------
  full_trajectories.csv   — raw per-step data
  late_summary.csv        — per-(network, obs_fraction, condition, seed) summary
  sweep_summary.csv       — aggregated over seeds, one row per (network, obs_fraction, condition)
  amplitude_heatmap.png   — amplitude vs obs_fraction x network, static vs dynamic
  phaselag_plot.png       — phase lag vs obs_fraction
  homogeneity_plot.png    — homogeneity vs obs_fraction
  violations_timeseries.png — representative timeseries at obs_fraction=0.3
"""

import os
import math
import random
import importlib.util
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

# ── Load model from file ──────────────────────────────────────────────────────
MODEL_PATH = "model_2026_04_24.py"
spec = importlib.util.spec_from_file_location("dynamic_model", MODEL_PATH)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ProblemSolvingModel = mod.ProblemSolvingModel

# Patch in the sinusoidal biased clause replacement (same as quick-dynamic-clause-test)
def _clause_probability(self):
    self.ANDbias = 1.0 - np.sin(self.sinfreq * self.steps * 2.0)
    self.XORbias = 1.0 - self.ANDbias

def _random_clause_biased(self):
    indices = random.sample(range(1, self.K + 1), 2)
    self._clause_probability()
    operator = "AND" if random.random() < self.ANDbias else "XOR"
    clause = (operator, tuple(indices))
    return self.canonicalise_clause(clause)

ProblemSolvingModel._clause_probability  = _clause_probability
ProblemSolvingModel._random_clause_biased = _random_clause_biased

# Patch replace_universal_clause to use biased version
def _replace_universal_clause(self):
    u = random.randint(0, self.M - 1)
    self.C[u] = self._random_clause_biased()

ProblemSolvingModel.replace_universal_clause = _replace_universal_clause

# ── Experiment parameters ─────────────────────────────────────────────────────
N             = 100
K             = 50
ALPHA         = 2.0
OBS_PROB      = 0.01
T_STEPS       = 8000        # total steps; last LATE_WINDOW used for analysis
LATE_WINDOW   = 3000        # steps used for late-time statistics
N_SEEDS       = 25
SEEDS         = list(range(100, 100 + N_SEEDS))
N_PROCS       = min(os.cpu_count() or 2, 16)
KB_FRACTION   = 0.2
SIN_FREQ      = 0.75        # matches model default

SF_MIN_DEG    = 3
TARGET_EDGES  = SF_MIN_DEG * (N - SF_MIN_DEG)
RAND_P        = TARGET_EDGES / (N * (N - 1))

# The key sweep variable
OBS_FRACTIONS = [0.1, 0.3, 0.5, 1.0]

# Conditions: dynamic clause pool vs static (control)
CONDITIONS = {
    "dynamic": {"clause_interval": 1},
    "static":  {"clause_interval": 100000},   # no clause replacement → fixed landscape
}

NETWORK_CONFIGS = {
    "Scale Free": {"type_network": "Scale Free", "min_deg": SF_MIN_DEG},
    "Random":     {"type_network": "Random",     "connect_prob": RAND_P},
}

OUTDIR = Path("output/network_response_test")
OUTDIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"Random": "#01696f", "Scale Free": "#964219"}

# ── Worker ────────────────────────────────────────────────────────────────────
def run_simulation(args):
    label, net_kwargs, obs_frac, condition_name, cond_kwargs, seed = args
    try:
        model = ProblemSolvingModel(
            N=N, K=K, alpha=ALPHA,
            obs_prob=OBS_PROB,
            R=T_STEPS,
            setup_source="generate",
            seed=seed,
            kb_fraction=KB_FRACTION,
            local_obs_fraction=obs_frac,
            **net_kwargs,
            **cond_kwargs,
        )
        # Force commscale = 1.0 (do NOT normalise by inflow)
        model.commscale = 1.0

        rows = []
        for _ in range(T_STEPS):
            model.step()
            step     = int(model.steps)
            and_bias = 1.0 - np.sin(SIN_FREQ * step * 2.0)
            kb_sizes = [len(a.kb) for a in model.agents]
            rows.append({
                "step":          step,
                "network":       label,
                "obs_fraction":  obs_frac,
                "condition":     condition_name,
                "seed":          seed,
                "avg_violations": float(model.avg_true_V),
                "homogeneity":   float(model.homogeneity),
                "AND_bias":      float(np.clip(and_bias, 0, 1)),
                "kb_fill":       float(np.mean(kb_sizes) / max(model.kb_capacity, 1)),
            })
        return rows
    except Exception as e:
        print(f"[ERROR] {label} obs={obs_frac} cond={condition_name} seed={seed}: {e}")
        import traceback; traceback.print_exc()
        return []

# ── Phase-lag and amplitude extraction ───────────────────────────────────────
def fit_sinusoid(phase_centers, values):
    """
    Given phase-binned violation values and their phase centres,
    fit V(phi) = A*sin(phi + delta) + C and return amplitude A and phase lag delta.
    The AND-bias signal is 1 - sin(phi), so its minimum is at phi = pi/2.
    Phase lag is defined as (phi_Vmin - pi/2) mod 2pi.
    """
    if len(phase_centers) < 4 or np.std(values) < 1e-9:
        return np.nan, np.nan
    # Fourier method: project onto sin and cos
    phi = np.array(phase_centers)
    v   = np.array(values) - np.mean(values)
    a1  = 2 * np.mean(v * np.sin(phi))
    b1  = 2 * np.mean(v * np.cos(phi))
    amplitude = np.sqrt(a1**2 + b1**2)
    # phase of violation signal: phi_V where V = amplitude * sin(phi + phi_V)
    phi_V = np.arctan2(b1, a1)
    # AND-bias min is at phi = pi/2; phase lag = phi_V - (-pi/2) rescaled
    # but more intuitively: find phase of V minimum
    phi_Vmin = (phi_V + np.pi) % (2 * np.pi)   # sin minimum is at phi + phi_V = -pi/2
    # AND-bias minimum is at phi = 3*pi/2 (since bias = 1 - sin(phi), min sin = 1)
    and_min_phase = 3 * np.pi / 2
    phase_lag = (phi_Vmin - and_min_phase) % (2 * np.pi)
    return amplitude, phase_lag

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("Network topology response test")
    print(f"N={N}, K={K}, α={ALPHA}, obs_prob={OBS_PROB}")
    print(f"T={T_STEPS}, late_window={LATE_WINDOW}, seeds={N_SEEDS}")
    print(f"obs_fractions={OBS_FRACTIONS}")
    print(f"SF min_deg={SF_MIN_DEG}  |  Random p={RAND_P:.4f} (density-matched)")
    print(f"commscale fixed at 1.0 (no normalisation)")
    print("=" * 72)

    # Build job list
    jobs = [
        (label, net_kw, obs_frac, cond_name, cond_kw, seed)
        for label,     net_kw    in NETWORK_CONFIGS.items()
        for obs_frac              in OBS_FRACTIONS
        for cond_name, cond_kw   in CONDITIONS.items()
        for seed                  in SEEDS
    ]
    print(f"Total jobs: {len(jobs)}")

    with Pool(processes=N_PROCS) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_simulation, jobs),
            total=len(jobs), desc="Simulating"
        ))

    df = pd.DataFrame([r for sub in results for r in sub])
    if df.empty:
        raise RuntimeError("No simulation rows produced — check earlier errors.")

    csv_path = OUTDIR / "full_trajectories.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull data → {csv_path}  ({len(df):,} rows)")

    # ── Late-window statistics ────────────────────────────────────────────────
    late = df[df["step"] > T_STEPS - LATE_WINDOW].copy()

    # Phase binning within the late window
    N_BINS = 24
    late = late.copy()
    late["phase"] = np.mod(SIN_FREQ * late["step"], 2 * np.pi)
    late["phase_bin"] = pd.cut(
        late["phase"],
        bins=np.linspace(0, 2 * np.pi, N_BINS + 1),
        include_lowest=True,
    )

    # Per-seed late summary
    summary_rows = []
    for (network, obs_frac, condition, seed), g in late.groupby(
        ["network", "obs_fraction", "condition", "seed"]
    ):
        summary_rows.append({
            "network":         network,
            "obs_fraction":    obs_frac,
            "condition":       condition,
            "seed":            seed,
            "violation_mean":  g["avg_violations"].mean(),
            "violation_std":   g["avg_violations"].std(),
            "homogeneity_mean": g["homogeneity"].mean(),
            "kb_fill_mean":    g["kb_fill"].mean(),
        })
    late_summary = pd.DataFrame(summary_rows)
    late_summary.to_csv(OUTDIR / "late_summary.csv", index=False)

    # Phase-binned data aggregated over seeds
    phase_agg = (
        late.groupby(["network", "obs_fraction", "condition", "phase_bin"], observed=False)
        .agg(
            phase_center=("phase", "mean"),
            avg_violations=("avg_violations", "mean"),
            homogeneity=("homogeneity", "mean"),
            AND_bias=("AND_bias", "mean"),
        )
        .reset_index()
    )

    # Amplitude and phase lag per (network, obs_fraction, condition)
    sweep_rows = []
    for (network, obs_frac, condition), g in phase_agg.groupby(
        ["network", "obs_fraction", "condition"]
    ):
        g = g.dropna(subset=["phase_center", "avg_violations"])
        amp, lag = fit_sinusoid(g["phase_center"].values, g["avg_violations"].values)
        base = late_summary[
            (late_summary["network"] == network) &
            (late_summary["obs_fraction"] == obs_frac) &
            (late_summary["condition"] == condition)
        ]
        sweep_rows.append({
            "network":          network,
            "obs_fraction":     obs_frac,
            "condition":        condition,
            "violation_mean":   base["violation_mean"].mean(),
            "violation_std":    base["violation_std"].mean(),
            "response_amplitude": amp,
            "phase_lag_rad":    lag,
            "phase_lag_steps":  lag / (SIN_FREQ * 2.0) if not np.isnan(lag) else np.nan,
            "homogeneity_mean": base["homogeneity_mean"].mean(),
            "kb_fill_mean":     base["kb_fill_mean"].mean(),
        })

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUTDIR / "sweep_summary.csv", index=False)

    print("\nSweep summary (dynamic condition):")
    print(sweep[sweep["condition"] == "dynamic"].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Amplitude vs obs_fraction — dynamic vs static side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, cond in zip(axes, ["dynamic", "static"]):
        sub = sweep[sweep["condition"] == cond]
        for net, grp in sub.groupby("network"):
            ax.plot(
                grp["obs_fraction"], grp["response_amplitude"],
                marker="o", label=net, color=PALETTE[net], linewidth=2
            )
        ax.set_title(f"Response amplitude — {cond} clause pool")
        ax.set_xlabel("Local observation fraction")
        ax.set_ylabel("Violation amplitude (half peak-to-peak)")
        ax.legend()
    fig.suptitle("How network topology and observation scope affect response amplitude", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTDIR / "amplitude_plot.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Phase lag vs obs_fraction (dynamic only)
    fig, ax = plt.subplots(figsize=(8, 5))
    dyn = sweep[sweep["condition"] == "dynamic"]
    for net, grp in dyn.groupby("network"):
        ax.plot(
            grp["obs_fraction"], grp["phase_lag_steps"],
            marker="o", label=net, color=PALETTE[net], linewidth=2
        )
    ax.set_title("Phase lag of violation response vs AND-bias forcing")
    ax.set_xlabel("Local observation fraction")
    ax.set_ylabel("Phase lag (steps)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "phase_lag_plot.png", dpi=200)
    plt.close()

    # 3. Homogeneity vs obs_fraction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cond in zip(axes, ["dynamic", "static"]):
        sub = sweep[sweep["condition"] == cond]
        for net, grp in sub.groupby("network"):
            ax.plot(
                grp["obs_fraction"], grp["homogeneity_mean"],
                marker="o", label=net, color=PALETTE[net], linewidth=2
            )
        ax.set_title(f"Homogeneity — {cond}")
        ax.set_xlabel("Local observation fraction")
        ax.set_ylabel("Mean homogeneity")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "homogeneity_plot.png", dpi=200)
    plt.close()

    # 4. Violation mean vs obs_fraction
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cond in zip(axes, ["dynamic", "static"]):
        sub = sweep[sweep["condition"] == cond]
        for net, grp in sub.groupby("network"):
            ax.plot(
                grp["obs_fraction"], grp["violation_mean"],
                marker="o", label=net, color=PALETTE[net], linewidth=2
            )
        ax.set_title(f"Cycle-average violations — {cond}")
        ax.set_xlabel("Local observation fraction")
        ax.set_ylabel("Mean avg violations")
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "violations_mean_plot.png", dpi=200)
    plt.close()

    # 5. Representative timeseries at obs_fraction = 0.3, dynamic condition
    REP_FRAC = 0.3
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    traj = (
        df[(df["obs_fraction"] == REP_FRAC) & (df["condition"] == "dynamic")]
        .groupby(["network", "step"])
        .agg(
            avg_violations=("avg_violations", "mean"),
            homogeneity=("homogeneity", "mean"),
            kb_fill=("kb_fill", "mean"),
            AND_bias=("AND_bias", "mean"),
        )
        .reset_index()
    )
    for net, grp in traj.groupby("network"):
        axes[0].plot(grp["step"], grp["avg_violations"],
                     label=net, color=PALETTE[net], linewidth=1.5)
    ax2 = axes[0].twinx()
    ref = traj[traj["network"] == "Random"].sort_values("step")
    ax2.plot(ref["step"], ref["AND_bias"], color="gray", alpha=0.3,
             linestyle="--", linewidth=1, label="AND bias")
    axes[0].set_ylabel("Avg violations")
    axes[0].set_title(f"Timeseries (obs_fraction={REP_FRAC}, dynamic)")
    axes[0].legend(loc="upper left")

    for net, grp in traj.groupby("network"):
        axes[1].plot(grp["step"], grp["homogeneity"],
                     label=net, color=PALETTE[net], linewidth=1.5)
    axes[1].set_ylabel("Homogeneity")
    axes[1].legend(loc="upper left")

    for net, grp in traj.groupby("network"):
        axes[2].plot(grp["step"], grp["kb_fill"],
                     label=net, color=PALETTE[net], linewidth=1.5)
    axes[2].set_ylabel("KB fill rate")
    axes[2].set_xlabel("Step")
    axes[2].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(OUTDIR / "violations_timeseries.png", dpi=200)
    plt.close()

    print(f"\nAll outputs saved to {OUTDIR}/")
    print("\nInterpretation guide:")
    print("  amplitude_plot   : if SF < Random → SF dampens oscillations (better tracking)")
    print("  phase_lag_plot   : larger lag → slower adaptation to env change")
    print("  homogeneity_plot : higher = agents more aligned")
    print("  violations_mean  : lower = better collective performance")
    print("  kb_fill_timeseries: if << 1.0 at low obs_fraction, knowledge is genuinely scarce")
    print("  At obs_fraction=1.0 both networks should converge to same metrics (sanity check)")
