"""
Policy distribution over time -- sanity check on the simple mean-field prediction.

The simple mean-field predicts:
  - p(t) := mean over (agent, variable) of x_j == 1   ->   1 in the long run
  - Every variable j converges so that f_j := fraction of agents with x_j=1   ->   1
  - Homogeneity H -> 1

In simulation we know this isn't quite right -- H saturates around 0.7-0.9. This
script runs ONE long simulation and records the per-variable f_j(t) so we can see
exactly which variables make it to f_j = 1 and which stick at 0 or in between.

Outputs in output/policy_distribution/:
    p_and_V_over_time.png    -- p(t) and avg_V(t) over time, with MF references
    per_variable_heatmap.png -- f_j(t) for every variable, time on x-axis
    final_f_distribution.png -- bar chart of f_j across variables at the final tick
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from model import ProblemSolvingModel


# ============================================================
# Parameters
# ============================================================
N               = 80
K               = 30
ALPHA           = 2          # M = alpha * K = 60 clauses
OBS_PROB        = 0.01
CLAUSE_INTERVAL = 10
T               = 10_000
SEED            = 42

# Standard random network at matched density
NETWORK = dict(type_network="Random", connect_prob=0.10)


# ============================================================
# Run
# ============================================================
m = ProblemSolvingModel(
    N=N, K=K, alpha=ALPHA, obs_prob=OBS_PROB, clause_interval=CLAUSE_INTERVAL,
    setup_source="generate", R=T, seed=SEED, **NETWORK,
)

# Storage
f_history    = np.zeros((T, K))    # f_history[t, j] = fraction of agents with x_j=1 at tick t+1
p_history    = np.zeros(T)         # overall fraction of 1's at tick t+1
avgV_history = np.zeros(T)
H_history    = np.zeros(T)

print(f"Running {T} ticks on {NETWORK['type_network']} network "
      f"(N={N}, K={K}, alpha={ALPHA}, M={ALPHA*K})...", flush=True)

for t in range(T):
    m.step()
    states = np.array([a.x for a in m.agents])   # shape (N, K)
    f_history[t]    = states.mean(axis=0)         # per-variable fraction
    p_history[t]    = states.mean()               # global fraction
    avgV_history[t] = m.avg_true_V
    H_history[t]    = m.homogeneity
    if (t + 1) % 1000 == 0:
        print(f"  tick {t+1:5d}:  p={p_history[t]:.3f}  "
              f"avg_V={avgV_history[t]:6.2f}  H={H_history[t]:.3f}",
              flush=True)


# ============================================================
# Output folder
# ============================================================
out = Path(__file__).parent / "output" / "policy_distribution"
out.mkdir(parents=True, exist_ok=True)


# ============================================================
# Plot 1: p(t) and V(t) over time
# ============================================================
M = round(ALPHA * K)
ticks = np.arange(1, T + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(ticks, p_history, color="tab:blue", lw=1)
axes[0].axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="MF prediction (p=1)")
axes[0].axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
axes[0].set_xlabel("tick")
axes[0].set_ylabel("p(t) = fraction of variables = 1")
axes[0].set_title("Mean fraction of 1s over time")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(ticks, avgV_history, color="tab:orange", lw=1, label="avg V")
axes[1].axhline(M / 2, color="k", ls="--", lw=1, alpha=0.5,
                label=f"MF attractor (alpha*K/2 = {M // 2})")
axes[1].set_xlabel("tick")
axes[1].set_ylabel("violations")
axes[1].set_title("Average violations over time")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out / "p_and_V_over_time.png", dpi=140)
plt.close(fig)


# ============================================================
# Plot 2: per-variable f_j(t) heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(f_history.T, aspect="auto", origin="lower", cmap="RdBu_r",
               vmin=0, vmax=1, extent=[1, T, 0.5, K + 0.5])
ax.set_xlabel("tick")
ax.set_ylabel("variable index j")
ax.set_title("Per-variable f_j(t) -- fraction of agents with x_j = 1\n"
             "(red = 1, blue = 0; uniform red = MF prediction)")
fig.colorbar(im, ax=ax, label="f_j(t)")
fig.tight_layout()
fig.savefig(out / "per_variable_heatmap.png", dpi=140)
plt.close(fig)


# ============================================================
# Plot 3: final-state f_j distribution
# ============================================================
final_f = f_history[-1]
sorted_idx = np.argsort(final_f)
sorted_f   = final_f[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["tab:blue" if v >= 0.5 else "tab:orange" for v in sorted_f]
ax.bar(range(1, K + 1), sorted_f, color=colors)
ax.axhline(0.5, color="k", ls=":", lw=1, alpha=0.5)
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5, label="MF prediction (f_j = 1)")
ax.set_xlabel("variable rank (sorted ascending by f_j)")
ax.set_ylabel("f_j at final tick")
ax.set_title(f"Final-state f_j across {K} variables (tick {T})")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(out / "final_f_distribution.png", dpi=140)
plt.close(fig)


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print(f"FINAL STATE at tick {T}")
print("=" * 60)
print(f"  Overall p     = {p_history[-1]:.3f}   "
      f"(MF prediction: 1)")
print(f"  Avg V         = {avgV_history[-1]:6.2f}   "
      f"(MF attractor: alpha*K/2 = {M//2})")
print(f"  Homogeneity   = {H_history[-1]:.3f}")
print()
print(f"  Per-variable f_j summary (across {K} variables):")
print(f"    min  = {final_f.min():.3f}")
print(f"    mean = {final_f.mean():.3f}")
print(f"    max  = {final_f.max():.3f}")
print(f"    # variables with f_j >= 0.95          : {(final_f >= 0.95).sum():>3} / {K}")
print(f"    # variables with f_j <= 0.05          : {(final_f <= 0.05).sum():>3} / {K}")
print(f"    # variables with 0.05 < f_j < 0.95    : "
      f"{((final_f > 0.05) & (final_f < 0.95)).sum():>3} / {K}")
print()
print(f"Plots saved under: {out}")
