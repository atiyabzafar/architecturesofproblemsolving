"""
Communication efficiency: SF vs Random, matched on edge count.

The question
------------
In the baseline model, Scale Free and Random networks converge to similar
long-run violation counts (see long_sweep_sf_vs_random.py). This script
asks a different question: given that both networks end up in the same
place, how much communication did each one need, and who did the work?

Edge-matched comparison
-----------------------
The baseline SF (min_deg=3) and Random (p=0.20) parameters in the rest of
the codebase give networks with very different numbers of edges (~220 vs
~1260 at N=80). Any labour-concentration difference between them could
therefore be a density artefact rather than a topology effect. We fix this
by choosing parameters that produce roughly the same edge count: SF
min_deg=9 (~616 edges) and Random connect_prob=0.10 (~624 edges). Both
networks thus have ~620 directed edges, so all cross-topology comparisons
are made at a fixed edge budget.

Two stories we want to tell:

  (1) EFFICIENCY: if Scale Free reaches the same violation level with fewer
      *total* successful elicitations across the whole organisation, then
      under a cost model where each communication is costly, SF is more
      efficient. Plot avg_V on the y-axis against cumulative successful
      elicitations on the x-axis: the lower curve is the more efficient one.

  (2) COMMUNICATION HUBS: at the end of the run, how are the
      elicitations distributed across agents? In Scale Free we expect hubs
      to carry a disproportionate share of the communication load; in
      Random we expect a roughly uniform distribution. Quantify with the
      Gini coefficient and visualise with Lorenz curves.

Together these support the combined story: if cost is proportional to
communication frequency, SF is efficient; if cost takes the form of
per-agent capacity caps, SF's hubs are bottlenecks and Random wins.
(The per-agent-cap case is already covered by sweep_comm_cost.py in
claude_investigations/.)

Mechanism
---------
We subclass Person into InstrumentedPerson which mirrors the stock step()
exactly but increments a counter each time a *communication-channel*
elicitation successfully adds a new clause to the agent's knowledge base.
Observation-channel additions are NOT counted (it's difficult to imagine
that in our case observation costs would tell us anything interesting).
The instrumented class is installed via monkey-patch so the rest of the
model pipeline stays untouched.

Outputs (all in output/comm_efficiency/):
    comm_efficiency_results.xlsx  -- per-seed summary, raw trajectories,
                                     per-agent elicit distributions
    pareto_avgV_vs_comms.png       -- the efficiency plot (main figure)
    lorenz_per_agent.png           -- Lorenz curves for cognitive labour
    per_agent_hist.png             -- histogram of per-agent elicit counts
    avgV_vs_time.png               -- classic time-series for reference

 (T=10,000 x 5 seeds x 2 networks= 10 runs).
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Monkey-patch requires importing the module so we can overwrite Person.
import model as _mm
from model import Person, ProblemSolvingModel


# ============================================================
# Step 1: Configuration
# ============================================================

T     = 10_000                              # ticks per run
SEEDS = [42, 43, 44, 45, 46]                # five seeds per (network)
BASE  = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
             setup_source="generate", R=T)

NETWORKS = [
    # MIDDLE-GROUND MATCHED: both produce ~620 directed edges at N=80.
    # Empirically: SF min_deg=9 -> mean 616 edges; Random p=0.10 -> mean 624.
    # (Sparse-matched alternative: SF min_deg=3 vs Random p=0.035, ~220 edges.)
    ("Scale Free", dict(type_network="Scale Free", min_deg=9)),
    ("Random",     dict(type_network="Random",     connect_prob=0.10)),
]

# Color conventions (kept consistent with long_sweep_sf_vs_random.py)
COLOR_SF   = "tab:blue"
COLOR_RAND = "tab:orange"


# ============================================================
# Step 2: Instrumented Person
# ============================================================
# Mirrors the stock Person.step() exactly -- same RNG calls in the same
# order, same add/local-update behaviour -- but increments a counter every
# time a COMMUNICATION-channel elicitation successfully transfers a new
# clause into self.kb. Observation-channel additions are NOT counted; they
# represent private observation of the environment, not social comms.

class InstrumentedPerson(Person):
    """Person that counts successful communication-channel elicitations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_elicits = 0               # cumulative across the run
        self._elicits_this_tick  = 0              # reset each tick

    def step(self):
        my_kb_set = set(self.kb)
        self._elicits_this_tick = 0

        # 1) Private observation -- NOT counted as communication
        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)

        # 2) Communication -- counted on every successful new-clause transfer
        in_neighbors = self.in_neighbors
        if in_neighbors:
            comm_scale = self.model.comm_scale
            for nbr_id in in_neighbors:
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get("weight", 1.0)
                link_probability = min(1.0, comm_scale * link_weight)

                if random.random() < link_probability:
                    nbr_agent = self.model.agent_list[nbr_id]
                    if nbr_agent.kb:
                        nbr_kb_set = set(nbr_agent.kb)
                        unknowns = nbr_kb_set - my_kb_set
                        if unknowns:
                            cprime = random.choice(list(unknowns))
                            self.add_clause_to_kb(cprime)
                            self.local_update_around(cprime)
                            my_kb_set.add(cprime)
                            self.successful_elicits  += 1
                            self._elicits_this_tick  += 1


# Install the patch BEFORE any model is built. setup_network() in
# ProblemSolvingModel references Person by name from the `model` module,
# so overwriting the module attribute is enough.
_mm.Person = InstrumentedPerson


# ============================================================
# Step 3: Single-run driver
# ============================================================

def run_one_instrumented(network_kwargs, seed):
    """Run one simulation for T ticks and return everything we need.

    Returns a dict with:
      avg_V                 -- length-T array of avg violations per tick
      min_V                 -- length-T array of min violations per tick
      elicits_per_tick      -- length-T array of network-wide successful
                               elicitations per tick
      per_agent_elicits     -- length-N array: total successful elicits
                               by each agent over the whole run
    """
    m = ProblemSolvingModel(seed=seed, **BASE, **network_kwargs)
    N = m.N

    avg_V             = np.zeros(T, dtype=float)
    min_V             = np.zeros(T, dtype=float)
    elicits_per_tick  = np.zeros(T, dtype=int)

    for t in range(T):
        m.step()
        avg_V[t] = m.avg_true_V
        min_V[t] = m.min_true_V
        # Sum successful comm elicitations across all agents for this tick
        elicits_per_tick[t] = sum(a._elicits_this_tick for a in m.agents)

    per_agent_elicits = np.array(
        [a.successful_elicits for a in m.agents], dtype=int
    )

    return {
        "avg_V":             avg_V,
        "min_V":             min_V,
        "elicits_per_tick":  elicits_per_tick,
        "per_agent_elicits": per_agent_elicits,
    }


# ============================================================
# Step 4: Concentration statistics
# ============================================================
# Gini coefficient and Lorenz curve over non-negative per-agent counts.

def gini(values):
    """Compute the Gini coefficient of a 1D array of non-negative values.
    Returns 0.0 for a perfectly uniform distribution, approaches 1.0 when
    one agent holds everything."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n


def lorenz_curve(values):
    """Compute the Lorenz curve.

    Returns (x, y) where x in [0,1] is the cumulative population share
    (sorted from least to most) and y in [0,1] is the cumulative share
    of elicitations they hold. Perfect equality = diagonal line."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return np.linspace(0, 1, max(n, 2)), np.linspace(0, 1, max(n, 2))
    cumulative = np.concatenate(([0.0], np.cumsum(arr))) / arr.sum()
    population = np.linspace(0.0, 1.0, n + 1)
    return population, cumulative


# ============================================================
# Step 5: Interpolate per-seed trajectories onto a common comms-axis
# ============================================================
# For the Pareto plot we want mean avg_V at each level of cumulative
# elicitations, not at each tick. Different runs accumulate comms at
# different rates, so we align them by resampling avg_V as a function
# of cumulative comms onto a common grid.

def interpolate_to_comm_axis(per_run, comm_grid):
    """Given a list of per-run result dicts (all same T), return an
    (n_runs, len(comm_grid)) matrix of avg_V values, linearly interpolated
    against cumulative elicits.

    If a given run never reaches the higher comm_grid values, those
    entries are set to NaN so the mean across runs can ignore them."""
    n_runs = len(per_run)
    out = np.full((n_runs, len(comm_grid)), np.nan, dtype=float)
    for i, r in enumerate(per_run):
        cumul = np.cumsum(r["elicits_per_tick"])
        avgV  = r["avg_V"]
        # np.interp requires the x-values to be increasing; cumsum of
        # non-negative ints is non-decreasing, and strictly increasing
        # except when elicits_per_tick[t]==0. Feed it directly; duplicate
        # x-values are handled by picking the first matching y.
        # We set "right" to NaN so points past the end of this run's comms
        # don't extrapolate.
        mask = comm_grid <= cumul[-1]
        out[i, mask] = np.interp(comm_grid[mask], cumul, avgV)
    return out


# ============================================================
# Step 6: Plots
# ============================================================

def plot_avgV_vs_time(aggregates, outfile):
    """Classic time-series plot: avg_V over time for SF and Random with SE."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for net_label, color in [("Scale Free", COLOR_SF), ("Random", COLOR_RAND)]:
        agg = aggregates[net_label]
        t = np.arange(1, T + 1)
        mean = agg["avg_V_mean"]
        sem  = agg["avg_V_sem"]
        ax.plot(t, mean, color=color, label=net_label)
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlabel("tick")
    ax.set_ylabel("avg violations")
    ax.set_title(f"avg violations over time  (T={T}, {len(SEEDS)} seeds, +/- 1 SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_pareto(per_run_by_net, outfile, n_grid=200):
    """Efficiency plot: avg_V vs cumulative successful elicitations.
    A curve that is LOWER means the network reaches that violation level
    with FEWER total communications -> Pareto-better on the cost side."""
    # Build a common comms grid from 0 to the minimum of each network's
    # max-comms across runs (so both networks have data at every grid point).
    max_comms_per_net = {
        net: min(np.cumsum(r["elicits_per_tick"])[-1] for r in runs)
        for net, runs in per_run_by_net.items()
    }
    upper = min(max_comms_per_net.values())
    comm_grid = np.linspace(0, upper, n_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for net_label, color in [("Scale Free", COLOR_SF), ("Random", COLOR_RAND)]:
        mat  = interpolate_to_comm_axis(per_run_by_net[net_label], comm_grid)
        mean = np.nanmean(mat, axis=0)
        sem  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(comm_grid, mean, color=color, label=net_label)
        ax.fill_between(comm_grid, mean - sem, mean + sem,
                        color=color, alpha=0.2)
    ax.set_xlabel("cumulative successful elicitations (network total)")
    ax.set_ylabel("avg violations")
    ax.set_title("Efficiency: avg violations vs communication cost  "
                 f"(T={T}, {len(SEEDS)} seeds, +/- 1 SE)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_lorenz(per_run_by_net, outfile):
    """Lorenz curves of per-agent elicit counts, averaged across seeds.
    A curve closer to the diagonal = more equal division of labour."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1, label="perfect equality")
    for net_label, color in [("Scale Free", COLOR_SF), ("Random", COLOR_RAND)]:
        # Each run gives a per-agent vector; we average over seeds AFTER
        # sorting within run so the Lorenz curve is meaningful.
        # Stack sorted vectors and average pointwise.
        sorted_stack = np.stack([np.sort(r["per_agent_elicits"])
                                 for r in per_run_by_net[net_label]])
        mean_sorted  = sorted_stack.mean(axis=0)
        x, y = lorenz_curve(mean_sorted)
        g = gini(mean_sorted)
        ax.plot(x, y, color=color, label=f"{net_label}  (Gini = {g:.2f})")
    ax.set_xlabel("cumulative share of agents (sorted low -> high)")
    ax.set_ylabel("cumulative share of successful elicitations")
    ax.set_title("Division of cognitive labour (Lorenz curves)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def plot_per_agent_histogram(per_run_by_net, outfile):
    """Histogram of per-agent elicit counts, pooling across seeds."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for net_label, color in [("Scale Free", COLOR_SF), ("Random", COLOR_RAND)]:
        all_agents = np.concatenate([r["per_agent_elicits"]
                                     for r in per_run_by_net[net_label]])
        ax.hist(all_agents, bins=40, alpha=0.45, color=color,
                label=f"{net_label}  (mean={all_agents.mean():.0f}, "
                      f"max={all_agents.max()})")
    ax.set_xlabel("successful elicitations per agent over the run")
    ax.set_ylabel("number of agents (pooled across seeds)")
    ax.set_title("Per-agent elicit count distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


# ============================================================
# Step 7: Main
# ============================================================

def main():
    out_dir = Path(__file__).parent / "output" / "comm_efficiency"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved under: {out_dir}", flush=True)

    # Sanity check that our instrumentation is actually in use
    test_m = ProblemSolvingModel(seed=0, **BASE, **NETWORKS[0][1])
    assert isinstance(test_m.agents[0], InstrumentedPerson), \
        "Monkey-patch failed: model is not using InstrumentedPerson."
    print(f"Patch OK: agents are {type(test_m.agents[0]).__name__}", flush=True)

    # Report edge counts so we can verify the networks are actually matched.
    # This is just for transparency; it doesn't affect any downstream logic.
    print("\nEdge counts (across the 5 seeds used in this run):")
    for net_label, net_kw in NETWORKS:
        ec = []
        for seed in SEEDS:
            m = ProblemSolvingModel(seed=seed, **BASE, **net_kw)
            ec.append(m.network.number_of_edges())
        print(f"  {net_label:<11}  edges = {ec}  (mean {np.mean(ec):.0f})",
              flush=True)
    print("", flush=True)

    # ---- 7a. Run every (network, seed) pair -----------------
    per_run_by_net = {net_label: [] for net_label, _ in NETWORKS}
    summary_rows = []
    t0 = time.time()

    for net_label, net_kw in NETWORKS:
        for seed in SEEDS:
            r = run_one_instrumented(net_kw, seed)
            per_run_by_net[net_label].append(r)

            total_comms = int(r["per_agent_elicits"].sum())
            g           = gini(r["per_agent_elicits"])
            summary_rows.append({
                "network":       net_label,
                "seed":          seed,
                "final_avg_V":   float(r["avg_V"][-1]),
                "final_min_V":   float(r["min_V"][-1]),
                "total_comms":   total_comms,
                "mean_comms_per_agent": float(r["per_agent_elicits"].mean()),
                "max_comms_per_agent":  int(r["per_agent_elicits"].max()),
                "gini_per_agent":       float(g),
            })
            elapsed = (time.time() - t0) / 60
            print(f"  {net_label:<11} seed={seed}  "
                  f"final_avgV={r['avg_V'][-1]:6.2f}  "
                  f"total_comms={total_comms:>7,}  "
                  f"gini={g:.3f}  "
                  f"(elapsed {elapsed:.1f}m)",
                  flush=True)

    summary = pd.DataFrame(summary_rows)
    total_minutes = (time.time() - t0) / 60
    print(f"\nAll runs finished in {total_minutes:.1f} minutes.", flush=True)

    # ---- 7b. Aggregates used by the time-series plot --------
    aggregates = {}
    for net_label in per_run_by_net:
        stack_avgV = np.stack([r["avg_V"] for r in per_run_by_net[net_label]])
        aggregates[net_label] = {
            "avg_V_mean": stack_avgV.mean(axis=0),
            "avg_V_sem":  stack_avgV.std(axis=0, ddof=1) / np.sqrt(stack_avgV.shape[0]),
        }

    # ---- 7c. Plots (written FIRST so a locked xlsx can't block them) ---
    plot_avgV_vs_time(aggregates,     out_dir / "avgV_vs_time.png")
    plot_pareto     (per_run_by_net, out_dir / "pareto_avgV_vs_comms.png")
    plot_lorenz     (per_run_by_net, out_dir / "lorenz_per_agent.png")
    plot_per_agent_histogram(per_run_by_net, out_dir / "per_agent_hist.png")
    print(f"-> saved 4 PNG plots under {out_dir}", flush=True)

    # ---- 7d. Spreadsheet (wrapped so an open-in-Excel lock doesn't crash) ---
    # Tabs:
    #   summary           -- one row per (network, seed)
    #   summary_by_network-- aggregated by network
    #   trajectories      -- per-tick mean avg_V and mean cumulative comms
    #                        (downsampled to every 50 ticks for size)
    #   per_agent_<net>   -- per-agent elicit counts (rows = seed)
    xlsx_path = out_dir / "comm_efficiency_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="summary", index=False)

            by_net = summary.groupby("network").agg(
                final_avg_V_mean=("final_avg_V", "mean"),
                final_avg_V_sem =("final_avg_V", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                total_comms_mean=("total_comms", "mean"),
                total_comms_sem =("total_comms", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                gini_mean       =("gini_per_agent", "mean"),
                gini_sem        =("gini_per_agent",
                                  lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                max_comms_mean  =("max_comms_per_agent", "mean"),
            ).round(3)
            by_net.to_excel(w, sheet_name="summary_by_network")

            # Downsampled trajectories
            step = 50
            ticks = np.arange(step, T + 1, step)
            traj_rows = []
            for net_label, runs in per_run_by_net.items():
                avg_V_stack = np.stack([r["avg_V"] for r in runs])
                cumul_stack = np.stack([np.cumsum(r["elicits_per_tick"]) for r in runs])
                for t in ticks:
                    traj_rows.append({
                        "network":            net_label,
                        "tick":               t,
                        "avg_V_mean":         float(avg_V_stack[:, t-1].mean()),
                        "avg_V_sem":          float(avg_V_stack[:, t-1].std(ddof=1)
                                                    / np.sqrt(len(runs))),
                        "cum_comms_mean":     float(cumul_stack[:, t-1].mean()),
                    })
            pd.DataFrame(traj_rows).to_excel(w, sheet_name="trajectories", index=False)

            # Per-agent elicit counts, rows=seed
            for net_label, runs in per_run_by_net.items():
                tab = f"per_agent_{net_label.replace(' ', '_')}"
                df = pd.DataFrame(
                    {f"seed_{s}": r["per_agent_elicits"]
                     for s, r in zip(SEEDS, runs)}
                )
                df.index.name = "agent_id"
                df.to_excel(w, sheet_name=tab[:31])

        print(f"-> saved {xlsx_path}", flush=True)
    except PermissionError:
        print(f"!! COULD NOT WRITE {xlsx_path} -- is it open in Excel?", flush=True)
        print(f"   (the PNG plots above were already saved; just close Excel and",
              "re-run if you want the spreadsheet too.)", flush=True)

    # ---- 7e. Print the headline numbers ---------------------
    print("\n" + "=" * 70)
    print("HEADLINE NUMBERS (positive = Scale Free 'wins' on this metric)")
    print("=" * 70)
    by_net = summary.groupby("network").agg(
        final_avg_V=("final_avg_V", "mean"),
        total_comms=("total_comms", "mean"),
        gini=("gini_per_agent", "mean"),
        max_per_agent=("max_comms_per_agent", "mean"),
    ).round(3)
    print(by_net.to_string())
    sf  = by_net.loc["Scale Free"]
    rnd = by_net.loc["Random"]
    print(f"\n  final avg_V:   SF - Random = {sf['final_avg_V'] - rnd['final_avg_V']:+.3f}")
    print(f"  total comms:   SF - Random = "
          f"{sf['total_comms'] - rnd['total_comms']:+,.0f}  "
          f"(negative means SF used LESS communication)")
    print(f"  Gini:          SF - Random = {sf['gini'] - rnd['gini']:+.3f}  "
          f"(positive means SF more concentrated)")
    print(f"  max per-agent: SF - Random = "
          f"{sf['max_per_agent'] - rnd['max_per_agent']:+.1f}  "
          f"(positive means SF's top hub works harder than Random's)")


if __name__ == "__main__":
    main()
