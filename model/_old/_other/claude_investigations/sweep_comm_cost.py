"""
Communication-cost variant: each agent has a per-tick budget of successful
elicitations. Once `comm_budget` clauses have been learned from neighbours
this tick, no further elicitation occurs until the next tick.

Motivation:
    In the baseline model hubs receive far more clauses per tick than
    peripherals (per-in-edge elicitation trials). If communication is
    costly — time spent listening, attention bandwidth — hubs hit a cap
    and the hub advantage should shrink. Peripheral agents rarely saturate
    the budget because they have few in-neighbours, so they're unaffected.

Mechanism:
    Subclass Person to wrap step() with a budget counter on successful
    elicits. Observation channel is NOT counted against the budget
    (private channel, no listening cost).

Design choices:
    * Budget is per successful elicitation. Attempts that fail
      (link_probability roll false, or neighbour has nothing new) do not
      consume budget.
    * In-neighbours are shuffled each tick so there's no systematic bias
      in who gets asked first.
    * Observation channel is unchanged.

Values swept:
    comm_budget in {1, 2, 5, 10, inf}
    (inf = no cost; recovers baseline)

Report, after T = 1000 rounds, 3 seeds:
    Scale Free  avg_V,  Random  avg_V,  gap_avg
    Scale Free  min_V,  Random  min_V,  gap_min
"""

import sys
from pathlib import Path
# model.py lives at the grandparent folder (this script is in _other/claude_investigations/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
import random
import pandas as pd

# Monkey-patch BEFORE any model is built: model.py instantiates Person(...)
# by class name inside setup_network, so swapping the attribute on the
# module makes the whole pipeline use our subclass.
import model as _mm
from model import Person, ProblemSolvingModel


class BudgetedPerson(Person):
    """Person with a per-tick cap on successful elicitations."""
    def step(self):
        my_kb_set = set(self.kb)
        budget = getattr(self.model, "comm_budget", math.inf)

        # 1) Private observation (free — doesn't cost anything)
        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)

        # 2) Communication - budgeted
        in_neighbors = self.in_neighbors
        if in_neighbors and budget > 0:
            comm_scale = self.model.comm_scale
            order = list(in_neighbors)
            random.shuffle(order)

            elicits_used = 0
            for nbr_id in order:
                if elicits_used >= budget:
                    break  # budget exhausted; stop listening this tick

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
                            elicits_used += 1


# Apply the patch so the model's Person() calls resolve to BudgetedPerson
_mm.Person = BudgetedPerson


class BudgetedModel(ProblemSolvingModel):
    """Model that accepts comm_budget and sets it on itself so the
    patched Person class can read it."""
    def __init__(self, *args, comm_budget=math.inf, **kwargs):
        self.comm_budget = comm_budget
        super().__init__(*args, **kwargs)


# ============================================================
# Experiment
# ============================================================
T       = 1000
SEEDS   = [42, 43, 44]
BUDGETS = [1, 2, 5, 10, math.inf]
BASE    = dict(N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
               setup_source="generate", R=T)

NETS = [
    ("Scale Free", dict(type_network="Scale Free", min_deg=3)),
    ("Random",     dict(type_network="Random",     connect_prob=0.20)),
]


def run_one(budget, net_kw, seed):
    m = BudgetedModel(comm_budget=budget, seed=seed, **BASE, **net_kw)
    for _ in range(T):
        m.step()
    return m.avg_true_V, m.min_true_V


if __name__ == "__main__":
    # Verify the monkey-patch took effect — first agent of a test model
    # should be a BudgetedPerson.
    test = BudgetedModel(comm_budget=math.inf, seed=0, **BASE, **NETS[0][1])
    assert isinstance(test.agents[0], BudgetedPerson), \
        "Monkey-patch failed: model is not using BudgetedPerson"
    print(f"Patch OK: agent class = {type(test.agents[0]).__name__}\n")

    rows = []
    total = len(BUDGETS) * len(NETS) * len(SEEDS)
    i = 0
    for budget in BUDGETS:
        b_str = "inf" if budget == math.inf else str(budget)
        for net_label, net_kw in NETS:
            for seed in SEEDS:
                i += 1
                avgV, minV = run_one(budget, net_kw, seed)
                rows.append({
                    "comm_budget": b_str,
                    "network":     net_label,
                    "seed":        seed,
                    "avg_V":       avgV,
                    "min_V":       minV,
                })
                print(f"  [{i:2d}/{total}] budget={b_str:>3}  "
                      f"{net_label:<11} seed={seed}  "
                      f"avgV={avgV:6.2f}  minV={minV:6.2f}")

    df = pd.DataFrame(rows)
    df["comm_budget"] = pd.Categorical(
        df["comm_budget"],
        categories=[str(b) if b != math.inf else "inf" for b in BUDGETS],
        ordered=True,
    )

    agg = (df.groupby(["comm_budget", "network"], observed=True)
             .agg(avg_V=("avg_V", "mean"),
                  min_V=("min_V", "mean"))
             .reset_index())
    sf  = agg[agg.network == "Scale Free"].set_index("comm_budget")
    rnd = agg[agg.network == "Random"].set_index("comm_budget")
    tbl = pd.DataFrame({
        "SF_avg":   sf["avg_V"],
        "Rand_avg": rnd["avg_V"],
        "gap_avg":  sf["avg_V"] - rnd["avg_V"],
        "SF_min":   sf["min_V"],
        "Rand_min": rnd["min_V"],
        "gap_min":  sf["min_V"] - rnd["min_V"],
    }).round(2)

    print("\n" + "=" * 70)
    print("COMMUNICATION-COST SWEEP: budget in {1, 2, 5, 10, inf}")
    print("Budget = max successful elicits per tick. inf = baseline (no cost)")
    print("=" * 70)
    print(tbl.to_string())

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    tbl.to_csv(out / "sweep_comm_cost.csv")
    print(f"\nCSV saved to {out / 'sweep_comm_cost.csv'}")
