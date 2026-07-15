"""
Learning-by-testing variant of the problem-solving model.
==================================================================

This evolves the earlier "learn-by-doing" idea (model_learnbydoing.py). There,
agents actively tested clauses and *dropped* a held clause when it turned out to
no longer exist. That was not enough to expel "fake news": a dropped clause was
simply re-imported the next time a neighbour (who had not yet tested it) shared
it. Learning-by-testing adds three changes so that disconfirmation can spread
and stick:

  (1) DOUBLED CAPACITY. The knowledge base now stores two kinds of belief, so
      its capacity is doubled: 2 * |C| beliefs (the baseline cap was |C| = M).

  (2) SIGNED BELIEFS. The KB is made up of beliefs about clauses *existing*
      (positive beliefs) and beliefs about clauses *not existing* (negative
      beliefs). When an agent tests a held belief and finds the clause no longer
      exists, the clause is NOT discarded - it is kept as a negative belief
      ("this clause does not exist"). Local optimisation only ever uses the
      positive beliefs (the clauses the agent believes are real).

  (3) SIGNED, RECENCY-ORDERED COMMUNICATION. When agents communicate they can
      transmit a positive OR a negative belief: the sender passes on its MOST
      RECENT belief (most recently learned, verified, or received) among those
      the receiver disagrees on, and the receiver updates its belief about that
      clause to match the sender's. Fresh news - including fresh disconfirmation
      - therefore travels first, and a negative belief can overwrite a
      neighbour's stale positive belief: this is the channel through which
      fake news is expelled.

Mechanics
---------
Each tick an agent's private channel does one of two things with equal
probability (this replaces passive observation):

  * TEST A NEW CLAUSE - generate a clause it holds no belief about and, if it
    actually exists in the universe C, add it as a positive belief. (A generated
    clause that does not exist is simply discarded - we do not flood the KB with
    negative beliefs about random never-held clauses.)

  * TEST A BELIEF - pick any belief it currently holds (positive or negative)
    and set it to the ground truth: a positive belief whose clause has vanished
    flips to negative ("falsified"); a negative belief whose clause has
    reappeared flips to positive ("resurrected").

Then communication runs (receiver-pull, as in the baseline): for each active
in-link the agent pulls the neighbour's most recent belief on which the two
disagree (a belief it lacks, or one it holds with the opposite sign) and adopts
the neighbour's sign.

Belief bookkeeping
------------------
  self.kb    - list of clauses believed to EXIST (positive beliefs). Reused from
               the base Person so every optimisation routine (local_update_around,
               violation_count(self.kb, x), ...) works unchanged.
  self.sign  - dict: clause -> +1 (exists) / -1 (does not exist). Membership in
               this dict means "the agent holds a belief about this clause".
  self.order - clauses in FIFO order. A belief's position is refreshed whenever
               it is tested or updated, so actively-maintained beliefs survive
               and only neglected ones are evicted once capacity is exceeded.

Usage
-----
Importing this module monkey-patches ``model.Person`` (same convention as
``model_learnbydoing.py`` and the sweep scripts), because
``ProblemSolvingModel.setup_network`` builds agents by the ``Person`` name bound
in ``model.py``.

    from model_learningbytesting import LearningByTestingModel
    m = LearningByTestingModel(N=80, K=30, alpha=2, clause_interval=10,
                               setup_source="generate", R=1000, seed=42,
                               type_network="Scale Free", min_deg=3)
    for _ in range(1000):
        m.step()

Pass ``learning_by_testing=False`` to recover the baseline passive-observation
behaviour (handy for the A/B demo in ``__main__``).
"""

import random
import numpy as np

import model as _base
from model import Person, ProblemSolvingModel


class LearningByTestingPerson(Person):
    """Person whose knowledge base holds signed beliefs and whose private
    channel is active testing rather than passive observation."""

    def __init__(self, unique_id, model, K):
        super().__init__(unique_id, model, K)
        # self.kb (positive beliefs) is created by the base __init__.
        self.sign = {}     # clause -> +1 (exists) / -1 (does not exist)
        self.order = []    # clauses in FIFO order (refreshed when re-touched)
        self.test_rate = 1  # tests per tick; a "super-tester" has >1

    # ---- belief bookkeeping ------------------------------------------------
    def belief_cap(self):
        """Doubled knowledge-base capacity: 2 * |C| (baseline cap was |C|=M)."""
        return 2 * self.model.M

    def set_belief(self, clause, s):
        """Record that the agent now believes ``clause`` has sign ``s``
        (+1 exists / -1 does not). Refreshes the clause's FIFO position and
        keeps the positive-belief list ``self.kb`` in sync. Returns a transition
        code: 'new_pos', 'new_neg', 'pos2neg', 'neg2pos' or 'confirm'."""
        cur = self.sign.get(clause)

        # Refresh FIFO youth: the agent just verified / was told about this clause.
        if cur is not None:
            try:
                self.order.remove(clause)
            except ValueError:
                pass
        self.order.append(clause)

        self.sign[clause] = s
        # Keep self.kb == {clauses believed to exist}.
        in_kb = clause in self.kb
        if s == 1 and not in_kb:
            self.kb.append(clause)
        elif s == -1 and in_kb:
            self.kb.remove(clause)

        self._enforce_cap()

        if cur is None:
            return 'new_pos' if s == 1 else 'new_neg'
        if cur == s:
            return 'confirm'
        return 'neg2pos' if s == 1 else 'pos2neg'

    def _enforce_cap(self):
        """Evict oldest beliefs (positive or negative) beyond the doubled cap."""
        cap = self.belief_cap()
        while len(self.order) > cap:
            old = self.order.pop(0)
            s = self.sign.pop(old, None)
            if s == 1:
                try:
                    self.kb.remove(old)
                except ValueError:
                    pass

    # ---- step --------------------------------------------------------------
    def step(self):
        if getattr(self.model, "learning_by_testing", True):
            self._step_testing()
        else:
            self._step_baseline()

    def _step_testing(self):
        C_set = getattr(self.model, "_C_set", None)
        if C_set is None:
            C_set = set(self.model.C)

        # Private channel: `test_rate` independent test actions per tick. A
        # "super-tester" (test_rate > 1) verifies several clauses each tick.
        for _ in range(self.test_rate):
            self._one_test(C_set)

        # --- communication: transmit signed beliefs (receiver-pull) ---
        self._communicate_signed()

    def _one_test(self, C_set):
        """One test action: with equal probability, test a NEW clause or a HELD
        belief, setting the outcome to ground truth."""
        # --- 50/50 test-a-new-clause / test-a-belief ---
        if random.random() < 0.5:
            # (1) test a new clause: one we hold no belief about.
            cand = self.model.random_clause()
            tries = 0
            while cand in self.sign and tries < 10:
                cand = self.model.random_clause(); tries += 1
            if cand not in self.sign and cand in C_set:
                self.set_belief(cand, 1)
                self.local_update_around(cand)
            # A generated clause that does not exist is discarded (no negative
            # belief recorded for clauses the agent never held).
        else:
            # (2) test a belief we hold (positive or negative) -> ground truth.
            if self.order:
                clause = random.choice(self.order)
                s = 1 if clause in C_set else -1
                t = self.set_belief(clause, s)
                if t == 'neg2pos':           # resurrected: now an active constraint
                    self.local_update_around(clause)

    def _communicate_signed(self):
        if not self.in_neighbors:
            return
        comm_scale = self.model.comm_scale
        for nbr_id in self.in_neighbors:
            edge_data = self.model.network[nbr_id][self.unique_id]
            link_weight = edge_data.get("weight", 1.0)
            if random.random() < min(1.0, comm_scale * link_weight):
                nbr = self.model.agent_list[nbr_id]
                if not nbr.order:
                    continue
                # Pick which belief the neighbour passes on among those we
                # disagree about (we lack it, or hold the OPPOSITE sign):
                if getattr(self.model, "recency", True):
                    # RECENCY (default): its MOST RECENT such belief. nbr.order
                    # runs oldest -> newest (refreshed on every touch), so the
                    # freshest disagreement is at the tail.
                    c = next((cc for cc in reversed(nbr.order)
                              if self.sign.get(cc) != nbr.sign[cc]), None)
                else:
                    # CLASSIC: a uniform-random such belief.
                    diff = [cc for cc in nbr.order if self.sign.get(cc) != nbr.sign[cc]]
                    c = random.choice(diff) if diff else None
                if c is not None:
                    # adopt the neighbour's belief, flipping ours if it differs;
                    # a transmitted negative thus overwrites a stale positive.
                    if self.set_belief(c, nbr.sign[c]) in ('new_pos', 'neg2pos'):
                        self.local_update_around(c)

    def _step_baseline(self):
        """Passive observation + unsigned communication, used when
        learning_by_testing is False. Communication honours self.model.recency:
        recency -> neighbour sends its most recently acquired clause the
        recipient lacks; else a uniform-random unknown clause (the classic
        model.py / serve.py behaviour)."""
        my_kb_set = set(self.kb)
        if random.random() < self.model.obs_prob:
            obs_clause = random.choice(self.model.C)
            if obs_clause not in my_kb_set:
                self.add_clause_to_kb(obs_clause)
                self.local_update_around(obs_clause)
                my_kb_set.add(obs_clause)
        if self.in_neighbors:
            comm_scale = self.model.comm_scale
            for nbr_id in self.in_neighbors:
                edge_data = self.model.network[nbr_id][self.unique_id]
                link_weight = edge_data.get("weight", 1.0)
                if random.random() < min(1.0, comm_scale * link_weight):
                    nbr_agent = self.model.agent_list[nbr_id]
                    if nbr_agent.kb:
                        if getattr(self.model, "recency", True):
                            cprime = next((c for c in reversed(nbr_agent.kb)
                                           if c not in my_kb_set), None)
                        else:
                            unknowns = [c for c in nbr_agent.kb if c not in my_kb_set]
                            cprime = random.choice(unknowns) if unknowns else None
                        if cprime is not None:
                            self.add_clause_to_kb(cprime)
                            self.local_update_around(cprime)
                            my_kb_set.add(cprime)


# Monkey-patch so ProblemSolvingModel.setup_network() builds these agents.
_base.Person = LearningByTestingPerson


class LearningByTestingModel(ProblemSolvingModel):
    """ProblemSolvingModel using signed-belief, learning-by-testing agents.

    ``learning_by_testing`` defaults to True; set it False to recover baseline
    passive observation (the agent class supports both via the flag)."""

    def __init__(self, *args, learning_by_testing=True, recency=True, **kwargs):
        self.learning_by_testing = learning_by_testing
        self.recency = recency   # communication: most-recent belief vs uniform-random
        self._C_set = set()
        super().__init__(*args, **kwargs)

    def step(self):
        # Cache the universe as a set once per tick (C is constant within a tick;
        # replacement happens after agents act). Agents read self.model._C_set.
        self._C_set = set(self.C)
        super().step()


# ======================================================================
# Demo / smoke test: does signed learning-by-testing expel stale beliefs
# better than the baseline (and than plain learn-by-doing)?
# ======================================================================
def _kb_stats(model):
    """Per-agent averages: positive beliefs, negative beliefs, and the fraction
    of positive beliefs that are STALE (no longer in the universe C)."""
    C_set = set(model.C)
    pos, neg, stale_frac = [], [], []
    for a in model.agents:
        npos = len(a.kb)
        # In baseline mode self.order is unused (all beliefs are positive); fall
        # back to the kb so the negative count reads 0 rather than going negative.
        nall = len(a.order) if getattr(a, "order", None) else npos
        pos.append(npos)
        neg.append(nall - npos)
        if npos:
            stale_frac.append(sum(1 for c in a.kb if c not in C_set) / npos)
        else:
            stale_frac.append(0.0)
    return float(np.mean(pos)), float(np.mean(neg)), float(np.mean(stale_frac))


def _run(learning_by_testing, T=1000, seed=42):
    m = LearningByTestingModel(
        N=80, K=30, alpha=2, obs_prob=0.01, clause_interval=10,
        setup_source="generate", R=T, seed=seed,
        type_network="Scale Free", min_deg=3,
        learning_by_testing=learning_by_testing,
    )
    for _ in range(T):
        m.step()
    pos, neg, stale = _kb_stats(m)
    return m.avg_true_V, pos, neg, stale


if __name__ == "__main__":
    test = LearningByTestingModel(N=10, K=10, alpha=2, setup_source="generate",
                                  R=10, seed=0, type_network="Random",
                                  connect_prob=0.3)
    assert isinstance(test.agents[0], LearningByTestingPerson), \
        "Monkey-patch failed: model is not using LearningByTestingPerson"
    print(f"Patch OK: agent class = {type(test.agents[0]).__name__}\n")

    T = 1000
    print(f"Scale Free, N=80 K=30 alpha=2 clause_interval=10, T={T}, seed=42")
    print(f"{'mode':<22}{'avg_V (vs C)':>14}{'pos bel':>10}{'neg bel':>10}{'stale pos':>12}")
    print("-" * 68)
    for label, lbt in [("baseline (obs)", False), ("learning-by-testing", True)]:
        avgV, pos, neg, stale = _run(learning_by_testing=lbt, T=T, seed=42)
        print(f"{label:<22}{avgV:>14.2f}{pos:>10.1f}{neg:>10.1f}{stale:>11.1%}")
    print("\nLearning-by-testing should drive the stale-positive fraction down "
          "further\nthan plain learn-by-doing did (62.7%->43.2%): negative "
          "beliefs now persist\nand propagate, overwriting neighbours' stale "
          "positive beliefs.")
