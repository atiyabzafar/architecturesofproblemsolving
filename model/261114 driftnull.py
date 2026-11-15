"""
261114 driftnull.py  --  a drifting null model for the violation distribution.

No network, no communication, no testing, no negative beliefs. Each period an agent

  1. ACQUIRES. With probability `acq` it picks one clause uniformly at random from the CURRENT
     constraint set C and, if it holds no belief about it, stores it as a positive belief. Beliefs
     are therefore always TRUE when acquired. Memory is FIFO with room for `cap` of them; the
     oldest is evicted when it overflows.
  2. REPAIRS. Around each newly acquired clause it runs the greedy step of model.py: for each of
     the clause's two variables, in random order, flip it if that strictly lowers the number of
     violated BELIEVED clauses containing it. With `rand_repair` > 0 it also picks that many
     variables at random each period and applies the same test to them.

After every agent has acted, one clause of C is retired and replaced, as in the agent-based model.
A belief whose clause has since left C is STALE: nothing tells the agent, and nothing removes it
but eviction. So the stale share is not a parameter — it follows from how long a belief sits in
memory (about cap/acq periods) against how long a clause lives (about M periods).

    L = cap / acq        belief lifetime in periods
    stale share  ~  1 - (M/L) [ 1 - (1 - 1/M)^L ]

cap = 25, acq = 0.25 gives L = 100 and a stale share near one half, which is roughly where the
simulations sit. Raise acq or lower cap and memory turns over faster, so beliefs are fresher.

Violations V are always counted against the true C, which the agent never sees.

    python "261114 driftnull.py"                 the default study
    python "261114 driftnull.py" quick           a short version
"""
import random
import sys
from collections import Counter

import numpy as np

K_DEF, ALPHA_DEF = 30, 2


class DriftNull:
    def __init__(self, K=K_DEF, alpha=ALPHA_DEF, N=100, cap=25, acq=0.25, rand_repair=0.0,
                 tau=1, seed=0):
        self.K, self.M, self.N = K, int(round(alpha * K)), N
        self.cap, self.acq, self.rand_repair, self.tau = cap, acq, rand_repair, tau
        self.rng = random.Random(seed)
        self.C = [self._clause() for _ in range(self.M)]
        self.inC = Counter(self.C)
        self.x = np.array([[self.rng.randint(0, 1) for _ in range(K)] for _ in range(N)], np.int8)
        self.order = [[] for _ in range(N)]                 # FIFO of held clauses, oldest first
        self.held = [set() for _ in range(N)]
        self.byvar = [[[] for _ in range(K)] for _ in range(N)]
        self.t = 0

    def _clause(self):
        a, b = sorted(self.rng.sample(range(self.K), 2))
        return (0 if self.rng.random() < 0.5 else 1, a, b)

    # ---- the agent's two moves
    def _add(self, i, c):
        self.order[i].append(c); self.held[i].add(c)
        self.byvar[i][c[1]].append(c); self.byvar[i][c[2]].append(c)
        if len(self.order[i]) > self.cap:
            old = self.order[i].pop(0); self.held[i].discard(old)
            self.byvar[i][old[1]].remove(old); self.byvar[i][old[2]].remove(old)

    def _repair(self, i, j):
        """Greedy flip of variable j, judged on the agent's own believed clauses (model.py)."""
        xi, bv = self.x[i], self.byvar[i][j]
        if not bv:
            return
        before = 0
        for (o, a, b) in bv:
            before += (1 - xi[a] * xi[b]) if o == 0 else int(xi[a] == xi[b])
        xi[j] ^= 1
        after = 0
        for (o, a, b) in bv:
            after += (1 - xi[a] * xi[b]) if o == 0 else int(xi[a] == xi[b])
        if after >= before:
            xi[j] ^= 1

    def step(self):
        rng = self.rng
        for i in range(self.N):
            if rng.random() < self.acq:
                c = self.C[rng.randrange(self.M)]
                if c not in self.held[i]:
                    self._add(i, c)
                    for j in ((c[1], c[2]) if rng.random() < 0.5 else (c[2], c[1])):
                        self._repair(i, j)
            for _ in range(int(self.rand_repair)):
                self._repair(i, rng.randrange(self.K))
            if self.rand_repair % 1 and rng.random() < self.rand_repair % 1:
                self._repair(i, rng.randrange(self.K))
        self.t += 1
        if self.t % self.tau == 0:                           # environment drifts
            u = rng.randrange(self.M)
            old = self.C[u]; new = self._clause()
            self.C[u] = new
            self.inC[old] -= 1
            if self.inC[old] == 0:
                del self.inC[old]
            self.inC[new] += 1

    # ---- measurement (the agent never sees any of this)
    def measure(self):
        op = np.fromiter((c[0] for c in self.C), int, self.M)
        a = np.fromiter((c[1] for c in self.C), int, self.M)
        b = np.fromiter((c[2] for c in self.C), int, self.M)
        xa, xb = self.x[:, a], self.x[:, b]                  # (N, M)
        V = np.where(op == 0, 1 - xa * xb, (xa == xb).astype(np.int8)).sum(1)
        T = np.fromiter((sum(1 for c in h if c in self.inC) for h in self.held), float, self.N)
        B = np.fromiter((len(h) for h in self.held), float, self.N)
        return V.astype(float), T, B - T, self.x.mean(1)


def ensemble(runs=60, periods=1500, burn=1000, every=10, N=100, progress=False, **kw):
    """`runs` independent environments. Returns per-sample arrays of V (runs*samples, N) and the
    population means of T, S, p."""
    V, T, S, P = [], [], [], []
    for r in range(runs):
        m = DriftNull(N=N, seed=1000 + r, **kw)
        for t in range(periods):
            m.step()
            if t >= burn and (t - burn) % every == 0:
                v, tt, ss, p = m.measure()
                V.append(v); T.append(tt.mean()); S.append(ss.mean()); P.append(p.mean())
        if progress:
            print(f"    run {r+1}/{runs}", end="\r", flush=True)
    return np.array(V), float(np.mean(T)), float(np.mean(S)), float(np.mean(P))


def report(V, label, T=None, S=None, p=None):
    f = V.ravel(); mn = V.min(1); N = V.shape[1]
    sk = float(((f - f.mean()) ** 3).mean() / f.std() ** 3)
    ku = float(((f - f.mean()) ** 4).mean() / f.std() ** 4 - 3)
    within = float(np.sqrt(V.var(1, ddof=1).mean()))
    z = (f.mean() - mn.mean()) / within
    extra = "" if T is None else f"  T {T:5.1f} S {S:5.1f} p {p:.3f}"
    print(f"  {label:30s} mean {f.mean():6.2f}  sd {within:4.2f}  skew {sk:+.2f}  kurt {ku:+.2f}  "
          f"best of {N} {mn.mean():6.2f}  z {z:4.2f}{extra}")
    return dict(mean=float(f.mean()), sd=within, skew=sk, kurt=ku, best=float(mn.mean()), z=float(z),
                N=N, T=T, S=S, p=p)


if __name__ == "__main__":
    quick = "quick" in sys.argv
    runs = 12 if quick else 60
    print(f"DRIFTING NULL MODEL  (K=30, alpha=2, M=60; {runs} environments, "
          f"periods 1000-1500 sampled every 10)\n")
    print("DEFAULT SETTING")
    V, T, S, p = ensemble(runs=runs, cap=25, acq=0.25)
    report(V, "cap 25, acq 0.25", T, S, p)
    print("\n  simulations for comparison: mean 28.6-29.8, best of 100 19.9-20.2, "
          "T 11-14, S 11, p 0.69")

    print("\nMEMORY SIZE cap  (acq 0.25)")
    for cap in (10, 15, 25, 40, 60, 120):
        V, T, S, p = ensemble(runs=runs, cap=cap, acq=0.25)
        report(V, f"cap = {cap}", T, S, p)

    print("\nACQUISITION RATE acq  (cap 25)")
    for acq in (0.05, 0.1, 0.25, 0.5, 1.0):
        V, T, S, p = ensemble(runs=runs, cap=25, acq=acq)
        report(V, f"acq = {acq}", T, S, p)

    print("\nPOPULATION SIZE N  (cap 25, acq 0.25): does the best agent follow the order statistic?")
    for N in (10, 30, 100, 300, 1000):
        V, T, S, p = ensemble(runs=max(6, runs // 4), N=N, cap=25, acq=0.25)
        report(V, f"N = {N}", T, S, p)

    print("\nRANDOM REPAIRS PER PERIOD  (cap 25, acq 0.25)")
    for rr in (0, 1, 3):
        V, T, S, p = ensemble(runs=runs, cap=25, acq=0.25, rand_repair=rr)
        report(V, f"rand_repair = {rr}", T, S, p)
