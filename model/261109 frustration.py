"""Fewest constraints that any assignment can leave violated (Section 3.2, footnote).

Draws 60 random constraint sets as the model draws its initial set (model.py,
generate_clauses and random_clause: M = 60 constraints, each over two distinct
variables out of K = 30, AND or XOR with probability 1/2) and finds, for each,
the exact minimum number of violated constraints by exhaustive search over all
2^30 assignments. The 30 variables are split into two halves of 15; constraints
across the halves enter through a matrix product, so each set takes seconds.

  python "261109 frustration.py" > "output/261109 results/frustration.txt"
"""
import random
import time

import numpy as np

K, M, SETS, SEED = 30, 60, 60, 1
H = K // 2


def draw_set(rng):
    """M constraints (op, a, b) with 0-based a < b; op 0 = AND, 1 = XOR, as model.random_clause."""
    out = []
    for _ in range(M):
        a, b = sorted(rng.sample(range(K), 2))
        out.append((0 if rng.random() < 0.5 else 1, a, b))
    return out


def violated(op, xa, xb):
    """AND is violated unless both variables are 1; XOR unless exactly one is 1 (model.clause_violated)."""
    return 1 - xa * xb if op == 0 else (xa == xb).astype(np.float32)


idx = np.arange(1 << H, dtype=np.int64)
bits = ((idx[:, None] >> np.arange(H)[None, :]) & 1).astype(np.float32)     # (2^15, 15)


def exact_minimum(C):
    low = np.zeros(1 << H, np.float32)        # constraints within variables 0-14
    high = np.zeros(1 << H, np.float32)       # constraints within variables 15-29
    cross0 = np.zeros(1 << H, np.float32)     # cross constraints with the low variable at 0
    D = np.zeros((H, 1 << H), np.float32)     # change when the low variable is 1 instead
    for op, a, b in C:
        if b < H:
            low += violated(op, bits[:, a], bits[:, b])
        elif a >= H:
            high += violated(op, bits[:, a - H], bits[:, b - H])
        else:
            xb = bits[:, b - H]
            v0, v1 = violated(op, np.float32(0), xb), violated(op, np.float32(1), xb)
            cross0 += v0; D[a] += v1 - v0
    best = np.inf
    for s0 in range(0, 1 << H, 2048):
        sl = slice(s0, s0 + 2048)
        tot = bits @ D[:, sl]                  # (2^15 low halves, 2048 high halves)
        tot += low[:, None]; tot += (high[sl] + cross0[sl])[None, :]
        best = min(best, float(tot.min()))
    return int(round(best))


if __name__ == "__main__":
    rng = random.Random(SEED)
    t0 = time.time(); res = []
    for i in range(SETS):
        res.append(exact_minimum(draw_set(rng)))
        print(f"set {i + 1}: exact minimum {res[-1]} violated", flush=True)
    r = np.array(res)
    print(f"exact minimum over {SETS} sets of {M} constraints: mean {r.mean():.2f}, sd {r.std(ddof=1):.2f}, "
          f"range {r.min()}-{r.max()} ({time.time() - t0:.0f}s)")
