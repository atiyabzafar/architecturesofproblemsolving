import numpy as np, random, sys, os
sys.path.insert(0, os.path.abspath('.'))
from model import ProblemSolvingModel

def run(alpha, K=30, N=80, T=4000, burn=2000, seeds=(0,1,2)):
    ps, vs = [], []
    for s in seeds:
        m = ProblemSolvingModel(N=N, K=K, alpha=alpha, obs_prob=0.01,
                                clause_interval=10, R=T, type_network="Random",
                                connect_prob=0.1, seed=s)
        pseries, vseries = [], []
        for t in range(T):
            m.step()
            if t >= burn:
                # fraction of ones across all agents/bits
                ones = sum(sum(a.x) for a in m.agents)
                pseries.append(ones / (N*K))
                vseries.append(m.avg_true_V / m.M)   # fraction of clauses violated
        ps.append(np.mean(pseries)); vs.append(np.mean(vseries))
    return np.mean(ps), np.std(ps), np.mean(vs), np.std(vs)

for a in (2, 4, 8):
    pm, psd, vm, vsd = run(a)
    print(f"alpha={a}:  p = {pm:.3f} +/- {psd:.3f}   V/M = {vm:.3f} +/- {vsd:.3f}")
