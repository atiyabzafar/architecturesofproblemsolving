import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

from model_2026_04_21 import ProblemSolvingModel

N = 100
K = 50
OBS_PROB = 0.01
CLAUSE_INTERVAL = 10
T_STEPS = 3000
N_SEEDS = 10
SEEDS = list(range(200, 200 + N_SEEDS))
N_PROCS = min(os.cpu_count() or 1, 8)

ALPHAS = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
NETWORKS = {
    'Random': {'type_network': 'Random', 'connect_prob': 0.03},
    'Scale Free': {'type_network': 'Scale Free', 'min_deg': 3},
}


def erf_clip(x):
    return math.erf(max(0.0, x))


def theory_h_final(alpha, K):
    M = alpha * K
    if M <= 0:
        return 0.5
    mu = M / 4.0
    sigma = math.sqrt(max(M * 3.0 / 16.0, 1e-12))
    z = mu / (math.sqrt(2.0) * sigma)
    return 0.5 + 0.5 * erf_clip(z)


def theory_h_final_alt(alpha, K):
    x = math.sqrt(alpha * K / 6.0)
    return 0.5 + 0.5 * erf_clip(x)


def run_one(job):
    alpha, net_label, net_kwargs, seed = job
    model = ProblemSolvingModel(
        N=N,
        K=K,
        alpha=alpha,
        obs_prob=OBS_PROB,
        clause_interval=CLAUSE_INTERVAL,
        R=T_STEPS,
        setup_source='generate',
        seed=seed,
        **net_kwargs,
    )
    for _ in range(T_STEPS):
        model.step()
    return {
        'alpha': alpha,
        'network': net_label,
        'seed': seed,
        'final_homogeneity': float(model.homogeneity),
        'final_avg_violations': float(model.avg_true_V),
        'final_min_violations': float(model.min_true_V),
        'theory_h': theory_h_final(alpha, K),
        'theory_h_alt': theory_h_final_alt(alpha, K),
        'realized_edges': int(model.network.number_of_edges()),
    }


def main():
    jobs = [(a, label, kwargs, s) for a in ALPHAS for label, kwargs in NETWORKS.items() for s in SEEDS]
    with Pool(processes=N_PROCS) as pool:
        rows = list(tqdm(pool.imap_unordered(run_one, jobs), total=len(jobs), desc='Homogeneity MF check'))

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(['alpha', 'network'], as_index=False)
          .agg(
              final_homogeneity_mean=('final_homogeneity', 'mean'),
              final_homogeneity_std=('final_homogeneity', 'std'),
              final_avg_violations_mean=('final_avg_violations', 'mean'),
              final_avg_violations_std=('final_avg_violations', 'std'),
              theory_h=('theory_h', 'mean'),
              theory_h_alt=('theory_h_alt', 'mean'),
              realized_edges_mean=('realized_edges', 'mean'),
          )
    )

    df.to_csv('output/homogeneity_meanfield_runs.csv', index=False)
    summary.to_csv('output/homogeneity_meanfield_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {'Random': '#01696f', 'Scale Free': '#964219'}

    for net in NETWORKS:
        d = summary[summary['network'] == net].sort_values('alpha')
        axes[0].errorbar(d['alpha'], d['final_homogeneity_mean'], yerr=d['final_homogeneity_std'],
                         marker='o', capsize=3, color=colors[net], label=f'{net} sim')
    d0 = summary[summary['network'] == 'Random'].sort_values('alpha')
    axes[0].plot(d0['alpha'], d0['theory_h'], 'k--', lw=2, label='Mean-field theory')
    axes[0].set_xlabel('alpha')
    axes[0].set_ylabel('Final homogeneity')
    axes[0].set_title('Final homogeneity vs alpha')
    axes[0].set_ylim(0.45, 1.02)
    axes[0].legend(frameon=False)

    for net in NETWORKS:
        d = summary[summary['network'] == net].sort_values('alpha')
        axes[1].errorbar(d['alpha'], d['final_avg_violations_mean'], yerr=d['final_avg_violations_std'],
                         marker='o', capsize=3, color=colors[net], label=f'{net} sim')
    axes[1].plot(d0['alpha'], d0['alpha'] * K / 2.0, 'k--', lw=2, label='V* = alpha K / 2')
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('Final avg violations')
    axes[1].set_title('Violation attractor check')
    axes[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig('output/homogeneity_meanfield_check.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
