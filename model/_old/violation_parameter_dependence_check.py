import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from model_2026_04_21 import ProblemSolvingModel

T_STEPS = 3000
N_SEEDS = 10
SEEDS = list(range(300, 300 + N_SEEDS))
N_PROCS = min(os.cpu_count() or 1, 16)
NETWORK_COLORS = {
    'Random': '#01696f',
    'Scale Free': '#964219',
}


def run_case(job):
    case_name, sweep_name, sweep_value, params, seed = job
    model = ProblemSolvingModel(seed=seed, setup_source='generate', R=T_STEPS, **params)
    for _ in range(T_STEPS):
        model.step()
    out = {
        'case': case_name,
        'sweep': sweep_name,
        'value': sweep_value,
        'seed': seed,
        'final_avg_violations': float(model.avg_true_V),
        'final_min_violations': float(model.min_true_V),
        'theory': float(model.alpha * model.K / 2.0),
        'N': model.N,
        'K': model.K,
        'alpha': model.alpha,
        'obs_prob': model.obs_prob,
        'clause_interval': model.clause_interval,
        'network': model.type_network,
    }
    return out


def summarize(df):
    return (df.groupby(['case', 'sweep', 'value', 'network'], as_index=False)
              .agg(final_mean=('final_avg_violations', 'mean'),
                   final_std=('final_avg_violations', 'std'),
                   theory=('theory', 'mean')))


def make_jobs():
    jobs = []

    # 1) K dependence at fixed alpha
    alpha_fixed = 2.0
    for K in [20, 40, 60, 80, 100]:
        for network, extra in [('Random', {'connect_prob': 0.03}), ('Scale Free', {'min_deg': 3})]:
            params = dict(N=100, K=K, alpha=alpha_fixed, obs_prob=0.01,
                          clause_interval=10, type_network=network, **extra)
            for seed in SEEDS:
                jobs.append(('K_dependence', 'K', K, params, seed))

    # 2) N independence
    for N in [50, 100, 200, 400]:
        for network, extra in [('Random', {'connect_prob': 0.03}), ('Scale Free', {'min_deg': 3})]:
            params = dict(N=N, K=50, alpha=2.0, obs_prob=0.01,
                          clause_interval=10, type_network=network, **extra)
            for seed in SEEDS:
                jobs.append(('N_independence', 'N', N, params, seed))

    # 3) obs_prob independence
    for obs_prob in [0.0, 0.001, 0.01, 0.05, 0.1]:
        for network, extra in [('Random', {'connect_prob': 0.03}), ('Scale Free', {'min_deg': 3})]:
            params = dict(N=100, K=50, alpha=2.0, obs_prob=obs_prob,
                          clause_interval=10, type_network=network, **extra)
            for seed in SEEDS:
                jobs.append(('obsprob_independence', 'obs_prob', obs_prob, params, seed))

    # 4) clause_interval independence
    for clause_interval in [1, 5, 10, 20, 50]:
        for network, extra in [('Random', {'connect_prob': 0.03}), ('Scale Free', {'min_deg': 3})]:
            params = dict(N=100, K=50, alpha=2.0, obs_prob=0.01,
                          clause_interval=clause_interval, type_network=network, **extra)
            for seed in SEEDS:
                jobs.append(('clause_interval_independence', 'clause_interval', clause_interval, params, seed))

    return jobs


def plot_summary(summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cases = [
        ('K_dependence', 'K', 'K', axes[0, 0]),
        ('N_independence', 'N', 'N', axes[0, 1]),
        ('obsprob_independence', 'obs_prob', 'obs_prob', axes[1, 0]),
        ('clause_interval_independence', 'clause_interval', 'clause interval', axes[1, 1]),
    ]

    for case_name, sweep, xlabel, ax in cases:
        sub = summary[summary['case'] == case_name].copy().sort_values('value')
        for network in ['Random', 'Scale Free']:
            d = sub[sub['network'] == network]
            ax.errorbar(d['value'], d['final_mean'], yerr=d['final_std'], marker='o',
                        capsize=3, color=NETWORK_COLORS[network], label=f'{network} sim')
        d0 = sub[sub['network'] == 'Random']
        ax.plot(d0['value'], d0['theory'], 'k--', lw=2, label='Theory: alpha K / 2')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Final avg violations')
        ax.set_title(case_name.replace('_', ' '))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('output/violation_parameter_dependence_check.png', dpi=300, bbox_inches='tight')


def main():
    jobs = make_jobs()
    with Pool(processes=N_PROCS) as pool:
        rows = list(tqdm(pool.imap_unordered(run_case, jobs), total=len(jobs), desc='Violation parameter check'))
    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv('output/violation_parameter_dependence_runs.csv', index=False)
    summary.to_csv('output/violation_parameter_dependence_summary.csv', index=False)
    plot_summary(summary)

if __name__ == '__main__':
    main()
