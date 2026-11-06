# 261001 analysis summary: main set

Default model (learning by testing + recency), N=100, K=30, alpha=2, tau=1; long-run experiment T=2000, last 500 ticks; shock experiment: 100% of the constraint set replaced at t=1500, run to t=2500; seeds 1..40. All uncertainties use the run as the unit of observation.

## Networks (means over seeds)

| network | <k>  | sd k_in | max k_in | Var(intake) | corr(k_in,k_out) | clustering | no in-link | largest strongly connected part |
|---------|------|---------|----------|-------------|------------------|------------|------------|---------------------------------|
| even    | 4.0  | 0.0     | 4        | 0.00        | none             | 0.06       | 0%         | 100%                            |
| uneven  | 4.0  | 4.8     | 37       | 1.44        | -0.01            | 0.21       | 0%         | 100%                            |
| dense   | 12.0 | 13.8    | 99       | 1.33        | -0.01            | 0.56       | 0%         | 100%                            |

## Long run (last 500 ticks; mean +- 1 s.e. over 40 seeds)

| network | average violations | best-agent violations | homogeneity    | stale beliefs (%) |
|---------|--------------------|-----------------------|----------------|-------------------|
| even    | 28.74 +- 0.11      | 20.07 +- 0.12         | 0.712 +- 0.003 | 43.2 +- 0.3       |
| uneven  | 29.61 +- 0.07      | 19.97 +- 0.09         | 0.704 +- 0.003 | 49.6 +- 0.2       |
| dense   | 29.56 +- 0.07      | 19.86 +- 0.10         | 0.701 +- 0.002 | 48.8 +- 0.3       |

## Second-order check

Curve fitted on the main set: V(1) = 28.95, V'(1) = -2.02, V''(1) = 1.63; predicted average = V(1) + V''(1) Var(intake) / 2.

| network | Var(intake) | predicted average | measured average |
|---------|-------------|-------------------|------------------|
| even    | 0.00        | 28.95             | 28.74 +- 0.11    |
| uneven  | 1.44        | 30.13             | 29.61 +- 0.07    |
| dense   | 1.33        | 30.04             | 29.56 +- 0.07    |

## Position (per run; mean +- 1 s.e. over seeds)

| network | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|---------|-----------------------|-------------------|---------------------------|-----------------------------------|
| even    | none (no spread)      | none              | none                      | -                                 |
| uneven  | -0.57 +- 0.01         | 100%              | -4.90 +- 0.15             | 98%                               |
| dense   | -0.59 +- 0.01         | 100%              | -5.40 +- 0.14             | 98%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|---------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| even    | 29                                            | 28.46 +- 0.14    | 32.13 +- 0.17    | 30.62 +- 0.21     | 29.14 +- 0.19      | 28.74 +- 0.17       | 28.58 +- 0.14       | 28.61 +- 0.09        |
| uneven  | 46                                            | 29.64 +- 0.13    | 32.22 +- 0.17    | 31.12 +- 0.17     | 30.14 +- 0.14      | 29.52 +- 0.15       | 29.62 +- 0.14       | 29.58 +- 0.08        |
| dense   | 34                                            | 29.46 +- 0.13    | 32.49 +- 0.16    | 31.00 +- 0.15     | 30.16 +- 0.14      | 29.91 +- 0.12       | 29.49 +- 0.14       | 29.49 +- 0.07        |

## Short run: new and retired clauses after the shock

| network | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|---------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| even    | 16.0 +- 0.4                         | 29.0 +- 0.8                         | 39.6 +- 0.9                          | 51                     | 40%        | 45                                       | 32%                     |
| uneven  | 13.0 +- 0.4                         | 20.9 +- 0.6                         | 27.3 +- 0.8                          | 79                     | 30%        | 52                                       | 22%                     |
| dense   | 13.0 +- 0.4                         | 21.0 +- 0.5                         | 28.3 +- 0.7                          | 78                     | 31%        | 50                                       | 21%                     |

## Differences between main-set networks (Welch; the run is the unit)

| pair           | quantity                                        | difference | s.e. | t     |
|----------------|-------------------------------------------------|------------|------|-------|
| even - uneven  | average violations                              | -0.87      | 0.13 | -6.8  |
| even - uneven  | best-agent violations                           | +0.10      | 0.15 | +0.7  |
| even - uneven  | stale beliefs (%)                               | -6.43      | 0.31 | -20.7 |
| even - uneven  | pre-shock level                                 | -1.17      | 0.19 | -6.1  |
| even - uneven  | level, ticks 1-20 after shock                   | -0.09      | 0.24 | -0.4  |
| even - uneven  | level, ticks 21-60 after shock                  | -0.50      | 0.27 | -1.9  |
| even - uneven  | level, ticks 61-120 after shock                 | -1.00      | 0.24 | -4.1  |
| even - uneven  | level, ticks 121-200 after shock                | -0.78      | 0.23 | -3.4  |
| even - uneven  | level, ticks 201-400 after shock                | -1.04      | 0.19 | -5.3  |
| even - uneven  | level, ticks 401-1000 after shock               | -0.97      | 0.12 | -8.2  |
| even - uneven  | excess over own pre-shock level, ticks 1-20     | +1.08      | 0.27 | +4.1  |
| even - uneven  | excess over own pre-shock level, ticks 21-60    | +0.67      | 0.28 | +2.4  |
| even - uneven  | excess over own pre-shock level, ticks 61-120   | +0.17      | 0.26 | +0.7  |
| even - uneven  | excess over own pre-shock level, ticks 121-200  | +0.39      | 0.29 | +1.4  |
| even - uneven  | excess over own pre-shock level, ticks 201-400  | +0.13      | 0.27 | +0.5  |
| even - uneven  | excess over own pre-shock level, ticks 401-1000 | +0.20      | 0.22 | +0.9  |
| even - uneven  | new clause held by (%), 30 ticks on             | +3.00      | 0.55 | +5.4  |
| even - uneven  | new clause held by (%), 60 ticks on             | +8.16      | 0.99 | +8.3  |
| even - uneven  | new clause held by (%), 100 ticks on            | +12.23     | 1.21 | +10.1 |
| dense - uneven | average violations                              | -0.05      | 0.10 | -0.5  |
| dense - uneven | best-agent violations                           | -0.11      | 0.14 | -0.8  |
| dense - uneven | stale beliefs (%)                               | -0.82      | 0.32 | -2.5  |
| dense - uneven | pre-shock level                                 | -0.17      | 0.19 | -0.9  |
| dense - uneven | level, ticks 1-20 after shock                   | +0.27      | 0.24 | +1.2  |
| dense - uneven | level, ticks 21-60 after shock                  | -0.12      | 0.23 | -0.5  |
| dense - uneven | level, ticks 61-120 after shock                 | +0.02      | 0.20 | +0.1  |
| dense - uneven | level, ticks 121-200 after shock                | +0.39      | 0.19 | +2.0  |
| dense - uneven | level, ticks 201-400 after shock                | -0.13      | 0.20 | -0.6  |
| dense - uneven | level, ticks 401-1000 after shock               | -0.09      | 0.11 | -0.8  |
| dense - uneven | excess over own pre-shock level, ticks 1-20     | +0.44      | 0.23 | +2.0  |
| dense - uneven | excess over own pre-shock level, ticks 21-60    | +0.05      | 0.24 | +0.2  |
| dense - uneven | excess over own pre-shock level, ticks 61-120   | +0.19      | 0.24 | +0.8  |
| dense - uneven | excess over own pre-shock level, ticks 121-200  | +0.56      | 0.25 | +2.3  |
| dense - uneven | excess over own pre-shock level, ticks 201-400  | +0.04      | 0.27 | +0.2  |
| dense - uneven | excess over own pre-shock level, ticks 401-1000 | +0.08      | 0.20 | +0.4  |
| dense - uneven | new clause held by (%), 30 ticks on             | +0.01      | 0.52 | +0.0  |
| dense - uneven | new clause held by (%), 60 ticks on             | +0.17      | 0.78 | +0.2  |
| dense - uneven | new clause held by (%), 100 ticks on            | +0.95      | 1.11 | +0.9  |
| even - dense   | average violations                              | -0.82      | 0.13 | -6.4  |
| even - dense   | best-agent violations                           | +0.21      | 0.15 | +1.4  |
| even - dense   | stale beliefs (%)                               | -5.61      | 0.37 | -15.3 |
| even - dense   | pre-shock level                                 | -1.00      | 0.19 | -5.1  |
| even - dense   | level, ticks 1-20 after shock                   | -0.36      | 0.24 | -1.5  |
| even - dense   | level, ticks 21-60 after shock                  | -0.38      | 0.25 | -1.5  |
| even - dense   | level, ticks 61-120 after shock                 | -1.02      | 0.24 | -4.3  |
| even - dense   | level, ticks 121-200 after shock                | -1.17      | 0.21 | -5.7  |
| even - dense   | level, ticks 201-400 after shock                | -0.91      | 0.20 | -4.6  |
| even - dense   | level, ticks 401-1000 after shock               | -0.88      | 0.11 | -7.7  |
| even - dense   | excess over own pre-shock level, ticks 1-20     | +0.64      | 0.27 | +2.4  |
| even - dense   | excess over own pre-shock level, ticks 21-60    | +0.62      | 0.29 | +2.2  |
| even - dense   | excess over own pre-shock level, ticks 61-120   | -0.02      | 0.29 | -0.1  |
| even - dense   | excess over own pre-shock level, ticks 121-200  | -0.17      | 0.29 | -0.6  |
| even - dense   | excess over own pre-shock level, ticks 201-400  | +0.09      | 0.29 | +0.3  |
| even - dense   | excess over own pre-shock level, ticks 401-1000 | +0.12      | 0.23 | +0.5  |
| even - dense   | new clause held by (%), 30 ticks on             | +2.99      | 0.57 | +5.2  |
| even - dense   | new clause held by (%), 60 ticks on             | +7.99      | 0.95 | +8.4  |
| even - dense   | new clause held by (%), 100 ticks on            | +11.29     | 1.14 | +9.9  |

# Appendix set: standard generators

## Networks (means over seeds)

| network    | <k> | sd k_in | max k_in | Var(intake) | corr(k_in,k_out) | clustering | no in-link | largest strongly connected part |
|------------|-----|---------|----------|-------------|------------------|------------|------------|---------------------------------|
| random     | 3.9 | 1.9     | 10       | 0.24        | -0.00            | 0.08       | 2%         | 96%                             |
| smallworld | 4.0 | 1.5     | 8        | 0.15        | -0.82            | 0.49       | 1%         | 99%                             |
| scalefree  | 3.7 | 3.9     | 24       | 1.16        | 0.74             | 0.22       | 8%         | 89%                             |
| layered    | 3.9 | 1.9     | 10       | 0.24        | 0.00             | 0.10       | 2%         | 96%                             |

## Long run (last 500 ticks; mean +- 1 s.e. over 40 seeds)

| network    | average violations | best-agent violations | homogeneity    | stale beliefs (%) |
|------------|--------------------|-----------------------|----------------|-------------------|
| random     | 29.21 +- 0.11      | 20.10 +- 0.11         | 0.702 +- 0.004 | 46.1 +- 0.2       |
| smallworld | 28.93 +- 0.08      | 20.01 +- 0.10         | 0.714 +- 0.003 | 45.3 +- 0.2       |
| scalefree  | 29.86 +- 0.08      | 20.25 +- 0.11         | 0.693 +- 0.003 | 49.6 +- 0.2       |
| layered    | 29.25 +- 0.10      | 20.11 +- 0.11         | 0.702 +- 0.003 | 46.4 +- 0.2       |

## Second-order check

Curve fitted on the main set: V(1) = 28.95, V'(1) = -2.02, V''(1) = 1.63; predicted average = V(1) + V''(1) Var(intake) / 2.

| network    | Var(intake) | predicted average | measured average |
|------------|-------------|-------------------|------------------|
| random     | 0.24        | 29.15             | 29.21 +- 0.11    |
| smallworld | 0.15        | 29.07             | 28.93 +- 0.08    |
| scalefree  | 1.16        | 29.90             | 29.86 +- 0.08    |
| layered    | 0.24        | 29.15             | 29.25 +- 0.10    |

## Position (per run; mean +- 1 s.e. over seeds)

| network    | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|------------|-----------------------|-------------------|---------------------------|-----------------------------------|
| random     | -0.58 +- 0.01         | 100%              | -4.12 +- 0.13             | 84%                               |
| smallworld | -0.48 +- 0.01         | 100%              | -3.09 +- 0.14             | 77%                               |
| scalefree  | -0.62 +- 0.01         | 100%              | -5.49 +- 0.14             | 96%                               |
| layered    | -0.57 +- 0.01         | 100%              | -3.75 +- 0.11             | 81%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network    | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|------------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| random     | 29                                            | 29.19 +- 0.14    | 32.34 +- 0.15    | 31.15 +- 0.12     | 29.88 +- 0.10      | 29.19 +- 0.15       | 29.07 +- 0.11       | 29.39 +- 0.07        |
| smallworld | 40                                            | 29.23 +- 0.14    | 32.26 +- 0.20    | 30.89 +- 0.21     | 29.78 +- 0.16      | 29.32 +- 0.17       | 29.16 +- 0.10       | 29.06 +- 0.09        |
| scalefree  | 36                                            | 29.79 +- 0.13    | 32.26 +- 0.17    | 31.01 +- 0.15     | 30.09 +- 0.11      | 29.90 +- 0.12       | 29.79 +- 0.11       | 29.95 +- 0.08        |
| layered    | 31                                            | 29.15 +- 0.14    | 32.45 +- 0.14    | 31.07 +- 0.17     | 29.78 +- 0.17      | 29.16 +- 0.15       | 29.07 +- 0.14       | 29.20 +- 0.09        |

## Short run: new and retired clauses after the shock

| network    | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|------------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| random     | 14.2 +- 0.4                         | 24.7 +- 0.5                         | 31.8 +- 0.9                          | 62                     | 32%        | 47                                       | 23%                     |
| smallworld | 14.7 +- 0.4                         | 27.6 +- 0.7                         | 35.6 +- 0.9                          | 54                     | 36%        | 46                                       | 24%                     |
| scalefree  | 11.8 +- 0.2                         | 19.2 +- 0.5                         | 24.3 +- 0.6                          | 112                    | 26%        | 51                                       | 18%                     |
| layered    | 13.7 +- 0.4                         | 23.7 +- 0.5                         | 29.5 +- 0.8                          | 68                     | 32%        | 47                                       | 21%                     |

## Files

- `261001 timeseries.npz`, `261001 agents.csv`, `261001 networks.csv`: raw results
- `261001 fig_*`: main-set figures; `261001 figA_*`: the same figures for the appendix set (.pdf and .png)
