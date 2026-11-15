# 261001 analysis summary: main set

Default model (learning by testing + recency), N=100, K=30, alpha=2, tau=1; long-run experiment T=2000, last 500 ticks; shock experiment: 100% of the constraint set replaced at t=1500, run to t=2500; seeds 1..100. All uncertainties use the run as the unit of observation.

## Networks (means over seeds)

| network | <k>  | sd k_in | max k_in | Var(intake) | corr(k_in,k_out) | clustering | no in-link | largest strongly connected part |
|---------|------|---------|----------|-------------|------------------|------------|------------|---------------------------------|
| even    | 4.0  | 0.0     | 4        | 0.00        | none             | 0.06       | 0%         | 100%                            |
| uneven  | 4.0  | 4.8     | 37       | 1.44        | -0.01            | 0.21       | 0%         | 100%                            |
| dense   | 12.0 | 13.8    | 99       | 1.33        | -0.01            | 0.56       | 0%         | 100%                            |

## Long run (last 500 ticks; mean +- 1 s.e. over 100 seeds)

| network | average violations | best-agent violations | homogeneity    | stale beliefs (%) |
|---------|--------------------|-----------------------|----------------|-------------------|
| even    | 28.71 +- 0.07      | 20.09 +- 0.07         | 0.712 +- 0.002 | 43.4 +- 0.1       |
| uneven  | 29.47 +- 0.05      | 19.90 +- 0.06         | 0.702 +- 0.002 | 49.2 +- 0.1       |
| dense   | 29.47 +- 0.06      | 19.92 +- 0.07         | 0.705 +- 0.002 | 48.8 +- 0.1       |

## Second-order check

Curve fitted on the main set: V(1) = 28.86, V'(1) = -1.99, V''(1) = 1.63; predicted average = V(1) + V''(1) Var(intake) / 2.

| network | Var(intake) | predicted average | measured average |
|---------|-------------|-------------------|------------------|
| even    | 0.00        | 28.86             | 28.71 +- 0.07    |
| uneven  | 1.44        | 30.04             | 29.47 +- 0.05    |
| dense   | 1.33        | 29.94             | 29.47 +- 0.06    |

## Position (per run; mean +- 1 s.e. over seeds)

| network | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|---------|-----------------------|-------------------|---------------------------|-----------------------------------|
| even    | none (no spread)      | none              | none                      | -                                 |
| uneven  | -0.58 +- 0.01         | 100%              | -5.14 +- 0.08             | 99%                               |
| dense   | -0.58 +- 0.01         | 100%              | -5.24 +- 0.08             | 98%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|---------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| even    | 26                                            | 28.69 +- 0.10    | 32.24 +- 0.14    | 30.58 +- 0.12     | 29.34 +- 0.09      | 28.75 +- 0.11       | 28.65 +- 0.08       | 28.63 +- 0.06        |
| uneven  | 43                                            | 29.53 +- 0.09    | 32.41 +- 0.09    | 30.91 +- 0.08     | 29.87 +- 0.10      | 29.56 +- 0.09       | 29.51 +- 0.07       | 29.38 +- 0.05        |
| dense   | 36                                            | 29.53 +- 0.09    | 32.38 +- 0.10    | 31.11 +- 0.10     | 30.05 +- 0.10      | 29.60 +- 0.09       | 29.70 +- 0.07       | 29.40 +- 0.06        |

## Short run: new and retired clauses after the shock

| network | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|---------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| even    | 15.4 +- 0.2                         | 28.5 +- 0.4                         | 38.5 +- 0.5                          | 52                     | 39%        | 47                                       | 31%                     |
| uneven  | 12.8 +- 0.2                         | 22.1 +- 0.3                         | 28.4 +- 0.5                          | 76                     | 31%        | 52                                       | 21%                     |
| dense   | 12.7 +- 0.2                         | 21.6 +- 0.4                         | 27.9 +- 0.6                          | 78                     | 31%        | 50                                       | 20%                     |

## Differences between main-set networks (Welch; the run is the unit)

| pair           | quantity                                        | difference | s.e. | t     |
|----------------|-------------------------------------------------|------------|------|-------|
| even - uneven  | average violations                              | -0.77      | 0.09 | -8.8  |
| even - uneven  | best-agent violations                           | +0.19      | 0.09 | +2.1  |
| even - uneven  | stale beliefs (%)                               | -5.84      | 0.20 | -29.5 |
| even - uneven  | pre-shock level                                 | -0.84      | 0.13 | -6.3  |
| even - uneven  | level, ticks 1-20 after shock                   | -0.17      | 0.17 | -1.0  |
| even - uneven  | level, ticks 21-60 after shock                  | -0.33      | 0.14 | -2.3  |
| even - uneven  | level, ticks 61-120 after shock                 | -0.53      | 0.13 | -4.0  |
| even - uneven  | level, ticks 121-200 after shock                | -0.81      | 0.14 | -5.6  |
| even - uneven  | level, ticks 201-400 after shock                | -0.87      | 0.10 | -8.4  |
| even - uneven  | level, ticks 401-1000 after shock               | -0.75      | 0.08 | -9.9  |
| even - uneven  | excess over own pre-shock level, ticks 1-20     | +0.67      | 0.19 | +3.6  |
| even - uneven  | excess over own pre-shock level, ticks 21-60    | +0.51      | 0.18 | +2.9  |
| even - uneven  | excess over own pre-shock level, ticks 61-120   | +0.31      | 0.18 | +1.7  |
| even - uneven  | excess over own pre-shock level, ticks 121-200  | +0.03      | 0.19 | +0.2  |
| even - uneven  | excess over own pre-shock level, ticks 201-400  | -0.03      | 0.17 | -0.2  |
| even - uneven  | excess over own pre-shock level, ticks 401-1000 | +0.09      | 0.15 | +0.6  |
| even - uneven  | new clause held by (%), 30 ticks on             | +2.66      | 0.34 | +7.7  |
| even - uneven  | new clause held by (%), 60 ticks on             | +6.41      | 0.54 | +11.8 |
| even - uneven  | new clause held by (%), 100 ticks on            | +10.12     | 0.75 | +13.6 |
| dense - uneven | average violations                              | -0.00      | 0.08 | -0.1  |
| dense - uneven | best-agent violations                           | +0.02      | 0.09 | +0.2  |
| dense - uneven | stale beliefs (%)                               | -0.43      | 0.20 | -2.2  |
| dense - uneven | pre-shock level                                 | +0.00      | 0.13 | +0.0  |
| dense - uneven | level, ticks 1-20 after shock                   | -0.03      | 0.14 | -0.2  |
| dense - uneven | level, ticks 21-60 after shock                  | +0.20      | 0.13 | +1.5  |
| dense - uneven | level, ticks 61-120 after shock                 | +0.18      | 0.14 | +1.3  |
| dense - uneven | level, ticks 121-200 after shock                | +0.04      | 0.13 | +0.3  |
| dense - uneven | level, ticks 201-400 after shock                | +0.19      | 0.10 | +1.9  |
| dense - uneven | level, ticks 401-1000 after shock               | +0.01      | 0.08 | +0.2  |
| dense - uneven | excess over own pre-shock level, ticks 1-20     | -0.03      | 0.16 | -0.2  |
| dense - uneven | excess over own pre-shock level, ticks 21-60    | +0.20      | 0.17 | +1.2  |
| dense - uneven | excess over own pre-shock level, ticks 61-120   | +0.18      | 0.19 | +1.0  |
| dense - uneven | excess over own pre-shock level, ticks 121-200  | +0.04      | 0.18 | +0.2  |
| dense - uneven | excess over own pre-shock level, ticks 201-400  | +0.19      | 0.16 | +1.2  |
| dense - uneven | excess over own pre-shock level, ticks 401-1000 | +0.01      | 0.15 | +0.1  |
| dense - uneven | new clause held by (%), 30 ticks on             | -0.08      | 0.33 | -0.2  |
| dense - uneven | new clause held by (%), 60 ticks on             | -0.48      | 0.55 | -0.9  |
| dense - uneven | new clause held by (%), 100 ticks on            | -0.48      | 0.79 | -0.6  |
| even - dense   | average violations                              | -0.76      | 0.09 | -8.2  |
| even - dense   | best-agent violations                           | +0.17      | 0.10 | +1.7  |
| even - dense   | stale beliefs (%)                               | -5.41      | 0.19 | -27.9 |
| even - dense   | pre-shock level                                 | -0.84      | 0.14 | -6.2  |
| even - dense   | level, ticks 1-20 after shock                   | -0.14      | 0.17 | -0.8  |
| even - dense   | level, ticks 21-60 after shock                  | -0.53      | 0.16 | -3.4  |
| even - dense   | level, ticks 61-120 after shock                 | -0.72      | 0.14 | -5.3  |
| even - dense   | level, ticks 121-200 after shock                | -0.85      | 0.15 | -5.8  |
| even - dense   | level, ticks 201-400 after shock                | -1.06      | 0.11 | -10.1 |
| even - dense   | level, ticks 401-1000 after shock               | -0.77      | 0.09 | -9.0  |
| even - dense   | excess over own pre-shock level, ticks 1-20     | +0.70      | 0.18 | +3.8  |
| even - dense   | excess over own pre-shock level, ticks 21-60    | +0.31      | 0.18 | +1.7  |
| even - dense   | excess over own pre-shock level, ticks 61-120   | +0.13      | 0.18 | +0.7  |
| even - dense   | excess over own pre-shock level, ticks 121-200  | -0.01      | 0.19 | -0.1  |
| even - dense   | excess over own pre-shock level, ticks 201-400  | -0.22      | 0.18 | -1.2  |
| even - dense   | excess over own pre-shock level, ticks 401-1000 | +0.08      | 0.16 | +0.5  |
| even - dense   | new clause held by (%), 30 ticks on             | +2.74      | 0.34 | +8.1  |
| even - dense   | new clause held by (%), 60 ticks on             | +6.89      | 0.59 | +11.7 |
| even - dense   | new clause held by (%), 100 ticks on            | +10.60     | 0.77 | +13.7 |

# Appendix set: standard generators

## Networks (means over seeds)

| network    | <k> | sd k_in | max k_in | Var(intake) | corr(k_in,k_out) | clustering | no in-link | largest strongly connected part |
|------------|-----|---------|----------|-------------|------------------|------------|------------|---------------------------------|
| random     | 4.0 | 1.9     | 10       | 0.24        | 0.00             | 0.08       | 2%         | 96%                             |
| smallworld | 4.0 | 1.5     | 8        | 0.15        | -0.83            | 0.49       | 1%         | 99%                             |
| scalefree  | 3.7 | 3.9     | 24       | 1.17        | 0.74             | 0.23       | 8%         | 89%                             |
| layered    | 3.9 | 1.9     | 10       | 0.24        | 0.00             | 0.09       | 2%         | 96%                             |

## Long run (last 500 ticks; mean +- 1 s.e. over 100 seeds)

| network    | average violations | best-agent violations | homogeneity    | stale beliefs (%) |
|------------|--------------------|-----------------------|----------------|-------------------|
| random     | 29.21 +- 0.06      | 20.12 +- 0.07         | 0.704 +- 0.002 | 46.3 +- 0.1       |
| smallworld | 28.98 +- 0.06      | 20.02 +- 0.07         | 0.707 +- 0.002 | 45.1 +- 0.1       |
| scalefree  | 29.89 +- 0.05      | 20.24 +- 0.07         | 0.691 +- 0.002 | 49.4 +- 0.1       |
| layered    | 29.34 +- 0.05      | 20.27 +- 0.06         | 0.701 +- 0.002 | 46.4 +- 0.1       |

## Second-order check

Curve fitted on the main set: V(1) = 28.86, V'(1) = -1.99, V''(1) = 1.63; predicted average = V(1) + V''(1) Var(intake) / 2.

| network    | Var(intake) | predicted average | measured average |
|------------|-------------|-------------------|------------------|
| random     | 0.24        | 29.06             | 29.21 +- 0.06    |
| smallworld | 0.15        | 28.98             | 28.98 +- 0.06    |
| scalefree  | 1.17        | 29.81             | 29.89 +- 0.05    |
| layered    | 0.24        | 29.06             | 29.34 +- 0.05    |

## Position (per run; mean +- 1 s.e. over seeds)

| network    | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|------------|-----------------------|-------------------|---------------------------|-----------------------------------|
| random     | -0.57 +- 0.01         | 100%              | -3.74 +- 0.09             | 83%                               |
| smallworld | -0.48 +- 0.01         | 100%              | -2.99 +- 0.09             | 78%                               |
| scalefree  | -0.63 +- 0.01         | 100%              | -5.48 +- 0.09             | 95%                               |
| layered    | -0.55 +- 0.01         | 100%              | -3.57 +- 0.09             | 81%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network    | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|------------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| random     | 30                                            | 29.16 +- 0.10    | 32.21 +- 0.10    | 30.78 +- 0.10     | 29.68 +- 0.10      | 29.44 +- 0.09       | 29.28 +- 0.07       | 29.26 +- 0.05        |
| smallworld | 34                                            | 28.92 +- 0.10    | 32.40 +- 0.12    | 30.82 +- 0.11     | 29.56 +- 0.10      | 29.26 +- 0.11       | 28.85 +- 0.08       | 29.06 +- 0.06        |
| scalefree  | 40                                            | 29.85 +- 0.08    | 32.34 +- 0.10    | 31.10 +- 0.08     | 30.27 +- 0.09      | 29.91 +- 0.09       | 29.87 +- 0.06       | 29.89 +- 0.04        |
| layered    | 29                                            | 29.14 +- 0.11    | 32.49 +- 0.11    | 30.94 +- 0.09     | 29.91 +- 0.08      | 29.40 +- 0.09       | 29.27 +- 0.08       | 29.26 +- 0.05        |

## Short run: new and retired clauses after the shock

| network    | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|------------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| random     | 13.9 +- 0.2                         | 24.8 +- 0.4                         | 31.1 +- 0.4                          | 61                     | 32%        | 47                                       | 21%                     |
| smallworld | 14.2 +- 0.2                         | 26.1 +- 0.4                         | 33.5 +- 0.5                          | 57                     | 33%        | 46                                       | 24%                     |
| scalefree  | 11.6 +- 0.2                         | 19.3 +- 0.3                         | 24.7 +- 0.4                          | 104                    | 26%        | 50                                       | 18%                     |
| layered    | 13.8 +- 0.2                         | 24.6 +- 0.4                         | 30.4 +- 0.5                          | 62                     | 31%        | 47                                       | 22%                     |

## Files

- `261001 timeseries.npz`, `261001 agents.csv`, `261001 networks.csv`: raw results
- `261001 fig_*`: main-set figures; `261001 figA_*`: the same figures for the appendix set (.pdf and .png)
