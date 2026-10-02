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
| even    | 28.51 +- 0.11      | 19.83 +- 0.12         | 0.718 +- 0.003 | 43.3 +- 0.2       |
| uneven  | 29.53 +- 0.07      | 20.05 +- 0.09         | 0.703 +- 0.003 | 49.3 +- 0.2       |
| dense   | 29.51 +- 0.10      | 19.87 +- 0.13         | 0.700 +- 0.003 | 48.8 +- 0.2       |

## Second-order check

Curve fitted on the main set: V(1) = 28.78, V'(1) = -2.05, V''(1) = 1.87; predicted average = V(1) + V''(1) Var(intake) / 2.

| network | Var(intake) | predicted average | measured average |
|---------|-------------|-------------------|------------------|
| even    | 0.00        | 28.78             | 28.51 +- 0.11    |
| uneven  | 1.44        | 30.13             | 29.53 +- 0.07    |
| dense   | 1.33        | 30.02             | 29.51 +- 0.10    |

## Position (per run; mean +- 1 s.e. over seeds)

| network | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|---------|-----------------------|-------------------|---------------------------|-----------------------------------|
| even    | none (no spread)      | none              | none                      | -                                 |
| uneven  | -0.58 +- 0.01         | 100%              | -5.18 +- 0.15             | 99%                               |
| dense   | -0.61 +- 0.01         | 100%              | -5.28 +- 0.12             | 98%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|---------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| even    | 28                                            | 28.51 +- 0.18    | 32.15 +- 0.16    | 30.63 +- 0.17     | 29.39 +- 0.16      | 29.01 +- 0.17       | 28.65 +- 0.14       | 28.55 +- 0.06        |
| uneven  | 46                                            | 29.46 +- 0.11    | 32.24 +- 0.20    | 30.93 +- 0.19     | 30.06 +- 0.14      | 29.70 +- 0.15       | 29.55 +- 0.13       | 29.45 +- 0.08        |
| dense   | 36                                            | 29.40 +- 0.16    | 32.17 +- 0.15    | 30.89 +- 0.18     | 29.92 +- 0.13      | 29.56 +- 0.14       | 29.32 +- 0.13       | 29.47 +- 0.09        |

## Short run: new and retired clauses after the shock

| network | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|---------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| even    | 15.5 +- 0.5                         | 28.7 +- 0.5                         | 39.5 +- 0.8                          | 51                     | 41%        | 47                                       | 31%                     |
| uneven  | 12.5 +- 0.4                         | 22.0 +- 0.7                         | 28.7 +- 0.9                          | 78                     | 32%        | 51                                       | 22%                     |
| dense   | 12.3 +- 0.3                         | 21.3 +- 0.6                         | 28.3 +- 0.8                          | 77                     | 31%        | 51                                       | 21%                     |

## Differences between main-set networks (Welch; the run is the unit)

| pair           | quantity                                        | difference | s.e. | t     |
|----------------|-------------------------------------------------|------------|------|-------|
| even - uneven  | average violations                              | -1.03      | 0.13 | -8.1  |
| even - uneven  | best-agent violations                           | -0.22      | 0.15 | -1.5  |
| even - uneven  | stale beliefs (%)                               | -6.02      | 0.28 | -21.3 |
| even - uneven  | pre-shock level                                 | -0.94      | 0.21 | -4.5  |
| even - uneven  | level, ticks 1-20 after shock                   | -0.10      | 0.25 | -0.4  |
| even - uneven  | level, ticks 21-60 after shock                  | -0.30      | 0.26 | -1.1  |
| even - uneven  | level, ticks 61-120 after shock                 | -0.67      | 0.21 | -3.2  |
| even - uneven  | level, ticks 121-200 after shock                | -0.69      | 0.22 | -3.1  |
| even - uneven  | level, ticks 201-400 after shock                | -0.90      | 0.19 | -4.6  |
| even - uneven  | level, ticks 401-1000 after shock               | -0.90      | 0.10 | -8.7  |
| even - uneven  | excess over own pre-shock level, ticks 1-20     | +0.85      | 0.30 | +2.8  |
| even - uneven  | excess over own pre-shock level, ticks 21-60    | +0.65      | 0.29 | +2.2  |
| even - uneven  | excess over own pre-shock level, ticks 61-120   | +0.27      | 0.26 | +1.0  |
| even - uneven  | excess over own pre-shock level, ticks 121-200  | +0.25      | 0.29 | +0.9  |
| even - uneven  | excess over own pre-shock level, ticks 201-400  | +0.04      | 0.25 | +0.2  |
| even - uneven  | excess over own pre-shock level, ticks 401-1000 | +0.04      | 0.22 | +0.2  |
| even - uneven  | new clause held by (%), 30 ticks on             | +3.01      | 0.60 | +5.0  |
| even - uneven  | new clause held by (%), 60 ticks on             | +6.68      | 0.84 | +8.0  |
| even - uneven  | new clause held by (%), 100 ticks on            | +10.79     | 1.25 | +8.7  |
| dense - uneven | average violations                              | -0.02      | 0.12 | -0.1  |
| dense - uneven | best-agent violations                           | -0.18      | 0.15 | -1.2  |
| dense - uneven | stale beliefs (%)                               | -0.56      | 0.27 | -2.0  |
| dense - uneven | pre-shock level                                 | -0.06      | 0.19 | -0.3  |
| dense - uneven | level, ticks 1-20 after shock                   | -0.07      | 0.25 | -0.3  |
| dense - uneven | level, ticks 21-60 after shock                  | -0.04      | 0.27 | -0.1  |
| dense - uneven | level, ticks 61-120 after shock                 | -0.14      | 0.19 | -0.7  |
| dense - uneven | level, ticks 121-200 after shock                | -0.14      | 0.20 | -0.7  |
| dense - uneven | level, ticks 201-400 after shock                | -0.23      | 0.18 | -1.2  |
| dense - uneven | level, ticks 401-1000 after shock               | +0.02      | 0.12 | +0.2  |
| dense - uneven | excess over own pre-shock level, ticks 1-20     | -0.01      | 0.29 | -0.0  |
| dense - uneven | excess over own pre-shock level, ticks 21-60    | +0.02      | 0.31 | +0.1  |
| dense - uneven | excess over own pre-shock level, ticks 61-120   | -0.08      | 0.25 | -0.3  |
| dense - uneven | excess over own pre-shock level, ticks 121-200  | -0.08      | 0.26 | -0.3  |
| dense - uneven | excess over own pre-shock level, ticks 201-400  | -0.17      | 0.25 | -0.7  |
| dense - uneven | excess over own pre-shock level, ticks 401-1000 | +0.08      | 0.21 | +0.4  |
| dense - uneven | new clause held by (%), 30 ticks on             | -0.21      | 0.50 | -0.4  |
| dense - uneven | new clause held by (%), 60 ticks on             | -0.63      | 0.89 | -0.7  |
| dense - uneven | new clause held by (%), 100 ticks on            | -0.40      | 1.23 | -0.3  |
| even - dense   | average violations                              | -1.01      | 0.15 | -6.9  |
| even - dense   | best-agent violations                           | -0.04      | 0.17 | -0.2  |
| even - dense   | stale beliefs (%)                               | -5.47      | 0.25 | -21.5 |
| even - dense   | pre-shock level                                 | -0.88      | 0.24 | -3.7  |
| even - dense   | level, ticks 1-20 after shock                   | -0.03      | 0.22 | -0.1  |
| even - dense   | level, ticks 21-60 after shock                  | -0.26      | 0.25 | -1.0  |
| even - dense   | level, ticks 61-120 after shock                 | -0.53      | 0.20 | -2.6  |
| even - dense   | level, ticks 121-200 after shock                | -0.55      | 0.22 | -2.5  |
| even - dense   | level, ticks 201-400 after shock                | -0.67      | 0.19 | -3.5  |
| even - dense   | level, ticks 401-1000 after shock               | -0.92      | 0.11 | -8.8  |
| even - dense   | excess over own pre-shock level, ticks 1-20     | +0.86      | 0.33 | +2.6  |
| even - dense   | excess over own pre-shock level, ticks 21-60    | +0.62      | 0.34 | +1.8  |
| even - dense   | excess over own pre-shock level, ticks 61-120   | +0.35      | 0.29 | +1.2  |
| even - dense   | excess over own pre-shock level, ticks 121-200  | +0.33      | 0.31 | +1.1  |
| even - dense   | excess over own pre-shock level, ticks 201-400  | +0.21      | 0.28 | +0.7  |
| even - dense   | excess over own pre-shock level, ticks 401-1000 | -0.04      | 0.25 | -0.2  |
| even - dense   | new clause held by (%), 30 ticks on             | +3.21      | 0.59 | +5.5  |
| even - dense   | new clause held by (%), 60 ticks on             | +7.31      | 0.80 | +9.1  |
| even - dense   | new clause held by (%), 100 ticks on            | +11.19     | 1.13 | +9.9  |

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
| random     | 29.09 +- 0.09      | 20.03 +- 0.10         | 0.708 +- 0.003 | 46.1 +- 0.2       |
| smallworld | 29.19 +- 0.08      | 20.20 +- 0.10         | 0.702 +- 0.002 | 45.3 +- 0.2       |
| scalefree  | 29.76 +- 0.09      | 20.14 +- 0.12         | 0.694 +- 0.003 | 49.5 +- 0.2       |
| layered    | 29.32 +- 0.09      | 20.35 +- 0.09         | 0.700 +- 0.002 | 46.2 +- 0.2       |

## Second-order check

Curve fitted on the main set: V(1) = 28.78, V'(1) = -2.05, V''(1) = 1.87; predicted average = V(1) + V''(1) Var(intake) / 2.

| network    | Var(intake) | predicted average | measured average |
|------------|-------------|-------------------|------------------|
| random     | 0.24        | 29.01             | 29.09 +- 0.09    |
| smallworld | 0.15        | 28.92             | 29.19 +- 0.08    |
| scalefree  | 1.16        | 29.87             | 29.76 +- 0.09    |
| layered    | 0.24        | 29.01             | 29.32 +- 0.09    |

## Position (per run; mean +- 1 s.e. over seeds)

| network    | Spearman rho(k_in, V) | runs with rho < 0 | gap: top 8 minus bottom 8 | best agent's in-degree percentile |
|------------|-----------------------|-------------------|---------------------------|-----------------------------------|
| random     | -0.54 +- 0.01         | 100%              | -3.93 +- 0.16             | 83%                               |
| smallworld | -0.47 +- 0.01         | 100%              | -2.88 +- 0.11             | 74%                               |
| scalefree  | -0.62 +- 0.01         | 100%              | -5.28 +- 0.14             | 97%                               |
| layered    | -0.54 +- 0.02         | 100%              | -3.63 +- 0.18             | 79%                               |

## Short run: average violations around the shock (mean +- 1 s.e. over seeds)

| network    | start-up: ticks to within 1 of long-run level | 100 ticks before | ticks 1-20 after | ticks 21-60 after | ticks 61-120 after | ticks 121-200 after | ticks 201-400 after | ticks 401-1000 after |
|------------|-----------------------------------------------|------------------|------------------|-------------------|--------------------|---------------------|---------------------|----------------------|
| random     | 31                                            | 29.19 +- 0.12    | 32.57 +- 0.15    | 31.01 +- 0.15     | 29.65 +- 0.14      | 29.15 +- 0.12       | 29.14 +- 0.12       | 29.24 +- 0.08        |
| smallworld | 35                                            | 29.05 +- 0.15    | 32.31 +- 0.16    | 30.80 +- 0.16     | 29.61 +- 0.13      | 29.13 +- 0.11       | 29.19 +- 0.13       | 29.01 +- 0.10        |
| scalefree  | 43                                            | 29.86 +- 0.13    | 32.49 +- 0.15    | 31.23 +- 0.15     | 30.22 +- 0.14      | 30.06 +- 0.13       | 29.90 +- 0.10       | 29.88 +- 0.07        |
| layered    | 29                                            | 29.12 +- 0.13    | 32.12 +- 0.13    | 30.77 +- 0.14     | 29.54 +- 0.14      | 29.15 +- 0.14       | 29.15 +- 0.11       | 29.11 +- 0.07        |

## Short run: new and retired clauses after the shock

| network    | new clause held by (%), 30 ticks on | new clause held by (%), 60 ticks on | new clause held by (%), 100 ticks on | ticks to 25% of agents | peak share | retired clause: ticks to halve believers | peak share disbelieving |
|------------|-------------------------------------|-------------------------------------|--------------------------------------|------------------------|------------|------------------------------------------|-------------------------|
| random     | 13.7 +- 0.3                         | 25.3 +- 0.6                         | 30.5 +- 0.6                          | 60                     | 32%        | 46                                       | 22%                     |
| smallworld | 14.6 +- 0.4                         | 26.9 +- 0.5                         | 34.8 +- 0.9                          | 55                     | 35%        | 46                                       | 24%                     |
| scalefree  | 11.5 +- 0.3                         | 19.9 +- 0.5                         | 25.7 +- 0.6                          | 96                     | 28%        | 49                                       | 18%                     |
| layered    | 13.4 +- 0.4                         | 24.7 +- 0.5                         | 29.8 +- 0.8                          | 62                     | 30%        | 46                                       | 22%                     |

## Files

- `261001 timeseries.npz`, `261001 agents.csv`, `261001 networks.csv`: raw results
- `261001 fig_*`: main-set figures; `261001 figA_*`: the same figures for the appendix set (.pdf and .png)
