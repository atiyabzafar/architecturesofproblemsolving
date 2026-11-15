# 260930 analysis summary

Default model (learning by testing + recency), N=100, K=30, alpha=2, tau=1, T=2000, long run = last 500 ticks, seeds = 1..10. Shock at t=1500: 100% of the constraint set replaced.

## Networks (means over seeds)

| key        | network                             | <k>  | sd k_in | max k_in | Var(k)/<k>^2 | corr(k_in,k_out) | Lambda/<k> | reciprocity | clustering | k_in = 0 |
|------------|-------------------------------------|------|---------|----------|--------------|------------------|------------|-------------|------------|----------|
| even       | Even (k = 4 for all)                | 4.0  | 0.0     | 4        | 0.00         | nan              | 1.00       | 0.04        | 0.06       | 0%       |
| uneven     | Uneven (heavy tail, mean 4)         | 4.0  | 4.8     | 37       | 1.44         | -0.04            | 0.93       | 0.04        | 0.21       | 0%       |
| aligned    | Uneven, aligned (hubs also talk)    | 4.0  | 4.8     | 37       | 1.44         | 1.00             | 2.05       | 0.22        | 0.30       | 0%       |
| opposed    | Uneven, opposed (hubs listen only)  | 4.0  | 4.8     | 37       | 1.44         | -0.29            | 0.54       | 0.01        | 0.16       | 0%       |
| dense      | Uneven, dense (same shape, mean 12) | 12.0 | 13.8    | 99       | 1.33         | -0.04            | 0.94       | 0.12        | 0.56       | 0%       |
| random     | Random (ER)                         | 3.9  | 1.9     | 9        | 0.24         | 0.03             | 1.00       | 0.03        | 0.07       | 2%       |
| smallworld | Small world (WS)                    | 4.0  | 1.5     | 8        | 0.15         | -0.82            | 0.89       | 0.01        | 0.48       | 1%       |
| scalefree  | Scale free (BA)                     | 3.6  | 4.2     | 25       | 1.35         | 0.76             | 1.54       | 0.03        | 0.25       | 10%      |
| layered    | Layered (3 layers)                  | 3.9  | 1.9     | 9        | 0.24         | 0.02             | 1.00       | 0.06        | 0.10       | 2%       |

## Long run (mean +- s.d. over seeds of the last-500-tick average)

| network    | average violations | best-agent violations | homogeneity  | stale share (%) |
|------------|--------------------|-----------------------|--------------|-----------------|
| even       | 28.87 +- 0.68      | 20.19 +- 0.74         | 0.71 +- 0.02 | 43.62 +- 2.12   |
| uneven     | 29.45 +- 0.39      | 20.25 +- 0.35         | 0.71 +- 0.02 | 49.47 +- 1.44   |
| aligned    | 29.86 +- 0.34      | 20.08 +- 0.49         | 0.69 +- 0.02 | 50.86 +- 1.11   |
| opposed    | 29.77 +- 0.59      | 20.16 +- 0.66         | 0.70 +- 0.02 | 50.90 +- 2.39   |
| dense      | 29.21 +- 0.69      | 19.53 +- 0.66         | 0.71 +- 0.02 | 47.78 +- 1.51   |
| random     | 29.10 +- 0.60      | 19.90 +- 0.45         | 0.71 +- 0.01 | 46.67 +- 1.41   |
| smallworld | 29.48 +- 0.40      | 20.45 +- 0.48         | 0.70 +- 0.01 | 45.52 +- 1.01   |
| scalefree  | 29.96 +- 0.30      | 19.87 +- 0.52         | 0.68 +- 0.01 | 49.42 +- 1.10   |
| layered    | 29.47 +- 0.34      | 20.39 +- 0.46         | 0.69 +- 0.01 | 45.59 +- 0.98   |

## Second-order check: the average against the spread of intake

Intake curve fitted on the main set's agents as V = a + b ln(lambda) + c ln(lambda)^2: V(1) = 29.06, V'(1) = -1.88, V''(1) = 1.43. Predicted average = V(1) + V''(1) Var(lambda) / 2.

| network    | Var(intake) | predicted average | measured average |
|------------|-------------|-------------------|------------------|
| even       | 0.00        | 29.06             | 28.87            |
| uneven     | 1.44        | 30.09             | 29.45            |
| aligned    | 1.44        | 30.09             | 29.86            |
| opposed    | 1.44        | 30.09             | 29.77            |
| dense      | 1.33        | 30.01             | 29.21            |
| random     | 0.24        | 29.23             | 29.10            |
| smallworld | 0.15        | 29.17             | 29.48            |
| scalefree  | 1.35        | 30.03             | 29.96            |
| layered    | 0.24        | 29.23             | 29.47            |

## Position (per run, then mean +- s.d. over seeds)

| network    | Spearman rho(k_in, V) | gap: top 8 minus bottom 8 | best agent's k_in percentile |
|------------|-----------------------|---------------------------|------------------------------|
| even       | n/a (no spread)       | 0.00 +- 0.00              | 50%                          |
| uneven     | -0.53 +- 0.07         | -4.44 +- 1.04             | 99%                          |
| aligned    | -0.57 +- 0.06         | -5.17 +- 0.50             | 99%                          |
| opposed    | -0.55 +- 0.08         | -5.18 +- 0.82             | 99%                          |
| dense      | -0.57 +- 0.06         | -5.34 +- 0.72             | 99%                          |
| random     | -0.58 +- 0.07         | -4.12 +- 1.13             | 83%                          |
| smallworld | -0.46 +- 0.10         | -2.74 +- 0.64             | 77%                          |
| scalefree  | -0.65 +- 0.05         | -6.21 +- 0.94             | 97%                          |
| layered    | -0.56 +- 0.06         | -3.74 +- 0.70             | 79%                          |

Negative rho and gap mean that better-connected agents do better.

## Short run (averages over seeds)

| network    | start-up: ticks to within 1 of long-run level | pre-shock level | peak after shock | shock half-life (ticks) | new clause: ticks to reach 25% of agents | held by, at 30 ticks | at 60 ticks | old clause: ticks to halve believers |
|------------|-----------------------------------------------|-----------------|------------------|-------------------------|------------------------------------------|----------------------|-------------|--------------------------------------|
| even       | 25                                            | 28.5            | 32.9             | 37                      | 50                                       | 15%                  | 29%         | 48                                   |
| uneven     | 34                                            | 29.3            | 32.4             | 35                      | 74                                       | 13%                  | 22%         | 52                                   |
| aligned    | 42                                            | 29.5            | 32.7             | 22                      | 81                                       | 13%                  | 21%         | 53                                   |
| opposed    | 42                                            | 29.8            | 32.9             | 35                      | 83                                       | 13%                  | 21%         | 57                                   |
| dense      | 58                                            | 29.6            | 32.8             | 37                      | 82                                       | 12%                  | 21%         | 51                                   |
| random     | 34                                            | 29.2            | 32.9             | 30                      | 58                                       | 15%                  | 26%         | 47                                   |
| smallworld | 24                                            | 28.8            | 33.0             | 47                      | 56                                       | 14%                  | 27%         | 47                                   |
| scalefree  | 34                                            | 29.5            | 32.4             | 35                      | 104                                      | 11%                  | 18%         | 50                                   |
| layered    | 29                                            | 29.4            | 32.6             | 37                      | 58                                       | 15%                  | 26%         | 46                                   |

-1 means the threshold was not reached within the recorded window.

## Files

- `260930 timeseries.npz`: per (network, seed, mode) time series and cohort curves
- `260930 agents.csv`: per-agent in-degree, out-degree and long-run violations
- `260930 networks.csv`: structural statistics per (network, seed)
- `260930 fig_networks / fig_misinformation / fig_longrun / fig_position / fig_shortrun` (.pdf and .png)
