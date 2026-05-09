# Problem solving and organisation structure — draft folder

Clean working directory for the May-2026 draft. Self-contained: the only
dependency on the parent folder is `model.py`, which is imported via a
`sys.path` insert at the top of `analysis.py`.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper draft, following the new outline (`260508 tentative new outline.md`). Most numbers are placeholders; figures point at `output/figures/...` produced by the analysis script. |
| `refs.bib` | Bibliography (copied from the previous paper; trimmed where references are no longer cited). |
| `analysis.py` | Self-contained analysis script. Builds five canonical network topologies, runs the model on each across multiple seeds, computes steady-state and transient metrics, optionally runs a shock-recovery experiment, and saves an xlsx + figures. |
| `output/topology_results.xlsx` | Numerical results (one sheet per metric set, plus per-network downsampled time series). |
| `output/figures/*.png` | Plots referenced in the paper. |

## How to reproduce

### Quick smoke test (~1 minute)

```
cd "G:\My Drive\phd works\santafe project\architecturesofproblemsolving\model\260508 draft"
py -u analysis.py --quick
```

Runs T=500 with 2 seeds. Confirms the pipeline is intact and gives a rough preview of the qualitative pattern.

### Full run (~30-40 minutes, no shock experiment)

```
py -u analysis.py
```

Produces `output/topology_results.xlsx` and `output/figures/*.png` — six figures total, sufficient to populate the four time-series panels and two summary bar charts in the paper.

### Full run with shock recovery (~50-70 minutes)

```
py -u analysis.py --shock
```

Adds the `shock_recovery` sheet to the xlsx and the `shock_recovery.png` figure that the paper references in section 3.3.

## Configuration knobs (top of `analysis.py`)

```python
DEFAULT_T_TOTAL = 10_000     # ticks per run
DEFAULT_BURN_IN = 5_000      # discarded as transient when computing "long" metrics
DEFAULT_SEEDS   = [42, 43, 44, 45, 46]
N         = 80
K         = 30
ALPHA     = 2
OBS_PROB  = 0.01
CINT      = 10
TRANSIENT_WINDOW = 1000      # window for transient metrics
```

To change networks: edit the five `network_*` builder functions and the `NETWORK_BUILDERS` list.

## What the paper depends on

The paper as currently drafted references seven figures:

- `output/figures/timeseries_avg_V.png`
- `output/figures/timeseries_min_V.png`
- `output/figures/timeseries_gini_V.png`
- `output/figures/timeseries_H.png`
- `output/figures/steady_state_summary.png`
- `output/figures/transient_summary.png`
- `output/figures/shock_recovery.png` (only if `--shock` was used)

After running `analysis.py [--shock]` once, all seven exist with current data and the paper compiles.

## Things still to do before submission

1. **Run the full analysis** (`py -u analysis.py --shock`) to fill in real numbers in Table 1 and update the verbal descriptions if any quantitative claims need adjusting.
2. **Decide whether to keep the empirical-network appendix.** The paper as-written treats this as optional; the analysis script doesn't currently include empirical-network runs. If you decide to include them, the `data/` folder upstream has the cleaned graphs (`EnronGraphWithData.graphml`, `EmailManufacturing.xml`, etc.) — wiring them into `analysis.py` is half a day of work.
3. **Sensitivity analyses.** The paper mentions sweeps over $\alpha$, $p_{\text{obs}}$, $\tau$ as a robustness check (Appendix B). The previous folder's `long_sweep_sf_vs_random.py` already does these for SF and Random; extending to the five canonical topologies is straightforward.
4. **Read-through pass.** The introduction borrows structure from the previous paper but the framing is genuinely new — worth a careful pass to make sure the new claims are correctly positioned against the existing literature.
