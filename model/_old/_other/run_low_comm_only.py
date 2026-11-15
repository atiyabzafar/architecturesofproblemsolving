"""
Run ONLY the low_comm time-series config (~5 minutes), without redoing the
full sweep. Appends a ts_low_comm tab to the existing long_sweep_results.xlsx
and saves the two PNGs alongside the others.

Usage:
    py -u run_low_comm_only.py
"""

import time
from pathlib import Path

import pandas as pd

from long_sweep_sf_vs_random import (
    SF_DEFAULT, RAND_DEFAULT,
    run_config_timeseries,
    plot_timeseries_one,
)


def main():
    out_dir = Path(__file__).parent / "output" / "long sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "long_sweep_results.xlsx"

    print(f"Outputs will be saved under: {out_dir}", flush=True)

    # ---- Run the time-series ----
    t0 = time.time()
    ts_df = run_config_timeseries(
        "low_comm",
        SF_DEFAULT, RAND_DEFAULT,
        overrides={"comm_rate": 0.1},
    )
    print(f"\nFinished in {(time.time() - t0)/60:.1f} minutes.", flush=True)

    # ---- Append to existing xlsx (or create if missing) ----
    if xlsx_path.exists():
        with pd.ExcelWriter(xlsx_path, engine="openpyxl",
                            mode="a", if_sheet_exists="replace") as w:
            ts_df.to_excel(w, sheet_name="ts_low_comm", index=False)
        print(f"-> appended ts_low_comm tab to {xlsx_path}", flush=True)
    else:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            ts_df.to_excel(w, sheet_name="ts_low_comm", index=False)
        print(f"-> created {xlsx_path} with ts_low_comm tab", flush=True)

    # ---- Plots ----
    plot_timeseries_one(ts_df, "avg", "low_comm",
                        outfile=out_dir / "timeseries_low_comm_avg.png",
                        ylabel="avg violations")
    plot_timeseries_one(ts_df, "min", "low_comm",
                        outfile=out_dir / "timeseries_low_comm_min.png",
                        ylabel="min violations")
    print(f"-> saved timeseries_low_comm_avg.png and _min.png", flush=True)


if __name__ == "__main__":
    main()
