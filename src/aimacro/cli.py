"""Command line entry point: simulate the three scenarios and write tables and figures.

    uv run aimacro --end 2050 --out output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .calibration import DEFAULT_PARAMS, SCENARIOS
from .report import annual_path, table
from .simulate import simulate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Economic scenarios for transformative AI")
    ap.add_argument("--end", type=float, default=2050.0, help="last date simulated (default 2050.0)")
    ap.add_argument("--dates", default="2030,2036,2050",
                    help="comma-separated dates at which to print Table-3 layouts (those within --end)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda x: f"{x:9.2f}")

    dfs = {name: simulate(sc, DEFAULT_PARAMS, t_end=args.end) for name, sc in SCENARIOS.items()}
    for name, df in dfs.items():
        df.to_csv(args.out / f"monthly_{name}.csv")

    dates = sorted({float(x) for x in args.dates.split(",") if float(x) <= args.end} | {float(int(args.end))})
    for t in dates:
        tab = table(dfs, t)
        tab.to_csv(args.out / f"table_{int(t)}.csv")
        print(f"\n=== The three scenarios at the start of {int(t)} (Table 3 layout) ===")
        print(tab.to_string())

    years = [float(y) for y in range(2026, int(args.end) + 1)]
    for name, df in dfs.items():
        path = annual_path(df, years)
        path.to_csv(args.out / f"annual_{name}.csv")
        print(f"\n=== {name}: annual path, start of each year ===")
        print(path.to_string())

    if not args.no_plots:
        from .plots import make_figures
        for f in make_figures(dfs, args.out):
            print(f"figure: {f}")


if __name__ == "__main__":
    main()
