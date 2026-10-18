"""Command line entry point: simulate the scenarios and write tables and figures.

uv run aimacro --end 2050 --out output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .calibration import DEFAULT_PARAMS, SCENARIOS
from .plots import make_figures
from .report import annual_path, table
from .simulate import simulate


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _fmt(x: float) -> str:
    return f"{x:9.2f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Economic scenarios for transformative AI")
    ap.add_argument(
        "--end",
        type=float,
        default=2050.0,
        help="last date simulated (default 2050.0)",
    )
    ap.add_argument(
        "--dates",
        default="2030,2036,2050",
        help="comma-separated dates at which to print Table-3 layouts (within --end)",
    )
    ap.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    end: float = args.end
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", _fmt)

    dfs = {
        name: simulate(sc, DEFAULT_PARAMS, t_end=end) for name, sc in SCENARIOS.items()
    }
    for name, df in dfs.items():
        df.to_csv(out / f"monthly_{name}.csv")

    requested = {float(x) for x in str(args.dates).split(",") if float(x) <= end}
    for t in sorted(requested | {float(int(end))}):
        tab = table(dfs, t)
        tab.to_csv(out / f"table_{int(t)}.csv")
        _out(f"\n=== The scenarios at the start of {int(t)} (Table 3 layout) ===")
        _out(tab.to_string())

    years = [float(y) for y in range(2026, int(end) + 1)]
    for name, df in dfs.items():
        path = annual_path(df, years)
        path.to_csv(out / f"annual_{name}.csv")
        _out(f"\n=== {name}: annual path, start of each year ===")
        _out(path.to_string())

    if not args.no_plots:
        for f in make_figures(dfs, out):
            _out(f"figure: {f}")


if __name__ == "__main__":
    main()
