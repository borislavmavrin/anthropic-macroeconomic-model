"""Command line entry point: simulate the scenario catalog and write tables and figures.

uv run aimacro --end 2050 --out output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .calibration import (
    AI_SCENARIOS,
    DEFAULT_PARAMS,
    DIAGONAL_SCENARIOS,
    ROBOT_SCENARIOS,
    SCENARIOS,
)
from .plots import make_figures
from .report import annual_path, grid, table
from .simulate import simulate

GRID_ROWS = ("gdp_gap_pct", "avg_wage_gap_pct", "labor_share_pct", "u_all_pct")


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")


def _fmt(x: float) -> str:
    return f"{x:9.2f}"


def _print_date(dfs: dict[str, pd.DataFrame], t: float, out: Path) -> None:
    year = int(t)
    tab = table(dfs, t)
    tab.to_csv(out / f"table_{year}.csv")
    families = (
        ("AI scenarios", AI_SCENARIOS),
        ("Robots-only scenarios", ROBOT_SCENARIOS),
        ("AI and robots combined, matched rungs", DIAGONAL_SCENARIOS),
    )
    for title, family in families:
        cols = [n for n in family if n in dfs]
        if cols:
            _out(f"\n=== {title} at the start of {year} (Table 3 layout) ===")
            _out(tab[cols].to_string())
    grids: list[pd.DataFrame] = []
    for row in GRID_ROWS:
        g = grid(dfs, t, row, list(AI_SCENARIOS), list(ROBOT_SCENARIOS))
        grids.append(g.assign(row=row).set_index("row", append=True))
        _out(f"\n=== {row} at the start of {year}: AI (down) x robots (across) ===")
        _out(g.to_string())
    pd.concat(grids).to_csv(out / f"grid_{year}.csv")


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
    ap.add_argument(
        "--scenarios",
        default="all",
        help="comma-separated scenario names, or 'all' (default)",
    )
    ap.add_argument("--out", type=Path, default=Path("output"), help="output directory")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    end: float = args.end
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    names = (
        list(SCENARIOS) if args.scenarios == "all" else str(args.scenarios).split(",")
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", _fmt)

    dfs = {name: simulate(SCENARIOS[name], DEFAULT_PARAMS, t_end=end) for name in names}
    years = [float(y) for y in range(2026, int(end) + 1)]
    for name, df in dfs.items():
        df.to_csv(out / f"monthly_{name}.csv")
        annual_path(df, years).to_csv(out / f"annual_{name}.csv")

    requested = {float(x) for x in str(args.dates).split(",") if float(x) <= end}
    for t in sorted(requested | {float(int(end))}):
        _print_date(dfs, t, out)

    if not args.no_plots:
        for f in make_figures(dfs, out):
            _out(f"figure: {f}")


if __name__ == "__main__":
    main()
