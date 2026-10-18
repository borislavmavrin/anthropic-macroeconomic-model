"""Reporting in the units of Table 3 of the paper, extended with the all-other group."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .calibration import Params

ROW_LABELS = {
    "gdp_gap_pct": "GDP, pct. above the no-AI path",
    "gdp_index": "GDP, index with 2024 = 100",
    "gdp_growth_pct": "GDP growth, pct. per year",
    "avg_wage_gap_pct": "Average wage, pct. above the no-AI path",
    "cog_wage_gap_pct": "  cognitive occupations, w_C",
    "other_wage_gap_pct": "  all other occupations, w_N",
    "net_return_pct": "Net return to capital r - delta, pct. per year",
    "capital_gap_pct": "Capital stock, pct. above the no-AI path",
    "labor_share_pct": "Labor share, pct. of income",
    "capital_share_pct": "Capital share, pct. of income",
    "labor_income_gap_pct": "Labor income, pct. above the no-AI path",
    "cog_wage_bill_gap_pct": "  wage bill of cognitive occupations",
    "other_wage_bill_gap_pct": "  wage bill of all other occupations",
    "capital_income_gap_pct": "Capital income, pct. above the no-AI path",
    "cog_employment_change_pct": "Cognitive employment, pct. change since mid-2026",
    "other_employment_change_pct": "All-other employment, pct. change since mid-2026",
    "u_cog_pct": "Unemployment rate, cognitive workers, pct.",
    "u_other_pct": "Unemployment rate, all-other workers, pct.",
    "u_all_pct": "Unemployment rate, all workers, pct.",
    "tfp_gap_pct": "Measured TFP, pct. above the no-AI path",
    "tfp_growth_pct": "Measured TFP growth, pct. per year",
    "ideas_gap_pct": "Ideas stock A, pct. above the no-AI path",
    "ideas_growth_pct": "Growth of the ideas stock, pct. per year",
}

_GRID_TOL = 1e-6  # tolerance for matching a date to the monthly grid, in years


def _row(df: pd.DataFrame, t: float) -> dict[str, float]:
    """The simulated variables at date t, as plain floats."""
    index = df.index.to_numpy(dtype=float)
    pos = int(np.argmin(np.abs(index - t)))
    if abs(float(index[pos]) - t) > _GRID_TOL:
        msg = f"date {t} not on the simulation grid"
        raise KeyError(msg)
    return {str(k): float(v) for k, v in df.iloc[pos].items()}


def snapshot(df: pd.DataFrame, t: float, t_ref: float = 2026.5) -> dict[str, float]:
    """The rows of Table 3 at date t.  Growth rates are log changes over the
    twelve months to t (Table 3 note).  Employment change is since t_ref."""
    p: Params = df.attrs["params"]
    r = _row(df, t)
    r_lag = _row(df, t - 1.0)
    r_ref = _row(df, t_ref)
    pct = 100.0
    ex = math.exp
    return {
        "gdp_gap_pct": pct * (ex(r["dlnY"]) - 1.0),
        "gdp_index": pct * ex((p.g + p.n) * (t - p.t0) + r["dlnY"]),
        "gdp_growth_pct": pct * ((p.g + p.n) + r["dlnY"] - r_lag["dlnY"]),
        "avg_wage_gap_pct": pct * (ex(r["dlnw_avg"]) - 1.0),
        "cog_wage_gap_pct": pct * (ex(r["dlnw_C"]) - 1.0),
        "other_wage_gap_pct": pct * (ex(r["dlnw_N"]) - 1.0),
        "net_return_pct": pct * (p.r_bar * ex(r["x_r"]) - p.delta),
        "capital_gap_pct": pct * (ex(r["dlnK"]) - 1.0),
        "labor_share_pct": pct * r["sL"],
        "capital_share_pct": pct * (1.0 - r["sL"]),
        "labor_income_gap_pct": pct * (ex(r["dln_labor_income"]) - 1.0),
        "cog_wage_bill_gap_pct": pct * (ex(r["dln_cog_wage_bill"]) - 1.0),
        "other_wage_bill_gap_pct": pct * (ex(r["dln_other_wage_bill"]) - 1.0),
        "capital_income_gap_pct": pct * (ex(r["x_r"] + r["dlnK"]) - 1.0),
        "cog_employment_change_pct": pct * (r["lC"] / r_ref["lC"] - 1.0),
        "other_employment_change_pct": pct * (r["lN"] / r_ref["lN"] - 1.0),
        "u_cog_pct": pct * r["UC"] / (r["lC"] + r["UC"]),
        "u_other_pct": pct * r["UN"] / (r["lN"] + r["UN"]),
        "u_all_pct": pct * (r["UC"] + r["UN"]),
        "tfp_gap_pct": pct * (ex(r["tfp"]) - 1.0),
        "tfp_growth_pct": pct * (p.gA + r["tfp"] - r_lag["tfp"]),
        "ideas_gap_pct": pct * (ex(r["dlnA"]) - 1.0),
        "ideas_growth_pct": pct * (p.g + r["dlnA"] - r_lag["dlnA"]),
    }


def table(dfs: dict[str, pd.DataFrame], t: float) -> pd.DataFrame:
    """Table-3 layout: one column per scenario at date t."""
    cols = {name: snapshot(df, t) for name, df in dfs.items()}
    return pd.DataFrame(cols).rename(index=ROW_LABELS)


def annual_path(df: pd.DataFrame, years: list[float]) -> pd.DataFrame:
    """Snapshots at several dates, one column per date."""
    return pd.DataFrame({y: snapshot(df, y) for y in years}).rename(index=ROW_LABELS)


def grid(
    dfs: dict[str, pd.DataFrame],
    t: float,
    row: str,
    ai_names: list[str],
    robot_names: list[str],
) -> pd.DataFrame:
    """One Table-3 row for every AI x robot combination: AI scenarios down,
    robot scenarios across, with the single-shock scenarios as a first row and
    column where they are present in `dfs`."""

    def cell(name: str) -> float:
        return snapshot(dfs[name], t)[row] if name in dfs else math.nan

    rows = ["none", *ai_names]
    cols = ["none", *robot_names]
    data = {c: [cell(_combo(a, c)) for a in rows] for c in cols}
    return pd.DataFrame(data, index=pd.Index(rows, name="AI \\ robots"))


def _combo(ai: str, robots: str) -> str:
    if ai == "none":
        return robots
    if robots == "none":
        return ai
    return f"{ai}+{robots}"
