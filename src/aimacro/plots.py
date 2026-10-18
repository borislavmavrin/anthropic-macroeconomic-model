"""Figures: the paper's Figures 2-4 (2025-2030) and their extension to the simulated
horizon for each scenario family, and the exogenous shock paths."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

from .calibration import (
    AI_SCENARIOS,
    DIAGONAL_SCENARIOS,
    PAPER_SCENARIOS,
    ROBOT_SCENARIOS,
)
from .plots_common import (
    PAPER_END,
    REF_DATE,
    col,
    color_of,
    dates,
    label_end,
    name_of,
    pos,
    style,
)
from .plots_labor import labor_market

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from matplotlib.axes import Axes

    from .calibration import Group, Params
    from .plots_common import Series

mpl.use("Agg")
import matplotlib.pyplot as plt

MONTHS = 12

# (suffix of the figure file, scenario names, title)
FAMILIES: list[tuple[str, list[str], str]] = [
    ("", list(AI_SCENARIOS), "AI scenarios"),
    ("_robots", list(ROBOT_SCENARIOS), "Robots-only scenarios"),
    ("_combined", list(DIAGONAL_SCENARIOS), "AI and robots combined, matched rungs"),
]


def _pct_gap(df: pd.DataFrame, name: str) -> Series:
    return 100 * (np.exp(col(df, name)) - 1)


def _growth(df: pd.DataFrame, name: str, base: float) -> Series:
    x = col(df, name)
    lag = np.full_like(x, np.nan)
    lag[MONTHS:] = x[:-MONTHS]
    return 100 * (base + x - lag)


def _emp_change(df: pd.DataFrame, column: str) -> Series:
    emp = col(df, column)
    return 100 * (emp / emp[pos(df, REF_DATE)] - 1)


def _u_rate(df: pd.DataFrame, g: str) -> Series:
    return 100 * col(df, f"U{g}") / (col(df, f"l{g}") + col(df, f"U{g}"))


_SERIES: dict[str, Callable[[pd.DataFrame, Params], Series]] = {
    "gdp_gap": lambda df, _p: _pct_gap(df, "dlnY"),
    "gdp_growth": lambda df, p: _growth(df, "dlnY", p.g + p.n),
    "avg_wage": lambda df, _p: _pct_gap(df, "dlnw_avg"),
    "cog_wage": lambda df, _p: _pct_gap(df, "dlnw_C"),
    "other_wage": lambda df, _p: _pct_gap(df, "dlnw_N"),
    "net_return": lambda df, p: 100 * (p.r_bar * np.exp(col(df, "x_r")) - p.delta),
    "labor_share": lambda df, _p: 100 * col(df, "sL"),
    "cog_emp": lambda df, _p: _emp_change(df, "lC"),
    "other_emp": lambda df, _p: _emp_change(df, "lN"),
    "u_cog": lambda df, _p: _u_rate(df, "C"),
    "u_other": lambda df, _p: _u_rate(df, "N"),
    "u_all": lambda df, _p: 100 * (col(df, "UC") + col(df, "UN")),
    "tfp": lambda df, _p: _pct_gap(df, "tfp"),
    "capital": lambda df, _p: _pct_gap(df, "dlnK"),
    "labor_income": lambda df, _p: _pct_gap(df, "dln_labor_income"),
    "capital_income": lambda df, _p: (
        100 * (np.exp(col(df, "x_r") + col(df, "dlnK")) - 1)
    ),
    "ideas": lambda df, _p: _pct_gap(df, "dlnA"),
}


def _series(df: pd.DataFrame, key: str) -> Series:
    return _SERIES[key](df, df.attrs["params"])


# (key, title, subtitle, no-AI baseline: a number, a key for the unemployment
# rates whose baseline comes from the steady state, or None)
PANELS: list[tuple[str, str, str, float | str | None]] = [
    ("gdp_gap", "GDP", "percent above the no-AI path", 0.0),
    ("gdp_growth", "GDP growth", "percent per year; 2 percent a year without AI", 2.0),
    ("tfp", "Measured TFP", "percent above the no-AI path", 0.0),
    ("avg_wage", "Average wage", "percent above the no-AI path", 0.0),
    ("cog_wage", "Cognitive wage", "percent above the no-AI path", 0.0),
    ("other_wage", "All-other wage", "percent above the no-AI path", 0.0),
    ("net_return", "Net return to capital", "percent per year; 6.5 without AI", 6.5),
    ("capital", "Capital stock", "percent above the no-AI path", 0.0),
    ("labor_share", "Labor share", "percent of income; 60 without AI", 60.0),
    ("labor_income", "Labor income", "percent above the no-AI path", 0.0),
    ("capital_income", "Capital income", "percent above the no-AI path", 0.0),
    ("ideas", "Ideas stock", "percent above the no-AI path", 0.0),
    ("cog_emp", "Cognitive employment", "percent change since mid-2026", 0.0),
    ("other_emp", "All-other employment", "percent change since mid-2026", 0.0),
    ("u_all", "Unemployment rate, all workers", "percent of the labor force", "u_all"),
    (
        "u_cog",
        "Unemployment rate, cognitive",
        "percent of the group's labor force",
        "u_cog",
    ),
    (
        "u_other",
        "Unemployment rate, all other",
        "percent of the group's labor force",
        "u_other",
    ),
]


def _baseline(dfs: dict[str, pd.DataFrame], spec: float | str | None) -> float | None:
    if isinstance(spec, str):
        df0 = next(iter(dfs.values()))
        if spec == "u_all":
            return 100 * (col(df0, "UC")[0] + col(df0, "UN")[0])
        return float(_u_rate(df0, "C" if spec == "u_cog" else "N")[0])
    return spec


def _plot_window(
    ax: Axes, df: pd.DataFrame, key: str, window: tuple[float, float]
) -> None:
    """Plot one series over the window in the scenario's colour, labelled at its end."""
    x = dates(df)
    s = _series(df, key)
    mask = (x >= window[0]) & (x <= window[1])
    ax.plot(x[mask], s[mask], color=color_of(df), lw=1.8, label=name_of(df))
    label_end(ax, x[mask], s[mask], color_of(df))


def scenario_panels(
    dfs: dict[str, pd.DataFrame],
    path: Path,
    window: tuple[float, float],
    title: str,
) -> None:
    ncol = 3
    nrow = math.ceil(len(PANELS) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow))
    flat = list(axes.flat)
    for ax, (key, ptitle, subtitle, base) in zip(flat, PANELS, strict=False):
        for df in dfs.values():
            _plot_window(ax, df, key, window)
        b = _baseline(dfs, base)
        if b is not None:
            ax.axhline(b, color="grey", ls="--", lw=0.8)
        style(ax, ptitle, subtitle, mark_2030=window[1] > PAPER_END)
    for ax in flat[len(PANELS) :]:
        ax.axis("off")
    flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _shock_row(
    axrow: list[Axes],
    g: Group,
    dfs: dict[str, pd.DataFrame],
    window: tuple[float, float],
) -> None:
    """One group's four exogenous objects, one line per single-shock scenario."""
    p: Params = next(iter(dfs.values())).attrs["params"]
    ceiling = p.m_bar(g)
    label = "AI on cognitive work" if g == "C" else "robots on all-other work"
    spec: list[tuple[str, str, str, float | None]] = [
        (
            f"m_{g}",
            f"Affected mass, {label}",
            f"share of the economy's tasks; ceiling {ceiling:.3f}",
            ceiling,
        ),
        (
            f"d_{g}",
            "Diffusion share",
            "share of affected instances performed by the technology",
            1.0,
        ),
        (
            f"md_{g}",
            "Performed share",
            "share of all task instances performed by the technology",
            None,
        ),
        (
            f"a_{g}",
            "Gain per instance",
            "log productivity gain per performed instance",
            None,
        ),
    ]
    for ax, (column, title, subtitle, ceil) in zip(axrow, spec, strict=True):
        for df in dfs.values():
            x, s = dates(df), col(df, column)
            mask = (x >= window[0]) & (x <= window[1])
            ax.plot(x[mask], s[mask], color=color_of(df), lw=1.8, label=name_of(df))
            label_end(ax, x[mask], s[mask], color_of(df), "{:.2f}")
        if ceil is not None:
            ax.axhline(ceil, color="grey", ls="--", lw=0.8)
        style(ax, title, subtitle, mark_2030=True)
    axrow[0].legend(frameon=False, fontsize=8)


def shock_paths(
    dfs: dict[str, pd.DataFrame], path: Path, window: tuple[float, float]
) -> None:
    """The exogenous objects of each single-shock scenario: AI on the cognitive
    group in the top row, robots on the all-other group in the bottom row."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 7.6), squeeze=False)
    families: list[tuple[Group, list[str]]] = [
        ("C", list(AI_SCENARIOS)),
        ("N", list(ROBOT_SCENARIOS)),
    ]
    for (g, names), axrow in zip(families, axes, strict=True):
        family = {k: dfs[k] for k in names if k in dfs}
        if family:
            _shock_row(list(axrow), g, family, window)
    fig.suptitle(
        "Exogenous scenario paths: logistics through the mid-2026 anchor and the scenario "
        "date (2030 for AI, 2035 for robots); beyond it the paths follow equation 8 unchanged",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(
    dfs: dict[str, pd.DataFrame], out: Path, start: float = 2025.0
) -> list[Path]:
    out = Path(out)
    end = float(dates(next(iter(dfs.values()))).max())
    full, paper = (start, end), (start, PAPER_END)
    written: list[Path] = []

    def save(filename: str, job: Callable[[Path], None]) -> None:
        f = out / filename
        job(f)
        written.append(f)

    paper_dfs = {k: v for k, v in dfs.items() if k in PAPER_SCENARIOS}
    if paper_dfs:
        save(
            "scenarios_2025_2030.png",
            lambda f: scenario_panels(
                paper_dfs,
                f,
                paper,
                "Replication of the paper's Figures 2-4: 2025-2030; labels are January 2030 values",
            ),
        )
    for suffix, names, title in FAMILIES:
        family = {k: dfs[k] for k in names if k in dfs}
        if not family:
            continue
        save(
            f"scenarios{suffix}.png",
            lambda f, fam=family, ttl=title: scenario_panels(
                fam,
                f,
                full,
                f"{ttl}, 2025-{int(end)} (vertical line: start of 2030, the paper's horizon)",
            ),
        )
        save(
            f"labor_market{suffix}.png",
            lambda f, fam=family: labor_market(fam, f, full),
        )
    save("shock_paths.png", lambda f: shock_paths(dfs, f, full))
    return written
