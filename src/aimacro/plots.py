"""Figures: the paper's Figures 2-4 (2025-2030), their extension to the simulated
horizon, the exogenous AI paths, and the labor-market flows behind unemployment."""

# matplotlib's type stubs leave most keyword arguments untyped, which pyright in
# strict mode reports on every call; the rule is disabled for this module only.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

from .calibration import PAPER_SCENARIOS
from .labor import steady_state

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from .calibration import Params

    Series = NDArray[np.float64]

mpl.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "modest": "#1f4e9c",
    "substantial": "#2e6b2e",
    "extreme": "#b5452b",
    "agi": "#5b2a86",
}
PAPER_END = 2030.0
REF_DATE = 2026.5  # employment changes are measured since mid-2026
MONTHS = 12


def _col(df: pd.DataFrame, name: str) -> Series:
    return df[name].to_numpy(dtype=float)


def _dates(df: pd.DataFrame) -> Series:
    return df.index.to_numpy(dtype=float)


def _pos(df: pd.DataFrame, t: float) -> int:
    return int(np.argmin(np.abs(_dates(df) - t)))


def _pct_gap(df: pd.DataFrame, name: str) -> Series:
    return 100 * (np.exp(_col(df, name)) - 1)


def _growth(df: pd.DataFrame, name: str, base: float) -> Series:
    x = _col(df, name)
    lag = np.full_like(x, np.nan)
    lag[MONTHS:] = x[:-MONTHS]
    return 100 * (base + x - lag)


def _cog_emp(df: pd.DataFrame) -> Series:
    lC = _col(df, "lC")
    return 100 * (lC / lC[_pos(df, REF_DATE)] - 1)


_SERIES: dict[str, Callable[[pd.DataFrame, Params], Series]] = {
    "gdp_gap": lambda df, _p: _pct_gap(df, "dlnY"),
    "gdp_growth": lambda df, p: _growth(df, "dlnY", p.g + p.n),
    "avg_wage": lambda df, _p: _pct_gap(df, "dlnw_avg"),
    "cog_wage": lambda df, _p: _pct_gap(df, "dlnw_C"),
    "other_wage": lambda df, _p: _pct_gap(df, "dlnw_N"),
    "net_return": lambda df, p: 100 * (p.r_bar * np.exp(_col(df, "x_r")) - p.delta),
    "labor_share": lambda df, _p: 100 * _col(df, "sL"),
    "cog_emp": lambda df, _p: _cog_emp(df),
    "u_cog": lambda df, _p: 100 * _col(df, "UC") / (_col(df, "lC") + _col(df, "UC")),
    "u_all": lambda df, _p: 100 * (_col(df, "UC") + _col(df, "UN")),
    "tfp": lambda df, _p: _pct_gap(df, "tfp"),
    "capital": lambda df, _p: _pct_gap(df, "dlnK"),
    "labor_income": lambda df, _p: _pct_gap(df, "dln_labor_income"),
    "capital_income": lambda df, _p: (
        100 * (np.exp(_col(df, "x_r") + _col(df, "dlnK")) - 1)
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
    (
        "u_cog",
        "Unemployment rate, cognitive",
        "percent of the group's labor force",
        "u_cog",
    ),
    ("u_all", "Unemployment rate, all workers", "percent of the labor force", "u_all"),
]


def _style(ax: Axes, title: str, subtitle: str, *, mark_2030: bool) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.text(
        0.0,
        1.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        color="dimgray",
        va="bottom",
    )
    if mark_2030:
        ax.axvline(PAPER_END, color="lightgrey", lw=0.8)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(b=False)


def _baseline(dfs: dict[str, pd.DataFrame], spec: float | str | None) -> float | None:
    if isinstance(spec, str):
        df0 = next(iter(dfs.values()))
        uC, uN, lC = _col(df0, "UC")[0], _col(df0, "UN")[0], _col(df0, "lC")[0]
        return 100 * uC / (lC + uC) if spec == "u_cog" else 100 * (uC + uN)
    return spec


def _plot_window(
    ax: Axes,
    df: pd.DataFrame,
    key: str,
    window: tuple[float, float],
) -> None:
    """Plot one series over [start, end] in the scenario's colour, labelled at its end."""
    name = str(df.attrs["scenario"].name)
    x = _dates(df)
    s = _series(df, key)
    mask = (x >= window[0]) & (x <= window[1])
    x, s = x[mask], s[mask]
    ax.plot(x, s, color=COLORS[name], lw=1.8, label=name)
    finite = np.flatnonzero(np.isfinite(s))
    last = float(s[finite[-1]])
    ax.annotate(
        f"{last:.1f}",
        (float(x[finite[-1]]), last),
        xytext=(3, 0),
        textcoords="offset points",
        color=COLORS[name],
        fontsize=8,
        va="center",
    )


def scenario_panels(
    dfs: dict[str, pd.DataFrame],
    path: Path,
    window: tuple[float, float],
    title: str,
) -> None:
    ncol = 3
    nrow = math.ceil(len(PANELS) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow))
    for ax, (key, ptitle, subtitle, base) in zip(axes.flat, PANELS, strict=False):
        for df in dfs.values():
            _plot_window(ax, df, key, window)
        b = _baseline(dfs, base)
        if b is not None:
            ax.axhline(b, color="grey", ls="--", lw=0.8)
        _style(ax, ptitle, subtitle, mark_2030=window[1] > PAPER_END)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ai_paths(
    dfs: dict[str, pd.DataFrame],
    path: Path,
    window: tuple[float, float],
) -> None:
    p: Params = next(iter(dfs.values())).attrs["params"]
    spec: list[tuple[str, str, str, float | None]] = [
        (
            "m",
            "Affected mass m_t",
            "share of the economy's tasks; ceiling 0.624",
            p.m_bar,
        ),
        (
            "d",
            "Diffusion share d_t",
            "share of affected instances done with AI; ceiling 1",
            1.0,
        ),
        (
            "md",
            "AI-performed share m_t d_t",
            "share of all task instances performed with AI",
            None,
        ),
        (
            "a",
            "Gain per instance a_t",
            "log productivity gain per AI-performed instance",
            None,
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
    for ax, (col, title, subtitle, ceiling) in zip(axes, spec, strict=True):
        for name, df in dfs.items():
            x, s = _dates(df), _col(df, col)
            mask = (x >= window[0]) & (x <= window[1])
            ax.plot(x[mask], s[mask], color=COLORS[name], lw=1.8, label=name)
            ax.annotate(
                f"{s[mask][-1]:.2f}",
                (x[mask][-1], s[mask][-1]),
                xytext=(3, 0),
                textcoords="offset points",
                color=COLORS[name],
                fontsize=8,
                va="center",
            )
        if ceiling is not None:
            ax.axhline(ceiling, color="grey", ls="--", lw=0.8)
        ax.axvline(p.t_anchor, color="lightgrey", lw=0.8, ls=":")
        _style(ax, title, subtitle, mark_2030=True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Exogenous scenario paths (dotted line: mid-2026 anchor; solid grey: 2030, where "
        "the scenarios' values are set; beyond 2030 the paths follow equation 8 unchanged)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _labor_row(axes: list[Axes], df: pd.DataFrame, window: tuple[float, float]) -> None:
    name = str(df.attrs["scenario"].name)
    ss = steady_state(df.attrs["params"])
    c, grey = COLORS[name], "dimgray"
    x = _dates(df)
    mask = (x >= window[0]) & (x <= window[1])
    far = window[1] > PAPER_END

    def line(ax: Axes, col: str, color: str, label: str, ls: str = "-") -> None:
        ax.plot(
            x[mask],
            100 * _col(df, col)[mask],
            color=color,
            lw=1.8,
            ls=ls,
            label=label,
        )

    ax = axes[0]
    line(ax, "lC", c, "cognitive, actual")
    line(ax, "lC_star", c, "cognitive, target", "--")
    line(ax, "lN", grey, "all other, actual")
    line(ax, "lN_star", grey, "all other, target", "--")
    _style(
        ax,
        f"{name}: employment",
        "percent of the labor force; dashed: target",
        mark_2030=far,
    )
    ax = axes[1]
    line(ax, "UC", c, "cognitive origin")
    line(ax, "UN", grey, "all-other origin")
    ax.axhline(100 * ss.UC, color=c, ls="--", lw=0.8)
    ax.axhline(100 * ss.UN, color=grey, ls="--", lw=0.8)
    _style(
        ax,
        f"{name}: unemployed by origin",
        "percent of the labor force; dashed: normal pool",
        mark_2030=far,
    )
    ax = axes[2]
    line(ax, "fC", c, "cognitive origin")
    line(ax, "fN", grey, "all-other origin")
    ax.axhline(100 * ss.fC, color=c, ls="--", lw=0.8)
    ax.axhline(100 * ss.fN, color=grey, ls="--", lw=0.8)
    _style(
        ax,
        f"{name}: job-finding rate",
        "percent per month; dashed: normal times",
        mark_2030=far,
    )
    ax = axes[3]
    line(ax, "D_C", c, "layoffs, cognitive")
    hires = 100 * (_col(df, "H_C") + _col(df, "H_N"))[mask]
    quits = (
        100 * (_col(df, "qC") * _col(df, "lC") + _col(df, "qN") * _col(df, "lN"))[mask]
    )
    ax.plot(x[mask], hires, color=grey, lw=1.8, label="hires, both groups")
    ax.plot(x[mask], quits, color=grey, lw=1.2, ls=":", label="quits, both groups")
    _style(
        ax,
        f"{name}: monthly flows",
        "percent of the labor force per month",
        mark_2030=far,
    )
    for a in axes:
        a.legend(frameon=False, fontsize=7)


def labor_market(
    dfs: dict[str, pd.DataFrame],
    path: Path,
    window: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(len(dfs), 4, figsize=(21, 3.6 * len(dfs)), squeeze=False)
    for row, df in zip(axes, dfs.values(), strict=True):
        _labor_row(list(row), df, window)
    fig.suptitle("The labor market block (Section 2.3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(
    dfs: dict[str, pd.DataFrame],
    out: Path,
    start: float = 2025.0,
) -> list[Path]:
    out = Path(out)
    end = float(_dates(next(iter(dfs.values()))).max())
    paper_dfs = {k: v for k, v in dfs.items() if k in PAPER_SCENARIOS} or dfs
    full, paper = (start, end), (start, PAPER_END)
    jobs: list[tuple[str, Callable[[Path], None]]] = [
        (
            "scenarios.png",
            lambda f: scenario_panels(
                dfs,
                f,
                full,
                f"Economic scenarios for transformative AI, 2025-{int(end)} "
                "(vertical line: start of 2030, the paper's horizon)",
            ),
        ),
        (
            "scenarios_2025_2030.png",
            lambda f: scenario_panels(
                paper_dfs,
                f,
                paper,
                "Replication of the paper's Figures 2-4: 2025-2030; "
                "labels are January 2030 values",
            ),
        ),
        ("ai_paths.png", lambda f: ai_paths(dfs, f, full)),
        ("labor_market.png", lambda f: labor_market(dfs, f, full)),
    ]
    written: list[Path] = []
    for filename, job in jobs:
        f = out / filename
        job(f)
        written.append(f)
    return written
