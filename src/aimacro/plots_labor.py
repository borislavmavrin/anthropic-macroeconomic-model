"""The labor-market figure: one row per scenario with employment against the
full-mobility targets, the unemployment pools and job-finding rates by origin,
and the monthly flows of layoffs, hires and quits."""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from .labor import steady_state
from .plots_common import PAPER_END, col, color_of, dates, name_of, style

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from matplotlib.axes import Axes


def _row(axes: list[Axes], df: pd.DataFrame, window: tuple[float, float]) -> None:
    name = name_of(df)
    ss = steady_state(df.attrs["params"])
    c, grey = color_of(df), "dimgray"
    x = dates(df)
    mask = (x >= window[0]) & (x <= window[1])
    far = window[1] > PAPER_END

    def line(ax: Axes, column: str, color: str, label: str, ls: str = "-") -> None:
        y = 100 * col(df, column)[mask]
        ax.plot(x[mask], y, color=color, lw=1.8, ls=ls, label=label)

    ax = axes[0]
    line(ax, "lC", c, "cognitive, actual")
    line(ax, "lC_star", c, "cognitive, target", "--")
    line(ax, "lN", grey, "all other, actual")
    line(ax, "lN_star", grey, "all other, target", "--")
    style(
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
    style(
        ax,
        f"{name}: unemployed by origin",
        "percent of the labor force; dashed: normal",
        mark_2030=far,
    )
    ax = axes[2]
    line(ax, "fC", c, "cognitive origin")
    line(ax, "fN", grey, "all-other origin")
    ax.axhline(100 * ss.fC, color=c, ls="--", lw=0.8)
    ax.axhline(100 * ss.fN, color=grey, ls="--", lw=0.8)
    style(
        ax,
        f"{name}: job-finding rate",
        "percent per month; dashed: normal times",
        mark_2030=far,
    )
    ax = axes[3]
    line(ax, "D_C", c, "layoffs, cognitive")
    line(ax, "D_N", c, "layoffs, all other", "--")
    hires = 100 * (col(df, "H_C") + col(df, "H_N"))[mask]
    quits = 100 * (col(df, "qC") * col(df, "lC") + col(df, "qN") * col(df, "lN"))[mask]
    ax.plot(x[mask], hires, color=grey, lw=1.8, label="hires, both groups")
    ax.plot(x[mask], quits, color=grey, lw=1.2, ls=":", label="quits, both groups")
    style(
        ax,
        f"{name}: monthly flows",
        "percent of the labor force per month",
        mark_2030=far,
    )
    for a in axes:
        a.legend(frameon=False, fontsize=7)


def labor_market(
    dfs: dict[str, pd.DataFrame], path: Path, window: tuple[float, float]
) -> None:
    fig, axes = plt.subplots(len(dfs), 4, figsize=(21, 3.6 * len(dfs)), squeeze=False)
    for axrow, df in zip(axes, dfs.values(), strict=True):
        _row(list(axrow), df, window)
    fig.suptitle("The labor market block (Section 2.3)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)
