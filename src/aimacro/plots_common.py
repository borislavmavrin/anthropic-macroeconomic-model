"""Helpers shared by the figure modules: column access, axis styling, colours."""

# matplotlib's type stubs leave most keyword arguments untyped, which pyright in
# strict mode reports on every call; the rule is disabled for the figure modules.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    Series = NDArray[np.float64]

COLORS = {  # by rung of the scenario ladder
    "modest": "#1f4e9c",
    "substantial": "#2e6b2e",
    "extreme": "#b5452b",
    "agi": "#5b2a86",
}
PAPER_END = 2030.0
REF_DATE = 2026.5  # employment changes are measured since mid-2026


def col(df: pd.DataFrame, name: str) -> Series:
    return df[name].to_numpy(dtype=float)


def dates(df: pd.DataFrame) -> Series:
    return df.index.to_numpy(dtype=float)


def pos(df: pd.DataFrame, t: float) -> int:
    return int(np.argmin(np.abs(dates(df) - t)))


def color_of(df: pd.DataFrame) -> str:
    return COLORS[str(df.attrs["scenario"].rung)]


def name_of(df: pd.DataFrame) -> str:
    return str(df.attrs["scenario"].name)


def style(ax: Axes, title: str, subtitle: str, *, mark_2030: bool) -> None:
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


def label_end(ax: Axes, x: Series, s: Series, color: str, fmt: str = "{:.1f}") -> None:
    """Annotate the last finite point of a series with its value."""
    finite = np.flatnonzero(np.isfinite(s))
    last = float(s[finite[-1]])
    ax.annotate(
        fmt.format(last),
        (float(x[finite[-1]]), last),
        xytext=(3, 0),
        textcoords="offset points",
        color=color,
        fontsize=8,
        va="center",
    )
