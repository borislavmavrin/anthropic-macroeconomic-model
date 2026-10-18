"""Figures: the paper's Figures 2-4 (2025-2030) and their extension to the simulated
horizon, the exogenous AI paths, and the labor-market flows behind the unemployment paths."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .labor import steady_state  # noqa: E402

COLORS = {"modest": "#1f4e9c", "substantial": "#2e6b2e", "extreme": "#b5452b"}
PAPER_END = 2030.0


def _series(df: pd.DataFrame, key: str) -> pd.Series:
    p = df.attrs["params"]
    ex = np.exp
    if key == "gdp_gap":
        return 100 * (ex(df.dlnY) - 1)
    if key == "gdp_growth":
        return 100 * ((p.g + p.n) + df.dlnY - df.dlnY.shift(12))
    if key == "avg_wage":
        return 100 * (ex(df.dlnw_avg) - 1)
    if key == "cog_wage":
        return 100 * (ex(df.dlnw_C) - 1)
    if key == "other_wage":
        return 100 * (ex(df.dlnw_N) - 1)
    if key == "net_return":
        return 100 * (p.r_bar * ex(df.x_r) - p.delta)
    if key == "labor_share":
        return 100 * df.sL
    if key == "cog_emp":
        return 100 * (df.lC / df.loc[2026.5, "lC"] - 1)
    if key == "u_cog":
        return 100 * df.UC / (df.lC + df.UC)
    if key == "u_all":
        return 100 * (df.UC + df.UN)
    if key == "tfp":
        return 100 * (ex(df.tfp) - 1)
    if key == "capital":
        return 100 * (ex(df.dlnK) - 1)
    if key == "labor_income":
        return 100 * (ex(df.dln_labor_income) - 1)
    if key == "capital_income":
        return 100 * (ex(df.x_r + df.dlnK) - 1)
    if key == "ideas":
        return 100 * (ex(df.dlnA) - 1)
    raise KeyError(key)


# (key, title, subtitle, no-AI baseline or None)
PANELS = [
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
    ("u_cog", "Unemployment rate, cognitive", "percent of the group's labor force", "u_cog"),
    ("u_all", "Unemployment rate, all workers", "percent of the labor force", "u_all"),
]


def _style(ax, title: str, subtitle: str, mark_2030: bool) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, fontsize=8, color="dimgray", va="bottom")
    if mark_2030:
        ax.axvline(PAPER_END, color="lightgrey", lw=0.8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _baseline(dfs: dict[str, pd.DataFrame], spec) -> float | None:
    if spec == "u_cog":
        df0 = next(iter(dfs.values()))
        return 100 * df0.UC.iloc[0] / (df0.lC.iloc[0] + df0.UC.iloc[0])
    if spec == "u_all":
        df0 = next(iter(dfs.values()))
        return 100 * (df0.UC.iloc[0] + df0.UN.iloc[0])
    return spec


def scenario_panels(dfs: dict[str, pd.DataFrame], path: Path, start: float, end: float, title: str) -> None:
    ncol = 3
    nrow = math.ceil(len(PANELS) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.5 * nrow))
    for ax, (key, ptitle, subtitle, base) in zip(axes.flat, PANELS):
        for name, df in dfs.items():
            s = _series(df, key)
            s = s[(s.index >= start) & (s.index <= end)]
            ax.plot(s.index, s.values, color=COLORS[name], lw=1.8, label=name)
            last = s.dropna().iloc[-1]
            ax.annotate(f"{last:.1f}", (s.index[-1], last), xytext=(3, 0),
                        textcoords="offset points", color=COLORS[name], fontsize=8, va="center")
        b = _baseline(dfs, base)
        if b is not None:
            ax.axhline(b, color="grey", ls="--", lw=0.8)
        _style(ax, ptitle, subtitle, mark_2030=end > PAPER_END)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def ai_paths(dfs: dict[str, pd.DataFrame], path: Path, start: float, end: float) -> None:
    p = next(iter(dfs.values())).attrs["params"]
    spec = [
        ("m", "Affected mass m_t", "share of the economy's tasks; ceiling 0.624", p.m_bar),
        ("d", "Diffusion share d_t", "share of affected instances performed with AI; ceiling 1", 1.0),
        ("md", "AI-performed share m_t d_t", "share of all task instances performed with AI", None),
        ("a", "Gain per instance a_t", "log productivity gain per AI-performed instance", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8))
    for ax, (col, title, subtitle, ceiling) in zip(axes, spec):
        for name, df in dfs.items():
            s = df[col]
            s = s[(s.index >= start) & (s.index <= end)]
            ax.plot(s.index, s.values, color=COLORS[name], lw=1.8, label=name)
            ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", color=COLORS[name], fontsize=8, va="center")
        if ceiling is not None:
            ax.axhline(ceiling, color="grey", ls="--", lw=0.8)
        ax.axvline(p.t_anchor, color="lightgrey", lw=0.8, ls=":")
        _style(ax, title, subtitle, mark_2030=True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Exogenous scenario paths (dotted line: mid-2026 anchor; solid grey: 2030, where the "
                 "scenarios' values are set; beyond 2030 the paths follow equation 8 unchanged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def labor_market(dfs: dict[str, pd.DataFrame], path: Path, start: float, end: float) -> None:
    p = next(iter(dfs.values())).attrs["params"]
    ss = steady_state(p)
    fig, axes = plt.subplots(len(dfs), 4, figsize=(21, 3.6 * len(dfs)), squeeze=False)
    for row, (name, df) in zip(axes, dfs.items()):
        d = df[(df.index >= start) & (df.index <= end)]
        c = COLORS[name]
        ax = row[0]
        ax.plot(d.index, 100 * d.lC, color=c, lw=1.8, label="cognitive, actual")
        ax.plot(d.index, 100 * d.lC_star, color=c, lw=1.2, ls="--", label="cognitive, target")
        ax.plot(d.index, 100 * d.lN, color="dimgray", lw=1.8, label="all other, actual")
        ax.plot(d.index, 100 * d.lN_star, color="dimgray", lw=1.2, ls="--", label="all other, target")
        _style(ax, f"{name}: employment", "percent of the labor force; dashed: full-mobility target", end > PAPER_END)
        ax.legend(frameon=False, fontsize=7)
        ax = row[1]
        ax.plot(d.index, 100 * d.UC, color=c, lw=1.8, label="cognitive origin")
        ax.plot(d.index, 100 * d.UN, color="dimgray", lw=1.8, label="all-other origin")
        ax.axhline(100 * ss.UC, color=c, ls="--", lw=0.8)
        ax.axhline(100 * ss.UN, color="dimgray", ls="--", lw=0.8)
        _style(ax, f"{name}: unemployed by origin", "percent of the labor force; dashed: normal pool", end > PAPER_END)
        ax.legend(frameon=False, fontsize=7)
        ax = row[2]
        ax.plot(d.index, 100 * d.fC, color=c, lw=1.8, label="cognitive origin")
        ax.plot(d.index, 100 * d.fN, color="dimgray", lw=1.8, label="all-other origin")
        ax.axhline(100 * ss.fC, color=c, ls="--", lw=0.8)
        ax.axhline(100 * ss.fN, color="dimgray", ls="--", lw=0.8)
        _style(ax, f"{name}: job-finding rate", "percent per month; dashed: normal times", end > PAPER_END)
        ax.legend(frameon=False, fontsize=7)
        ax = row[3]
        ax.plot(d.index, 100 * d.D_C, color=c, lw=1.8, label="layoffs, cognitive")
        ax.plot(d.index, 100 * (d.H_C + d.H_N), color="dimgray", lw=1.8, label="hires, both groups")
        ax.plot(d.index, 100 * (d.qC * d.lC + d.qN * d.lN), color="dimgray", lw=1.2, ls=":", label="quits, both groups")
        _style(ax, f"{name}: monthly flows", "percent of the labor force per month", end > PAPER_END)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("The labor market block (Section 2.3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(dfs: dict[str, pd.DataFrame], out: Path, start: float = 2025.0) -> list[Path]:
    out = Path(out)
    end = float(next(iter(dfs.values())).index.max())
    written = []
    f = out / "scenarios.png"
    scenario_panels(dfs, f, start, end,
                    f"Economic scenarios for transformative AI, 2025-{int(end)} "
                    "(vertical line: start of 2030, the paper's horizon)")
    written.append(f)
    f = out / "scenarios_2025_2030.png"
    scenario_panels(dfs, f, start, PAPER_END,
                    "Replication of the paper's Figures 2-4: 2025-2030; labels are January 2030 values")
    written.append(f)
    f = out / "ai_paths.png"
    ai_paths(dfs, f, start, end)
    written.append(f)
    f = out / "labor_market.png"
    labor_market(dfs, f, start, end)
    written.append(f)
    return written
