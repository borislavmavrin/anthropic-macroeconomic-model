"""Acceptance tests: the predictions reported in Korinek, Jones, Sacher, Cotter and
McCrory (2026), "Economic Scenarios for Transformative AI", Anthropic Institute
WP 2026-02 (doc/econ_scenarios_transformative_ai.pdf).

These are the ONLY tests of the implementation.  Each test checks one number
the paper prints:

  * Table 3  (p. 31): the three scenarios at the start of 2030.
  * Table 5  (p. 37): substantial and extreme scenarios at eps = 1, 3, 6, inf.
  * Table 6  (p. 38): substantial and extreme scenarios at xi = 0, 0.5, 0.75, 0.9.
  * Section 3.2 (p. 21): the normal-times steady state of the labor market.

Table 4 (outcomes implied by the survey) needs the respondent-level survey
microdata, which the paper does not publish, so it cannot be reproduced.

Tolerance.  The paper prints most values to one decimal.  A value passes if it
is within max(one unit of the last printed digit, 1% of the printed value) of
the paper.  The 1% term allows for the paper's own rounding of inputs such as
"-21.5 percent since mid-2026" when it reports the largest magnitudes.
"""

from __future__ import annotations

import math

import pytest

from aimacro.calibration import DEFAULT_PARAMS
from aimacro.labor import steady_state
from aimacro.report import snapshot
from aimacro.scenarios import SCENARIOS
from aimacro.simulate import simulate

T2030 = 2030.0


def _tol(printed: str) -> float:
    value = float(printed)
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    return max(10.0 ** (-decimals), 0.01 * abs(value))


def _check(actual: float, printed: str, label: str) -> None:
    expected = float(printed)
    tol = _tol(printed)
    assert math.isfinite(actual), f"{label}: got {actual!r}"
    assert abs(actual - expected) <= tol, (
        f"{label}: model {actual:.4f} vs paper {expected} (tolerance {tol:.3f})"
    )


# --------------------------------------------------------------------------
# Simulation cache: one monthly simulation per (scenario, eps, xi) to 2030.
# --------------------------------------------------------------------------

_CACHE: dict[tuple[str, float, float], dict[str, float]] = {}


def snap(name: str, *, eps: float = 3.0, xi: float = 0.5) -> dict[str, float]:
    key = (name, eps, xi)
    if key not in _CACHE:
        sc = SCENARIOS[name].with_(eps=eps, xi=xi)
        df = simulate(sc, DEFAULT_PARAMS, t_end=T2030)
        _CACHE[key] = snapshot(df, T2030)
    return _CACHE[key]


# --------------------------------------------------------------------------
# Table 3: The three scenarios in 2030.  Columns: modest, substantial, extreme.
# --------------------------------------------------------------------------

TABLE3 = {
    # A. Output
    "gdp_gap_pct": ("1.6", "8.3", "32.4"),
    "gdp_index": ("114.5", "122.1", "149.3"),
    "gdp_growth_pct": ("2.4", "5.4", "15.4"),
    # B. Factor prices and income shares
    "avg_wage_gap_pct": ("0.7", "2.1", "9.7"),
    "cog_wage_gap_pct": ("0.4", "-0.3", "-11.5"),
    "other_wage_gap_pct": ("1.1", "5.9", "33.6"),
    "net_return_pct": ("6.6", "7.0", "8.3"),
    "capital_gap_pct": ("2.3", "13.8", "56.3"),
    "labor_share_pct": ("59.4", "56.1", "45.2"),
    "capital_share_pct": ("40.6", "43.9", "54.8"),
    "labor_income_gap_pct": ("0.6", "1.4", "0.5"),
    "cog_wage_bill_gap_pct": ("-0.3", "-4.6", "-31.0"),
    "capital_income_gap_pct": ("3.1", "18.9", "81.4"),
    # C. The labor market
    "cog_employment_change_pct": ("-0.5", "-3.9", "-21.5"),
    "u_cog_pct": ("2.9", "4.5", "17.9"),
    "u_all_pct": ("3.9", "4.6", "11.9"),
    # D. Technology: the two channels
    "tfp_gap_pct": ("0.7", "3.1", "13.4"),
    "tfp_growth_pct": ("1.2", "2.3", "7.3"),
    "ideas_gap_pct": ("0.07", "0.20", "0.61"),
    "ideas_growth_pct": ("1.69", "1.76", "2.02"),
}
TABLE3_SCENARIOS = ("modest", "substantial", "extreme")


@pytest.mark.parametrize("row", list(TABLE3))
@pytest.mark.parametrize("scenario", TABLE3_SCENARIOS)
def test_table3(scenario: str, row: str) -> None:
    printed = TABLE3[row][TABLE3_SCENARIOS.index(scenario)]
    _check(snap(scenario)[row], printed, f"Table 3 / {scenario} / {row}")


def test_table3_no_ai_column() -> None:
    """The No-AI column of Table 3 is the balanced growth path."""
    p = DEFAULT_PARAMS
    _check(
        100 * math.exp((p.g + p.n) * (T2030 - p.t0)),
        "112.7",
        "Table 3 / No AI / gdp_index",
    )
    _check(100 * (p.g + p.n), "2.0", "Table 3 / No AI / gdp_growth_pct")
    _check(100 * (p.r_bar - p.delta), "6.5", "Table 3 / No AI / net_return_pct")
    _check(100 * p.sL0, "60.0", "Table 3 / No AI / labor_share_pct")
    _check(100 * p.gA, "1.0", "Table 3 / No AI / tfp_growth_pct")
    _check(100 * p.g, "1.67", "Table 3 / No AI / ideas_growth_pct")


# --------------------------------------------------------------------------
# Table 5: Scenarios in 2030 at four elasticities of capital supply.
# --------------------------------------------------------------------------

TABLE5_EPS = (1.0, 3.0, 6.0, math.inf)
TABLE5 = {
    "substantial": {
        "gdp_gap_pct": ("6.4", "8.3", "9.1", "10.0"),
        "avg_wage_gap_pct": ("-1.6", "2.1", "3.7", "5.6"),
        "net_return_pct": ("7.6", "7.0", "6.8", "6.5"),
        "capital_gap_pct": ("9.3", "13.8", "15.7", "18.2"),
        "labor_share_pct": ("55.1", "56.1", "56.5", "57.0"),
    },
    "extreme": {
        "gdp_gap_pct": ("21.3", "32.4", "37.2", "43.3"),
        "avg_wage_gap_pct": ("-9.2", "9.7", "18.3", "30.1"),
        "net_return_pct": ("10.3", "8.3", "7.5", "6.5"),
        "capital_gap_pct": ("33.5", "56.3", "67.1", "82.2"),
        "labor_share_pct": ("41.2", "45.2", "46.9", "49.1"),
    },
}


@pytest.mark.parametrize("row", list(TABLE5["substantial"]))
@pytest.mark.parametrize("eps", TABLE5_EPS, ids=["eps1", "eps3", "eps6", "epsinf"])
@pytest.mark.parametrize("scenario", list(TABLE5))
def test_table5(scenario: str, eps: float, row: str) -> None:
    printed = TABLE5[scenario][row][TABLE5_EPS.index(eps)]
    _check(
        snap(scenario, eps=eps)[row],
        printed,
        f"Table 5 / {scenario} / eps={eps} / {row}",
    )


# --------------------------------------------------------------------------
# Table 6: Scenarios in 2030 at four rigidities of the cognitive wage.
# --------------------------------------------------------------------------

TABLE6_COLUMNS = (
    ("substantial", 0.5),
    ("substantial", 0.75),
    ("substantial", 0.9),
    ("extreme", 0.0),
    ("extreme", 0.5),
    ("extreme", 0.75),
    ("extreme", 0.9),
)
TABLE6 = {
    "gdp_gap_pct": ("8.3", "7.9", "7.7", "36.6", "32.4", "30.5", "29.2"),
    "avg_wage_gap_pct": ("2.1", "2.2", "2.3", "1.6", "9.7", "11.1", "11.9"),
    "cog_wage_gap_pct": ("-0.3", "0.7", "1.4", "-42.2", "-11.5", "-2.9", "2.8"),
    "other_wage_gap_pct": ("5.9", "4.5", "3.7", "70.1", "33.6", "25.8", "21.1"),
    "cog_employment_change_pct": (
        "-3.9",
        "-4.6",
        "-5.0",
        "-1.3",
        "-21.5",
        "-25.9",
        "-28.5",
    ),
    "u_cog_pct": ("4.5", "5.1", "5.4", "2.6", "17.9", "21.7", "24.0"),
    "u_all_pct": ("4.6", "4.9", "5.2", "3.1", "11.9", "13.9", "15.2"),
}


@pytest.mark.parametrize("row", list(TABLE6))
@pytest.mark.parametrize(
    "column",
    TABLE6_COLUMNS,
    ids=[f"{s}-xi{x}" for s, x in TABLE6_COLUMNS],
)
def test_table6(column: tuple[str, float], row: str) -> None:
    scenario, xi = column
    printed = TABLE6[row][TABLE6_COLUMNS.index(column)]
    _check(
        snap(scenario, xi=xi)[row],
        printed,
        f"Table 6 / {scenario} / xi={xi} / {row}",
    )


# --------------------------------------------------------------------------
# Section 3.2 / Section 2.3.2: the normal-times steady state (p. 21).
# --------------------------------------------------------------------------


def test_steady_state_pool_split_and_filling_rates() -> None:
    ss = steady_state(DEFAULT_PARAMS)
    _check(100 * ss.UC, "1.76", "steady state / U_C, pct of labor force")
    _check(100 * ss.UN, "2.08", "steady state / U_N, pct of labor force")
    _check(
        100 * ss.UC / (DEFAULT_PARAMS.lC0 + ss.UC),
        "2.9",
        "steady state / cognitive unemployment rate",
    )
    _check(
        100 * ss.UN / (DEFAULT_PARAMS.lN0 + ss.UN),
        "5.4",
        "steady state / all-other unemployment rate",
    )
    _check(ss.piC, "0.66", "steady state / filling rate, cognitive")
    _check(ss.piN, "0.64", "steady state / filling rate, all other")
    _check(ss.chi, "0.76", "steady state / matching efficiency chi")
    f_agg = (ss.fC * ss.UC + ss.fN * ss.UN) / (ss.UC + ss.UN)
    _check(f_agg, "0.23", "steady state / aggregate job-finding rate per month")
