"""The scenario catalog, one module per scenario.

Each module defines one `SCENARIO`.  The AI modules shock the cognitive group, the
robot modules the all-other group.  Each robot module mirrors one of the paper's
AI scenarios on the all-other group.  Their combinations are derived here: an AI
scenario paired with a robot scenario keeps the AI scenario's capital-supply
elasticity and rung.
"""

from __future__ import annotations

from aimacro.calibration import Scenario

from . import (
    agi,
    extreme,
    modest,
    robots_extreme,
    robots_modest,
    robots_substantial,
    substantial,
)


def _catalog(*scenarios: Scenario) -> dict[str, Scenario]:
    return {sc.name: sc for sc in scenarios}


def combine(ai: Scenario, robots: Scenario) -> Scenario:
    return Scenario(
        f"{ai.name}+{robots.name}",
        ai=ai.ai,
        robots=robots.robots,
        eps=ai.eps,
        rung=ai.rung,
    )


AI_SCENARIOS: dict[str, Scenario] = _catalog(
    modest.SCENARIO,
    substantial.SCENARIO,
    extreme.SCENARIO,
    agi.SCENARIO,
)
ROBOT_SCENARIOS: dict[str, Scenario] = _catalog(
    robots_modest.SCENARIO,
    robots_substantial.SCENARIO,
    robots_extreme.SCENARIO,
)
COMBINED_SCENARIOS: dict[str, Scenario] = _catalog(
    *(combine(a, r) for a in AI_SCENARIOS.values() for r in ROBOT_SCENARIOS.values())
)
DIAGONAL_SCENARIOS: dict[str, Scenario] = _catalog(
    *(
        combine(a, r)
        for a in AI_SCENARIOS.values()
        for r in ROBOT_SCENARIOS.values()
        if a.rung == r.rung
    )
)
PAPER_SCENARIOS: dict[str, Scenario] = _catalog(
    modest.SCENARIO,
    substantial.SCENARIO,
    extreme.SCENARIO,
)
SCENARIOS: dict[str, Scenario] = {
    **AI_SCENARIOS,
    **ROBOT_SCENARIOS,
    **COMBINED_SCENARIOS,
}

__all__ = [
    "AI_SCENARIOS",
    "COMBINED_SCENARIOS",
    "DIAGONAL_SCENARIOS",
    "PAPER_SCENARIOS",
    "ROBOT_SCENARIOS",
    "SCENARIOS",
    "combine",
]
