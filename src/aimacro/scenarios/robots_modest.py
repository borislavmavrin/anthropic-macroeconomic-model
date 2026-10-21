"""Modest robotization of the all-other group (this repository, not the paper).

The paper's modest AI scenario applied to physical and interpersonal work:
the same five objects, mid-2026 anchors, 2030 scenario date and frictions for the
displaced, with the affected mass rescaled to the all-other group's share of tasks.
"""

from __future__ import annotations

from aimacro.calibration import Scenario, on_other_group

from .modest import SHOCK as AI_SHOCK

SCENARIO = Scenario(
    "robots_modest",
    ai=None,
    robots=on_other_group(AI_SHOCK),
    rung="modest",
)
