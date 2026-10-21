"""Substantial robotization of the all-other group (this repository, not the paper).

The paper's substantial AI scenario applied to physical and interpersonal work:
the same five objects, mid-2026 anchors, 2030 scenario date and frictions for the
displaced, with the affected mass rescaled to the all-other group's share of tasks.
"""

from __future__ import annotations

from aimacro.calibration import Scenario, on_other_group

from .substantial import SHOCK as AI_SHOCK

SCENARIO = Scenario(
    "robots_substantial",
    ai=None,
    robots=on_other_group(AI_SHOCK),
    rung="substantial",
)
