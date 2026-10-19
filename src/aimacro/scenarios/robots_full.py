"""Full robotization of the all-other group (this repository, not the paper).

The AGI scenario mirrored on physical and interpersonal work: robots reach
essentially all of the group's tasks and perform 99 percent of their instances by
2035, all automated, with a tenfold gain per task at that date.  It pairs with the
AGI scenario as the top rung of the ladder.
"""

from __future__ import annotations

from aimacro.calibration import FULL_GAIN, REPLACED_SPLIT, Scenario, robot_shock

SCENARIO = Scenario(
    "robots_full",
    ai=None,
    robots=robot_shock(
        share=REPLACED_SPLIT,
        d=REPLACED_SPLIT,
        a=(0.45, FULL_GAIN),
        psi_rho=(1.0, 0.0),
        frictions=(0.04, 0.50),
    ),
    rung="agi",
)
