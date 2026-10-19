"""Extreme robotization of the all-other group (this repository, not the paper).

The extreme AI scenario mirrored on physical and interpersonal work: the same
affected share and diffusion of the group's work reached in 2035, a gain that
reaches the AI scenario's 2030 value in 2035, a higher automation share, and the
extreme AI frictions for displaced workers moving into cognitive occupations.
"""

from __future__ import annotations

from aimacro.calibration import Scenario, robot_shock

SCENARIO = Scenario(
    "robots_extreme",
    ai=None,
    robots=robot_shock(
        share=0.5,
        d=0.6,
        a=(0.45, 0.80),
        psi_rho=(0.95, 0.0),
        frictions=(0.04, 0.50),
    ),
    rung="extreme",
)
