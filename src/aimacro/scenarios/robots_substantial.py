"""Substantial robotization of the all-other group (this repository, not the paper).

The substantial AI scenario mirrored on physical and interpersonal work: the same
affected share and diffusion of the group's work reached in 2035, a gain that
reaches the AI scenario's 2030 value in 2035, a higher automation share, and the
substantial AI frictions for displaced workers moving into cognitive occupations.
"""

from __future__ import annotations

from aimacro.calibration import Scenario, robot_shock

SCENARIO = Scenario(
    "robots_substantial",
    ai=None,
    robots=robot_shock(
        share=0.3,
        d=0.4,
        a=(0.35, 0.45),
        psi_rho=(0.90, 0.25),
        frictions=(0.08, 0.25),
    ),
    rung="substantial",
)
