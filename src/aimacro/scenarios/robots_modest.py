"""Modest robotization of the all-other group (this repository, not the paper).

The modest AI scenario mirrored on physical and interpersonal work: the same
affected share and diffusion of the group's work reached in 2035, a constant gain,
a higher automation share (robots replace rather than assist), and the modest AI
frictions for displaced workers moving into cognitive occupations.
"""

from __future__ import annotations

from aimacro.calibration import Scenario, robot_shock

SCENARIO = Scenario(
    "robots_modest",
    ai=None,
    robots=robot_shock(
        share=0.2,
        d=0.2,
        a=(0.30, 0.30),
        psi_rho=(0.90, 0.50),
        frictions=(0.17, 0.10),
    ),
    rung="modest",
)
