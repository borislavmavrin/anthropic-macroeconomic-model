"""The paper's extreme AI scenario (Table 1, panel B, column 3)."""

from __future__ import annotations

from aimacro.calibration import Scenario, Shock

SCENARIO = Scenario(
    "extreme",
    ai=Shock(
        m_target=0.5,
        d_target=0.6,
        a_anchor=0.45,
        g_a=0.10,
        psi=0.90,
        rho=0.0,
        mu=0.04,
        thetaH=0.50,
    ),
    robots=None,
    rung="extreme",
)
