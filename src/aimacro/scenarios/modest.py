"""The paper's modest AI scenario (Table 1, panel B, column 1)."""

from __future__ import annotations

from aimacro.calibration import Scenario, Shock

SCENARIO = Scenario(
    "modest",
    ai=Shock(
        m_target=0.2,
        d_target=0.2,
        a_anchor=0.30,
        g_a=0.0,
        psi=0.50,
        rho=0.50,
        mu=0.17,
        thetaH=0.10,
    ),
    robots=None,
    rung="modest",
)
