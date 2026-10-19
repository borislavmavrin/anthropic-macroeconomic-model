"""The paper's substantial AI scenario (Table 1, panel B, column 2)."""

from __future__ import annotations

from aimacro.calibration import Scenario, Shock

SCENARIO = Scenario(
    "substantial",
    ai=Shock(
        m_target=0.3,
        d_target=0.4,
        a_anchor=0.35,
        g_a=0.028,
        psi=0.75,
        rho=0.25,
        mu=0.08,
        thetaH=0.25,
    ),
    robots=None,
    rung="substantial",
)
