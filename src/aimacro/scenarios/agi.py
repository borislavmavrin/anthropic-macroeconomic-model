"""An AGI scenario beyond the paper's three (this repository, not the paper).

AI reaches essentially all cognitive tasks and performs 99 percent of their
instances by the start of 2030, all automated, with a tenfold gain per task at
that date.  The frictions are those of the extreme scenario.
"""

from __future__ import annotations

from aimacro.calibration import (
    FULL_GAIN,
    M_BAR_C,
    REPLACED_SPLIT,
    T_ANCHOR,
    T_TARGET_AI,
    Scenario,
    Shock,
)

SCENARIO = Scenario(
    "agi",
    ai=Shock(
        m_target=REPLACED_SPLIT * M_BAR_C,
        d_target=REPLACED_SPLIT,
        a_anchor=0.45,
        g_a=(FULL_GAIN - 0.45) / (T_TARGET_AI - T_ANCHOR),
        psi=1.0,
        rho=0.0,
        mu=0.04,
        thetaH=0.50,
    ),
    robots=None,
    rung="agi",
)
