"""Calibration of Korinek, Jones, Sacher, Cotter and McCrory (2026),
"Economic Scenarios for Transformative AI", Anthropic Institute WP 2026-02.

Every number below is taken from Table 1 (pp. 23-25) and Table A.2 of the paper,
except the AGI scenario at the end, which is an addition of this repository.
Section references in comments point to the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

DEFAULT_SC_OVER_SL = (
    0.624  # cognitive share of employment; ceiling of the affected mass
)


@dataclass(frozen=True)
class Params:
    """Parameters common to all scenarios (Table 1, panels A and D)."""

    # --- A. Technology, factor markets, ideas production ------------------
    sigma: float = 0.5  # elasticity of substitution across tasks
    sL0: float = 0.60  # base-period labor share
    sC_over_sL: float = DEFAULT_SC_OVER_SL  # cognitive share of employment (CPS 2025)
    r_bar: float = 0.115  # no-AI gross rental rate, per year
    delta: float = 0.05  # depreciation, per year (net return 6.5%)
    t0: float = 2024.0  # base period
    t_anchor: float = 2026.5  # anchor date of the AI paths (mid-2026)
    t_target: float = 2030.0  # date of the scenarios' "2030" values
    h: float = 1.0 / 12.0  # period length, years (monthly)
    lam: float = 1.0  # lambda: returns to research input
    one_minus_phiR: float = 2.86  # fishing out, labor-augmenting units (eq. 40)
    g: float = 0.0167  # no-AI growth of the ideas stock, per year
    n: float = 0.0033  # growth of the labor force, per year
    # mid-2026 anchors of the AI paths, common to the scenarios (panel B)
    m_anchor: float = 0.14  # affected mass, mid-2026
    d_anchor: float = 0.10  # diffusion share, mid-2026
    d_bar: float = 1.0  # ceiling on diffusion

    # --- D. The labor market in normal times ------------------------------
    q_bar_annual: float = 0.11  # normal quit rate, per year
    qT_share: float = 0.55  # share of quits that responds to job prospects
    qC_rel: float = 0.69  # q_C / q_bar (relative separation rates)
    qN_rel: float = 1.52  # q_N / q_bar
    # normal search pool, share of L: CPS 2025 gives 3.84 percent (Section 3.2);
    # Table 1 rounds it to 0.038
    U_bar: float = 0.0384
    mu_bar: float = 0.17  # search discount in normal times
    iota: float = 1.27  # matching curvature (den Haan et al.)
    pi_bar_mean: float = 0.65  # employment-weighted mean filling rate, per month

    # --- derived -----------------------------------------------------------
    @property
    def sK0(self) -> float:
        return 1.0 - self.sL0

    @property
    def sC0(self) -> float:
        return self.sC_over_sL * self.sL0

    @property
    def sN0(self) -> float:
        return self.sL0 - self.sC0

    @property
    def m_bar(self) -> float:
        """Ceiling on the affected mass: all cognitive work (Section 3.1)."""
        return self.sC_over_sL

    @property
    def gA(self) -> float:
        """No-AI growth of measured TFP, s_L g = 1 percent."""
        return self.sL0 * self.g

    @property
    def gamma(self) -> float:
        """Long-run elasticity of ideas w.r.t. research input, eq. (24)."""
        return self.lam / self.one_minus_phiR

    @property
    def L_emp0(self) -> float:
        """Base-period employment L - U_bar (L = 1)."""
        return 1.0 - self.U_bar

    @property
    def lC0(self) -> float:
        return self.sC_over_sL * self.L_emp0

    @property
    def lN0(self) -> float:
        return (1.0 - self.sC_over_sL) * self.L_emp0

    @property
    def q_bar(self) -> float:
        """Normal quit rate per month (0.92 percent, Section 3.2)."""
        return self.q_bar_annual / 12.0

    @property
    def qC_bar(self) -> float:
        return self.qC_rel * self.q_bar

    @property
    def qN_bar(self) -> float:
        return self.qN_rel * self.q_bar


@dataclass(frozen=True)
class Scenario:
    """A scenario: the 2030 values of the AI paths and the disruptiveness
    parameters (Table 1, panels B and C)."""

    name: str
    m2030: float  # affected mass in 2030
    d2030: float  # diffusion share in 2030
    a_anchor: float  # log gain per instance, mid-2026
    g_a: float  # slope of the gain, per year
    psi: float  # automation share (held constant)
    rho: float  # reinstatement ratio
    mu: float  # search discount on the path
    thetaH: float  # posting speed, per month
    xi: float = 0.50  # rigidity of the cognitive wage, per year
    eps: float = 3.0  # elasticity of capital supply (math.inf = pegged rental rate)

    def with_(self, **kw: float) -> Scenario:
        return replace(self, **kw)

    @property
    def a2030(self) -> float:
        return self.a_anchor + self.g_a * (2030.0 - 2026.5)


MODEST = Scenario(
    name="modest",
    m2030=0.2,
    d2030=0.2,
    a_anchor=0.30,
    g_a=0.0,
    psi=0.50,
    rho=0.50,
    mu=0.17,
    thetaH=0.10,
)
SUBSTANTIAL = Scenario(
    name="substantial",
    m2030=0.3,
    d2030=0.4,
    a_anchor=0.35,
    g_a=0.028,
    psi=0.75,
    rho=0.25,
    mu=0.08,
    thetaH=0.25,
)
EXTREME = Scenario(
    name="extreme",
    m2030=0.5,
    d2030=0.6,
    a_anchor=0.45,
    g_a=0.10,
    psi=0.90,
    rho=0.0,
    mu=0.04,
    thetaH=0.50,
)

# An AGI scenario beyond the paper's three (added in this repository, not in the paper).
# AI reaches essentially all cognitive tasks and performs 99 percent of their instances by
# the start of 2030, all of them automated (no augmentation, no new tasks), and the gain
# per task is tenfold (ln 10 = 2.30 log points) at that date, still rising linearly
# afterwards as eq. (8) prescribes.  Frictions are those of the extreme scenario.  The
# affected mass and the diffusion share split the 99 percent symmetrically so that
# neither sits on its ceiling, which the logistic path cannot reach.
AGI_REPLACED_SHARE = 0.99
_AGI_SPLIT = AGI_REPLACED_SHARE**0.5
AGI = Scenario(
    name="agi",
    m2030=_AGI_SPLIT * DEFAULT_SC_OVER_SL,
    d2030=_AGI_SPLIT,
    a_anchor=0.45,
    g_a=(math.log(10.0) - 0.45) / (2030.0 - 2026.5),
    psi=1.0,
    rho=0.0,
    mu=0.04,
    thetaH=0.50,
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (MODEST, SUBSTANTIAL, EXTREME, AGI)
}
PAPER_SCENARIOS: dict[str, Scenario] = {
    s.name: s for s in (MODEST, SUBSTANTIAL, EXTREME)
}

DEFAULT_PARAMS = Params()
