"""Calibration of Korinek, Jones, Sacher, Cotter and McCrory (2026),
"Economic Scenarios for Transformative AI", Anthropic Institute WP 2026-02,
extended with robotization of the all-other (physical and interpersonal) occupations.

Panel A and D numbers are taken from Table 1 (pp. 23-25) and Table A.2 of the paper.
The scenarios themselves live one per file in aimacro.scenarios: the paper's three AI
scenarios, and the AGI, robot and combined scenarios this repository adds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Literal

Group = Literal["C", "N"]  # cognitive occupations, all other occupations

DEFAULT_SC_OVER_SL = (
    0.624  # cognitive share of employment; ceiling of AI's affected mass
)
M_BAR_C = (
    DEFAULT_SC_OVER_SL  # all cognitive work, in units of the economy's labor tasks
)
M_BAR_N = 1.0 - DEFAULT_SC_OVER_SL  # all other work, same units
T_ANCHOR = 2026.5  # anchor date of the AI paths (mid-2026)
T_TARGET_AI = 2030.0  # date of the paper's scenario values


def other(g: Group) -> Group:
    return "N" if g == "C" else "C"


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
    h: float = 1.0 / 12.0  # period length, years (monthly)
    lam: float = 1.0  # lambda: returns to research input
    one_minus_phiR: float = 2.86  # fishing out, labor-augmenting units (eq. 40)
    g: float = 0.0167  # no-AI growth of the ideas stock, per year
    n: float = 0.0033  # growth of the labor force, per year

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

    def m_bar(self, g: Group) -> float:
        """Ceiling on a shock's affected mass: all of the group's work (Section 3.1)."""
        return self.sC_over_sL if g == "C" else 1.0 - self.sC_over_sL

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

    def l0(self, g: Group) -> float:
        return self.lC0 if g == "C" else self.lN0

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
class Shock:
    """The five scenario objects of Section 2.1.2 for one group of occupations,
    with the frictions that govern the workers it displaces.

    Masses are in units of the economy's labor tasks, as in the paper.  The paths
    are logistics through (t_anchor, m_anchor) and (t_target, m_target) for the
    affected mass, likewise for diffusion, and linear in time for the gain.
    """

    m_target: float  # affected mass at t_target
    d_target: float  # diffusion share at t_target
    a_anchor: float  # log gain per instance at t_anchor
    g_a: float  # slope of the gain, per year
    psi: float  # automation share (held constant)
    rho: float  # reinstatement ratio
    mu: float  # search discount of this group's displaced workers in the other group
    thetaH: float  # posting speed of the other group when it absorbs them, per month
    xi: float = 0.50  # rigidity of this group's wage while it sheds workers, per year
    m_anchor: float = (
        0.14  # affected mass at t_anchor (AI: observed exposure, mid-2026)
    )
    d_anchor: float = 0.10  # diffusion at t_anchor (AI: firm use, mid-2026)
    t_anchor: float = T_ANCHOR
    t_target: float = T_TARGET_AI

    @property
    def a_target(self) -> float:
        return self.a_anchor + self.g_a * (self.t_target - self.t_anchor)


@dataclass(frozen=True)
class Scenario:
    """An AI shock on the cognitive group, a robot shock on the all-other group,
    either of which may be absent, and the elasticity of capital supply."""

    name: str
    ai: Shock | None
    robots: Shock | None
    eps: float = 3.0  # elasticity of capital supply (math.inf = pegged rental rate)
    rung: str = "modest"  # step of the scenario ladder, used for figure colours

    def with_(self, **kw: float) -> Scenario:
        """Return a copy with scenario fields or shock fields replaced.  Shock
        fields go to the AI shock when there is one, else to the robot shock."""
        own = {f.name for f in fields(Scenario)}
        mine = {k: v for k, v in kw.items() if k in own}
        theirs = {k: v for k, v in kw.items() if k not in own}
        sc = replace(self, **mine) if mine else self
        if not theirs:
            return sc
        if sc.ai is not None:
            return replace(sc, ai=replace(sc.ai, **theirs))
        if sc.robots is not None:
            return replace(sc, robots=replace(sc.robots, **theirs))
        return sc

    def shock(self, g: Group) -> Shock | None:
        return self.ai if g == "C" else self.robots

    def shock_for(self, g: Group) -> Shock:
        """The shock on group g, or the other group's shock where g has none (so a
        single-shock scenario uses one set of frictions, as in the paper)."""
        own = self.shock(g)
        if own is not None:
            return own
        fallback = self.shock(other(g))
        if fallback is None:
            msg = f"scenario {self.name!r} has no shock"
            raise ValueError(msg)
        return fallback

    def xi(self, g: Group) -> float:
        """Rigidity of group g's wage while it sheds workers."""
        return self.shock_for(g).xi

    def mu_from(self, g: Group) -> float:
        """Search discount of workers of origin g searching in the other group."""
        return self.shock_for(g).mu

    def thetaH_into(self, g: Group) -> float:
        """Posting speed of group g when it hires: the speed set by the shock that
        displaces workers into it."""
        return self.shock_for(other(g)).thetaH


# --- Building blocks of the scenario files in aimacro.scenarios ------------------

# The AGI shock: AI reaches essentially all of the group's tasks and performs 99
# percent of their instances at the target date, all automated, with a tenfold gain
# per task.  The affected mass and the diffusion share split the 99 percent
# symmetrically so that neither sits on its ceiling.
REPLACED_SHARE = 0.99
REPLACED_SPLIT = REPLACED_SHARE**0.5
FULL_GAIN = math.log(10.0)  # log gain per instance at the target date


def on_other_group(ai: Shock) -> Shock:
    """The AI shock mirrored on the all-other group: the same objects, anchors, dates
    and frictions, with the affected mass rescaled from cognitive to all-other tasks
    so that it covers the same share of the group's work."""
    scale = M_BAR_N / M_BAR_C
    return replace(ai, m_target=ai.m_target * scale, m_anchor=ai.m_anchor * scale)


DEFAULT_PARAMS = Params()
