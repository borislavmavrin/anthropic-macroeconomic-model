"""Calibration of Korinek, Jones, Sacher, Cotter and McCrory (2026),
"Economic Scenarios for Transformative AI", Anthropic Institute WP 2026-02,
extended with robotization of the all-other (physical and interpersonal) occupations.

Panel A and D numbers are taken from Table 1 (pp. 23-25) and Table A.2 of the paper.
The AI shocks reproduce the paper's three scenarios.  The AGI shock, the robot shocks
and all combinations are additions of this repository.
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
T_TARGET_ROBOTS = 2035.0  # date of the robot scenario values (robots lag AI)


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


# --- AI shocks on the cognitive group (Table 1) ----------------------------------

AI_SHOCKS: dict[str, Shock] = {
    "modest": Shock(
        m_target=0.2,
        d_target=0.2,
        a_anchor=0.30,
        g_a=0.0,
        psi=0.50,
        rho=0.50,
        mu=0.17,
        thetaH=0.10,
    ),
    "substantial": Shock(
        m_target=0.3,
        d_target=0.4,
        a_anchor=0.35,
        g_a=0.028,
        psi=0.75,
        rho=0.25,
        mu=0.08,
        thetaH=0.25,
    ),
    "extreme": Shock(
        m_target=0.5,
        d_target=0.6,
        a_anchor=0.45,
        g_a=0.10,
        psi=0.90,
        rho=0.0,
        mu=0.04,
        thetaH=0.50,
    ),
}

# An AGI shock beyond the paper's three.  AI reaches essentially all cognitive tasks
# and performs 99 percent of their instances by the start of 2030, all automated,
# with a tenfold gain per task at that date.  The affected mass and the diffusion
# share split the 99 percent symmetrically so that neither sits on its ceiling.
REPLACED_SHARE = 0.99
_SPLIT = REPLACED_SHARE**0.5
AI_SHOCKS["agi"] = Shock(
    m_target=_SPLIT * M_BAR_C,
    d_target=_SPLIT,
    a_anchor=0.45,
    g_a=(math.log(10.0) - 0.45) / (T_TARGET_AI - T_ANCHOR),
    psi=1.0,
    rho=0.0,
    mu=0.04,
    thetaH=0.50,
)

# --- Robot shocks on the all-other group (this repository, not the paper) --------
# The AI ladder mirrored on physical and interpersonal work: the same affected
# shares and diffusion of the group's work, reached in 2035 rather than 2030 from
# near-zero mid-2026 anchors (2 percent of the group's work affected, 5 percent
# diffusion), gains that reach the AI ladder's 2030 values in 2035, a higher
# automation share (robots replace rather than assist), and the AI ladder's
# frictions for displaced workers moving into cognitive occupations.

ROBOT_ANCHOR_SHARE = 0.02  # of all-other work affected, mid-2026
ROBOT_ANCHOR_DIFFUSION = 0.05


def _robot_shock(
    share: float,
    d: float,
    a: tuple[float, float],
    psi_rho: tuple[float, float],
    frictions: tuple[float, float],
) -> Shock:
    a_anchor, a_target = a
    return Shock(
        m_target=share * M_BAR_N,
        d_target=d,
        a_anchor=a_anchor,
        g_a=(a_target - a_anchor) / (T_TARGET_ROBOTS - T_ANCHOR),
        psi=psi_rho[0],
        rho=psi_rho[1],
        mu=frictions[0],
        thetaH=frictions[1],
        m_anchor=ROBOT_ANCHOR_SHARE * M_BAR_N,
        d_anchor=ROBOT_ANCHOR_DIFFUSION,
        t_target=T_TARGET_ROBOTS,
    )


ROBOT_SHOCKS: dict[str, Shock] = {
    "modest": _robot_shock(0.2, 0.2, (0.30, 0.30), (0.90, 0.50), (0.17, 0.10)),
    "substantial": _robot_shock(0.3, 0.4, (0.35, 0.45), (0.90, 0.25), (0.08, 0.25)),
    "extreme": _robot_shock(0.5, 0.6, (0.45, 0.80), (0.95, 0.0), (0.04, 0.50)),
    "full": _robot_shock(
        _SPLIT,
        _SPLIT,
        (0.45, math.log(10.0)),
        (1.0, 0.0),
        (0.04, 0.50),
    ),
}

# --- The scenario catalog --------------------------------------------------------

RUNG_OF = {"full": "agi"}  # the robot ladder's top rung pairs with AGI

AI_SCENARIOS: dict[str, Scenario] = {
    name: Scenario(name, ai=shock, robots=None, rung=name)
    for name, shock in AI_SHOCKS.items()
}
ROBOT_SCENARIOS: dict[str, Scenario] = {
    f"robots_{name}": Scenario(
        f"robots_{name}",
        ai=None,
        robots=shock,
        rung=RUNG_OF.get(name, name),
    )
    for name, shock in ROBOT_SHOCKS.items()
}


def combine(ai: Scenario, robots: Scenario) -> Scenario:
    return Scenario(
        f"{ai.name}+{robots.name}",
        ai=ai.ai,
        robots=robots.robots,
        eps=ai.eps,
        rung=ai.rung,
    )


COMBINED_SCENARIOS: dict[str, Scenario] = {
    combine(a, r).name: combine(a, r)
    for a in AI_SCENARIOS.values()
    for r in ROBOT_SCENARIOS.values()
}
DIAGONAL_SCENARIOS: dict[str, Scenario] = {
    sc.name: sc
    for sc in COMBINED_SCENARIOS.values()
    if sc.robots is not None
    and sc.ai is not None
    and ROBOT_SCENARIOS[sc.name.split("+")[1]].rung == sc.rung
}

PAPER_SCENARIOS: dict[str, Scenario] = {
    k: AI_SCENARIOS[k] for k in ("modest", "substantial", "extreme")
}
SCENARIOS: dict[str, Scenario] = {
    **AI_SCENARIOS,
    **ROBOT_SCENARIOS,
    **COMBINED_SCENARIOS,
}

DEFAULT_PARAMS = Params()
