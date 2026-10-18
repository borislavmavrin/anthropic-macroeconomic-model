"""The task economy at a date: Proposition 1 and the general system (39).

All gaps are log differences from the no-AI path at the same date.  Wages and
output are handled "deflated by A_t" as in Table A.1: x_C, x_N are gaps of
w_o/A, `yhat` is the gap of Y/A, and the ideas gap dlnA is added back where the
paper does.  The rental-rate gap x_r is undeflated.

Derivation (Section 2.1.1-2.1.2).  With the final good as numeraire and base
unit costs all equal to one, the CES price index over task instances is

    sL0*Lambda_C*e^{(1-s)x_C} + sN0*e^{(1-s)x_N} + B*e^{(1-s)x_r} = 1,

where Lambda_C is the surviving base mass of cognitive labor instances
(remaining + augmented at cost e^{-a} + reinstated) and B is capital's base
mass (original capital tasks, less those that drop out when new labor tasks
enter, plus automated instances at cost e^{-a}).  Labor demand per group is the
CES demand for a good of price w_o/A:  l_o/l_o0 = (mass_o/mass_o0) e^{yhat - s x_o}.
Capital demand follows from the capital share, and supply is dlnK = eps*x_r.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from collections.abc import Callable

    from .calibration import Params
    from .paths import PathPoint


@dataclass(frozen=True)
class Instances:
    """Masses of task instances at a date, given the AI paths."""

    Lambda_C: (
        float  # surviving mass of cognitive labor instances (base-wage-bill units)
    )
    B: float  # capital's bracket in the labor-share row, eq. (14)
    lN_tilde: float  # eq. (15): required log change in all-other employment
    md: float  # m_t d_t
    a: float


def instances(p: Params, pt: PathPoint) -> Instances:
    s = p.sigma
    md = pt.m * pt.d
    e_a = math.exp(-(1.0 - s) * pt.a)
    # share of an AI-performed instance's mass that leaves labor
    c = 1.0 - pt.rho * pt.psi - (1.0 - pt.psi) * e_a
    shift = md * c
    # Lambda_C is the ceiling minus m d c.  It is formed from the complements of the
    # logistic paths so that it stays exact when m and d sit at their ceilings (the
    # AGI scenario), where the plain difference would cancel to zero.
    x = (1.0 - pt.m_frac_c) * pt.d
    one_minus_x = pt.m_frac_c + (1.0 - pt.m_frac_c) * pt.d_c
    Lambda_C = pt.m_bar * (one_minus_x + x * (1.0 - c))
    B = p.sK0 + p.sL0 * pt.psi * md * (e_a - pt.rho)
    lN_tilde = -math.log(1.0 - shift)
    return Instances(Lambda_C=Lambda_C, B=B, lN_tilde=lN_tilde, md=md, a=pt.a)


def targets(p: Params, inst: Instances) -> tuple[float, float]:
    """Equation (13): full-mobility employment targets (l*_C, l*_N)."""
    lN_star = p.lN0 * math.exp(inst.lN_tilde)
    return p.lC0 + p.lN0 - lN_star, lN_star


@dataclass(frozen=True)
class Economy:
    x_C: float  # gap of the A-deflated cognitive wage (the MPL at realized employment)
    x_N: float  # gap of the A-deflated all-other wage
    x_r: float  # rental-rate gap
    yhat: float  # gap of Y/A (output deflated by the ideas stock)
    sL: float  # labor share from the price index
    dlnK: float  # capital-stock gap
    lC: float
    lN: float

    @property
    def sK(self) -> float:
        return 1.0 - self.sL


def _solve_capital_market(
    F: Callable[[float], float],
    eps: float,
    x_hi: float,
) -> float:
    """Root of F(x_r) = dlnK_demand(x_r) - eps*x_r, strictly decreasing in x_r,
    on (-inf, x_hi) where x_hi is the rental gap at which the capital share hits one."""
    if math.isinf(eps):
        return 0.0
    lo = -5.0
    while F(lo) <= 0:
        lo -= 5.0
    gap = 1e-8
    hi = x_hi - gap
    while F(hi) >= 0:  # analytically F -> -inf at x_hi
        gap *= 0.1
        hi = x_hi - gap
    return float(brentq(F, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=500))


@dataclass(frozen=True)
class TaskEconomy:
    """System (39) at one date: parameters, task masses, the ideas gap and the
    capital-supply elasticity.  Methods solve it for different unknowns."""

    p: Params
    inst: Instances
    dlnA: float
    eps: float

    def _sK(self, x_r: float) -> float:
        return self.inst.B * math.exp((1.0 - self.p.sigma) * x_r)

    def _close(
        self,
        yhat_of: Callable[[float], float],
        x_hi: float,
    ) -> tuple[float, float]:
        """Clear the capital market given output as a function of the rental gap."""

        def F(x_r: float) -> float:
            dlnK = math.log(self._sK(x_r) / self.p.sK0) + yhat_of(x_r) + self.dlnA - x_r
            return dlnK - self.eps * x_r

        x_r = _solve_capital_market(F, self.eps, x_hi)
        return x_r, yhat_of(x_r)

    def at_employment(self, lC: float, lN: float) -> Economy:
        """Given employment in both groups: the wages that make firms demand exactly
        that employment, the rental rate and output.  At the targets this is
        Proposition 1 with x_C == x_N."""
        p, inst = self.p, self.inst
        s = p.sigma
        k = (1.0 - s) / s
        cC = math.log(lC / p.lC0 * p.sC_over_sL / inst.Lambda_C)  # yhat minus s x_C
        cN = math.log(lN / p.lN0)  # yhat minus s x_N
        Q = p.sL0 * inst.Lambda_C * math.exp(-k * cC) + p.sN0 * math.exp(-k * cN)

        def yhat_of(x_r: float) -> float:
            return math.log((1.0 - self._sK(x_r)) / Q) / k

        x_r, yhat = self._close(yhat_of, math.log(1.0 / inst.B) / (1.0 - s))
        sK = self._sK(x_r)
        dlnK = math.log(sK / p.sK0) + yhat + self.dlnA - x_r
        return Economy(
            x_C=(yhat - cC) / s,
            x_N=(yhat - cN) / s,
            x_r=x_r,
            yhat=yhat,
            sL=1.0 - sK,
            dlnK=dlnK,
            lC=lC,
            lN=lN,
        )

    def cognitive_demand(self, x_C: float, lN: float) -> Economy:
        """Cognitive employment l^d_C demanded at a given (sticky) cognitive wage
        gap x_C and given all-other employment lN."""
        p, inst = self.p, self.inst
        s = p.sigma
        k = (1.0 - s) / s
        cN = math.log(lN / p.lN0)
        termC = p.sL0 * inst.Lambda_C * math.exp((1.0 - s) * x_C)

        def yhat_of(x_r: float) -> float:
            return cN + math.log((1.0 - self._sK(x_r) - termC) / p.sN0) / k

        x_hi = math.log((1.0 - termC) / inst.B) / (1.0 - s)
        x_r, yhat = self._close(yhat_of, x_hi)
        sK = self._sK(x_r)
        dlnK = math.log(sK / p.sK0) + yhat + self.dlnA - x_r
        lC = p.lC0 * inst.Lambda_C / p.sC_over_sL * math.exp(yhat - s * x_C)
        return Economy(
            x_C=x_C,
            x_N=(yhat - cN) / s,
            x_r=x_r,
            yhat=yhat,
            sL=1.0 - sK,
            dlnK=dlnK,
            lC=lC,
            lN=lN,
        )


def tfp_gap(p: Params, inst: Instances, dlnA: float) -> float:
    """Equation (45) in the form of Table A.1: measured TFP gap, the base-price
    index over all tasks.  The ideas stock is labor-augmenting, so it lowers the
    cost of labor instances only; to first order TFP rises by sL0 * dlnA (eq. 26)."""
    s = p.sigma
    labor = (
        p.sL0
        * (1.0 - inst.md * (1.0 - math.exp(-(1.0 - s) * inst.a)))
        * math.exp(-(1.0 - s) * dlnA)
    )
    return -math.log(p.sK0 + labor) / (1.0 - s)
