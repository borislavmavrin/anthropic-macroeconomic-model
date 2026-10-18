"""The task economy at a date: Proposition 1 and the general system (39), for
shocks on either or both groups of occupations.

All gaps are log differences from the no-AI path at the same date.  Wages and
output are handled "deflated by A_t" as in Table A.1: x_C, x_N are gaps of
w_o/A, `yhat` is the gap of Y/A, and the ideas gap dlnA is added back where the
paper does.  The rental-rate gap x_r is undeflated.

With the final good as numeraire and base unit costs all equal to one, the CES
price index over task instances is

    sL0*Lambda_C*e^{(1-s)x_C} + sL0*Lambda_N*e^{(1-s)x_N} + B*e^{(1-s)x_r} = 1,

where Lambda_g is the surviving base mass of group g's labor instances
(remaining + augmented at cost e^{-a} + reinstated) and B is capital's base
mass (original capital tasks, less those that drop out when new labor tasks
enter, plus automated instances at cost e^{-a}).  Labor demand per group is the
CES demand for a good of price w_g/A:  l_g/l_g0 = (Lambda_g/m_bar_g) e^{yhat - s x_g}.
Capital demand follows from the capital share, and supply is dlnK = eps*x_r.
The paper is the case Lambda_N = m_bar_N.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from .calibration import other

if TYPE_CHECKING:
    from collections.abc import Callable

    from .calibration import Group, Params
    from .paths import PathPoint

LAMBDA_FLOOR = 1e-200  # keeps logs finite once a group's labor tasks have all gone
W_UNDERFLOW = -1e4  # a log share this low is below the smallest positive double


@dataclass(frozen=True)
class Instances:
    """Masses of task instances at a date, given both groups' paths."""

    Lambda_C: (
        float  # surviving mass of cognitive labor instances (base-wage-bill units)
    )
    Lambda_N: float  # same for the all-other group
    B: float  # capital's bracket in the labor-share row, eq. (14)
    tfp_loss: float  # sum over groups of m d (1 - e^{-(1-s)a}), eq. (45)
    md_C: float
    md_N: float
    a_C: float
    a_N: float

    def Lambda(self, g: Group) -> float:
        return self.Lambda_C if g == "C" else self.Lambda_N


def _group(p: Params, pt: PathPoint) -> tuple[float, float, float]:
    """One group's surviving labor mass, its addition to capital's bracket, and
    its contribution to the TFP gain."""
    s = p.sigma
    md = pt.m * pt.d
    e_a = math.exp(-(1.0 - s) * pt.a)
    # share of an AI-performed instance's mass that leaves labor
    c = 1.0 - pt.rho * pt.psi - (1.0 - pt.psi) * e_a
    # Lambda is the ceiling minus m d c, formed from the complements of the logistic
    # paths so that it stays exact when m and d sit at their ceilings, where the
    # plain difference would cancel to zero.
    x = (1.0 - pt.m_frac_c) * pt.d
    one_minus_x = pt.m_frac_c + (1.0 - pt.m_frac_c) * pt.d_c
    lam = max(pt.m_bar * (one_minus_x + x * (1.0 - c)), LAMBDA_FLOOR)
    return lam, pt.psi * md * (e_a - pt.rho), md * (1.0 - e_a)


def instances(p: Params, ptC: PathPoint, ptN: PathPoint) -> Instances:
    lamC, bC, lossC = _group(p, ptC)
    lamN, bN, lossN = _group(p, ptN)
    return Instances(
        Lambda_C=lamC,
        Lambda_N=lamN,
        B=p.sK0 + p.sL0 * (bC + bN),
        tfp_loss=lossC + lossN,
        md_C=ptC.m * ptC.d,
        md_N=ptN.m * ptN.d,
        a_C=ptC.a,
        a_N=ptN.a,
    )


def targets(p: Params, inst: Instances) -> dict[Group, float]:
    """Equation (13) generalized: with free mobility and a common wage, employment
    splits in proportion to the groups' surviving task masses."""
    total = inst.Lambda_C + inst.Lambda_N
    return {
        "C": p.L_emp0 * inst.Lambda_C / total,
        "N": p.L_emp0 * inst.Lambda_N / total,
    }


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

    def x(self, g: Group) -> float:
        return self.x_C if g == "C" else self.x_N

    def emp(self, g: Group) -> float:
        return self.lC if g == "C" else self.lN


def _solve_capital_market(
    F: Callable[[float], float],
    w_hi: float,
) -> float:
    """Root of F(w), strictly increasing in w on (-inf, w_hi), where w is the log
    of the expenditure share left to the labor side of the price index once
    capital's share is taken out.  Working in w keeps every quantity finite even
    when labor's share goes to zero (all tasks automated)."""
    lo = -50.0
    while F(lo) >= 0:
        lo *= 2.0
        if lo < W_UNDERFLOW:
            return lo
    gap = 1e-8
    hi = w_hi - gap
    while F(hi) <= 0:  # analytically F -> +inf at w_hi
        gap *= 0.1
        hi = w_hi - gap
    return float(brentq(F, lo, hi, xtol=1e-14, rtol=1e-13, maxiter=500))


@dataclass(frozen=True)
class TaskEconomy:
    """System (39) at one date: parameters, task masses, the ideas gap and the
    capital-supply elasticity.  Methods solve it for different unknowns.

    The price index is solved in the variable w = ln(1 - s_K - term), the log of
    the expenditure share that the groups' labor instances must absorb; `term` is
    the share already fixed by a group with a given wage.  Given w, capital's
    share, the rental gap and output follow in closed form, and the capital-supply
    schedule dlnK = eps x_r is one monotone equation in w.
    """

    p: Params
    inst: Instances
    dlnA: float
    eps: float

    def _ln_c(self, g: Group, emp: float) -> float:
        """yhat minus sigma x_g implied by employment emp in group g."""
        p = self.p
        return (
            math.log(emp / p.l0(g))
            + math.log(p.m_bar(g))
            - math.log(self.inst.Lambda(g))
        )

    def _close(self, slope: float, term: float) -> tuple[float, float, float]:
        """Clear the capital market.  Output is yhat = slope + w / k; returns
        (w, x_r, yhat)."""
        p = self.p
        s = p.sigma
        k = (1.0 - s) / s
        ceiling = 1.0 - term  # s_K + e^w must stay below it
        w_hi = math.log(ceiling)

        def x_r_of(w: float) -> float:
            sK = ceiling - math.exp(w)
            return math.log(sK / self.inst.B) / (1.0 - s)

        def F(w: float) -> float:
            x_r = x_r_of(w)
            sK = ceiling - math.exp(w)
            dlnK = math.log(sK / p.sK0) + slope + w / k + self.dlnA - x_r
            return dlnK - self.eps * x_r

        if math.isinf(self.eps):
            w = math.log(ceiling - self.inst.B)
        else:
            w = _solve_capital_market(F, w_hi)
        return w, x_r_of(w), slope + w / k

    def _economy(
        self,
        x: dict[Group, float],
        x_r: float,
        yhat: float,
        emp: dict[Group, float],
    ) -> Economy:
        sK = self.inst.B * math.exp((1.0 - self.p.sigma) * x_r)
        dlnK = math.log(sK / self.p.sK0) + yhat + self.dlnA - x_r
        return Economy(
            x_C=x["C"],
            x_N=x["N"],
            x_r=x_r,
            yhat=yhat,
            sL=1.0 - sK,
            dlnK=dlnK,
            lC=emp["C"],
            lN=emp["N"],
        )

    def at_employment(self, lC: float, lN: float) -> Economy:
        """Given employment in both groups: the wages that make firms demand exactly
        that employment, the rental rate and output.  At the targets this is
        Proposition 1 with x_C == x_N."""
        p, inst = self.p, self.inst
        s = p.sigma
        k = (1.0 - s) / s
        c = {"C": self._ln_c("C", lC), "N": self._ln_c("N", lN)}
        # ln Q, Q = sL0 (Lambda_C e^{-k c_C} + Lambda_N e^{-k c_N}), in logs throughout
        lnQ = math.log(p.sL0) + float(
            np.logaddexp(
                math.log(inst.Lambda_C) - k * c["C"],
                math.log(inst.Lambda_N) - k * c["N"],
            ),
        )
        _w, x_r, yhat = self._close(-lnQ / k, 0.0)
        x: dict[Group, float] = {"C": (yhat - c["C"]) / s, "N": (yhat - c["N"]) / s}
        return self._economy(x, x_r, yhat, {"C": lC, "N": lN})

    def demand_at_wage(self, shed: Group, x_shed: float, l_other: float) -> Economy:
        """Employment demanded in the shedding group at a given (sticky) wage gap,
        given the other group's employment."""
        p, inst = self.p, self.inst
        s = p.sigma
        k = (1.0 - s) / s
        ab = other(shed)
        c_ab = self._ln_c(ab, l_other)
        term = p.sL0 * inst.Lambda(shed) * math.exp((1.0 - s) * x_shed)
        ln_lam_ab = math.log(p.sL0) + math.log(inst.Lambda(ab))
        _w, x_r, yhat = self._close(c_ab - ln_lam_ab / k, term)
        ln_l_shed = (
            math.log(p.l0(shed))
            + math.log(inst.Lambda(shed))
            - math.log(p.m_bar(shed))
            + yhat
            - s * x_shed
        )
        x: dict[Group, float] = {shed: x_shed, ab: (yhat - c_ab) / s}
        return self._economy(x, x_r, yhat, {shed: math.exp(ln_l_shed), ab: l_other})


def tfp_gap(p: Params, inst: Instances, dlnA: float) -> float:
    """Equation (45) in the form of Table A.1: measured TFP gap, the base-price
    index over all tasks.  The ideas stock is labor-augmenting, so it lowers the
    cost of labor instances only; to first order TFP rises by sL0 * dlnA (eq. 26)."""
    s = p.sigma
    labor = p.sL0 * (1.0 - inst.tfp_loss) * math.exp(-(1.0 - s) * dlnA)
    return -math.log(p.sK0 + labor) / (1.0 - s)
