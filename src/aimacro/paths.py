"""Exogenous scenario paths for the five AI objects (Section 2.1.2, eq. 8 and 8')."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .calibration import Params, Scenario


def logistic_slope(ceiling: float, x_anchor: float, x_target: float, years: float) -> float:
    """Equation (8'): slope that takes a logistic from x_anchor to x_target in `years`."""
    return math.log((ceiling - x_anchor) / x_anchor * x_target / (ceiling - x_target)) / years


def logistic_midpoint(ceiling: float, x_target: float, kappa: float, t_target: float) -> float:
    """Date at which the logistic with ceiling and slope kappa reaches ceiling/2,
    given that it equals x_target at t_target."""
    return t_target + math.log(ceiling / x_target - 1.0) / kappa


def logistic(t: float, ceiling: float, kappa: float, t_mid: float) -> float:
    return ceiling / (1.0 + math.exp(-kappa * (t - t_mid)))


@dataclass(frozen=True)
class PathPoint:
    m: float    # affected mass
    d: float    # diffusion share
    a: float    # log gain per AI-performed instance
    psi: float  # automation share
    rho: float  # reinstatement ratio


class AIPaths:
    """m_t and d_t follow logistics through the mid-2026 anchor and the 2030 value,
    a_t is linear from its mid-2026 anchor, psi and rho are constant."""

    def __init__(self, p: Params, sc: Scenario):
        self.p, self.sc = p, sc
        years = p.t_target - p.t_anchor
        self.m_bar = p.m_bar
        self.kappa_m = logistic_slope(p.m_bar, p.m_anchor, sc.m2030, years)
        self.t_m = logistic_midpoint(p.m_bar, sc.m2030, self.kappa_m, p.t_target)
        self.d_bar = p.d_bar
        self.kappa_d = logistic_slope(p.d_bar, p.d_anchor, sc.d2030, years)
        self.t_d = logistic_midpoint(p.d_bar, sc.d2030, self.kappa_d, p.t_target)
        # a_t = a_0 + g_a (t - t0), anchored at mid-2026
        self.a0 = sc.a_anchor - sc.g_a * (p.t_anchor - p.t0)

    def m(self, t: float) -> float:
        return logistic(t, self.m_bar, self.kappa_m, self.t_m)

    def d(self, t: float) -> float:
        return logistic(t, self.d_bar, self.kappa_d, self.t_d)

    def a(self, t: float) -> float:
        return self.a0 + self.sc.g_a * (t - self.p.t0)

    def at(self, t: float) -> PathPoint:
        return PathPoint(m=self.m(t), d=self.d(t), a=self.a(t), psi=self.sc.psi, rho=self.sc.rho)
