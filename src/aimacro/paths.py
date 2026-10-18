"""Exogenous scenario paths for the five objects of a shock (Section 2.1.2, eq. 8, 8')."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.special import expit

if TYPE_CHECKING:
    from .calibration import Params, Scenario, Shock


def logistic_slope(
    ceiling: float,
    x_anchor: float,
    x_target: float,
    years: float,
) -> float:
    """Equation (8'): slope that takes a logistic from x_anchor to x_target in `years`."""
    return (
        math.log((ceiling - x_anchor) / x_anchor * x_target / (ceiling - x_target))
        / years
    )


def logistic_midpoint(
    ceiling: float,
    x_target: float,
    kappa: float,
    t_target: float,
) -> float:
    """Date at which the logistic with ceiling and slope kappa reaches ceiling/2,
    given that it equals x_target at t_target."""
    return t_target + math.log(ceiling / x_target - 1.0) / kappa


def logistic(t: float, ceiling: float, kappa: float, t_mid: float) -> float:
    return ceiling * float(expit(kappa * (t - t_mid)))


def logistic_complement(t: float, kappa: float, t_mid: float) -> float:
    """One minus the logistic over its ceiling, exact when the path is near the ceiling."""
    return float(expit(-kappa * (t - t_mid)))


@dataclass(frozen=True)
class PathPoint:
    m: float  # affected mass
    d: float  # diffusion share
    a: float  # log gain per AI-performed instance
    psi: float  # automation share
    rho: float  # reinstatement ratio
    m_bar: float  # ceiling of the affected mass
    m_frac_c: float  # one minus m/m_bar, exact near the ceiling
    d_c: float  # one minus d, exact near the ceiling


def no_shock(m_bar: float) -> PathPoint:
    """The paths of a group that no shock touches."""
    return PathPoint(
        m=0.0, d=0.0, a=0.0, psi=0.0, rho=0.0, m_bar=m_bar, m_frac_c=1.0, d_c=1.0
    )


class ShockPaths:
    """m_t and d_t follow logistics through the shock's anchor and target values,
    a_t is linear from its anchor, psi and rho are constant."""

    def __init__(self, shock: Shock, m_bar: float, t0: float) -> None:
        self.shock, self.m_bar, self.t0 = shock, m_bar, t0
        years = shock.t_target - shock.t_anchor
        self.kappa_m = logistic_slope(m_bar, shock.m_anchor, shock.m_target, years)
        self.t_m = logistic_midpoint(
            m_bar, shock.m_target, self.kappa_m, shock.t_target
        )
        self.kappa_d = logistic_slope(1.0, shock.d_anchor, shock.d_target, years)
        self.t_d = logistic_midpoint(1.0, shock.d_target, self.kappa_d, shock.t_target)
        self.a0 = shock.a_anchor - shock.g_a * (shock.t_anchor - t0)

    def m(self, t: float) -> float:
        return logistic(t, self.m_bar, self.kappa_m, self.t_m)

    def d(self, t: float) -> float:
        return logistic(t, 1.0, self.kappa_d, self.t_d)

    def a(self, t: float) -> float:
        return self.a0 + self.shock.g_a * (t - self.t0)

    def at(self, t: float) -> PathPoint:
        return PathPoint(
            m=self.m(t),
            d=self.d(t),
            a=self.a(t),
            psi=self.shock.psi,
            rho=self.shock.rho,
            m_bar=self.m_bar,
            m_frac_c=logistic_complement(t, self.kappa_m, self.t_m),
            d_c=logistic_complement(t, self.kappa_d, self.t_d),
        )


class ScenarioPaths:
    """The paths of both groups: AI on the cognitive group, robots on the other."""

    def __init__(self, p: Params, sc: Scenario) -> None:
        self.p = p
        self.C = ShockPaths(sc.ai, p.m_bar("C"), p.t0) if sc.ai is not None else None
        self.N = (
            ShockPaths(sc.robots, p.m_bar("N"), p.t0) if sc.robots is not None else None
        )

    def at(self, t: float) -> tuple[PathPoint, PathPoint]:
        ptC = self.C.at(t) if self.C is not None else no_shock(self.p.m_bar("C"))
        ptN = self.N.at(t) if self.N is not None else no_shock(self.p.m_bar("N"))
        return ptC, ptN
