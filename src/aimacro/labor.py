"""The frictional labor market: steady state (eq. 38) and matching function (eq. 34)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from .calibration import Params


def matches(chi: float, iota: float, S: float, v: float) -> float:
    """den Haan, Ramey and Watson (2000) matching function, eq. (34)."""
    if S <= 0 or v <= 0:
        return 0.0
    return chi * S * v / (S**iota + v**iota) ** (1.0 / iota)


def filling_rate(chi: float, iota: float, hires_per_search: float) -> float:
    """Invert the matching function: filling rate given H/S (eq. 38, third condition)."""
    x = (hires_per_search / chi) ** iota
    return chi * (1.0 - x) ** (1.0 / iota)


@dataclass(frozen=True)
class SteadyState:
    UC: float
    UN: float
    HC: float
    HN: float
    SC: float
    SN: float
    fC: float
    fN: float
    piC: float
    piN: float
    chi: float
    qC: float
    qN: float
    qXC: float
    qTC: float
    qXN: float
    qTN: float


def steady_state(p: Params) -> SteadyState:
    """Normal-times steady state of the flow block, evaluated at mu_bar (Section 2.3.2)."""
    qC, qN = p.qC_bar, p.qN_bar
    HC, HN = qC * p.lC0, qN * p.lN0
    mu = p.mu_bar
    U = p.U_bar

    def fC_of(uC: float) -> float:
        uN = U - uC
        SC, SN = uC + mu * uN, mu * uC + uN
        return HC / SC + mu * HN / SN

    # split the pool so that the cognitive inflow equals the cognitive outflow
    UC = float(brentq(lambda u: fC_of(u) * u - HC, 1e-9, U - 1e-9, xtol=1e-15))
    UN = U - UC
    SC, SN = UC + mu * UN, mu * UC + UN
    fC = HC / SC + mu * HN / SN
    fN = mu * HC / SC + HN / SN

    # matching efficiency chi: employment-weighted mean filling rate = pi_bar_mean
    def mean_pi(chi: float) -> float:
        piC = filling_rate(chi, p.iota, HC / SC)
        piN = filling_rate(chi, p.iota, HN / SN)
        return (p.lC0 * piC + p.lN0 * piN) / p.L_emp0

    chi_lo = max(HC / SC, HN / SN) * (1.0 + 1e-9)
    chi = float(brentq(lambda c: mean_pi(c) - p.pi_bar_mean, chi_lo, 5.0, xtol=1e-14))
    piC = filling_rate(chi, p.iota, HC / SC)
    piN = filling_rate(chi, p.iota, HN / SN)
    return SteadyState(
        UC=UC,
        UN=UN,
        HC=HC,
        HN=HN,
        SC=SC,
        SN=SN,
        fC=fC,
        fN=fN,
        piC=piC,
        piN=piN,
        chi=chi,
        qC=qC,
        qN=qN,
        qXC=(1.0 - p.qT_share) * qC,
        qTC=p.qT_share * qC,
        qXN=(1.0 - p.qT_share) * qN,
        qTN=p.qT_share * qN,
    )
