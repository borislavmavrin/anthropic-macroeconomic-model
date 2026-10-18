"""Monthly simulation of the model with unemployment (Appendix A, Table A.1).

Each month the nine steps of Appendix A are evaluated in order.  The state
carried from month to month is employment and the unemployment pool by group,
last month's job-finding rates, last month's cognitive wage discount to the
common wage, and the ideas gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .calibration import DEFAULT_PARAMS, Params, Scenario
from .labor import matches, steady_state
from .paths import AIPaths
from .static import Economy, Instances, TaskEconomy, instances, targets, tfp_gap


@dataclass(frozen=True)
class _WageBlock:
    """Steps 2-4: targets, the common wage, the sticky cognitive wage and demand."""

    fe: Economy  # full-employment economy (Proposition 1)
    lC_star: float
    lN_star: float
    gapC: float  # ln(w_C / w), the cognitive discount to the common wage
    x_C: float  # sticky cognitive wage gap (A-deflated)
    lC_d: float  # cognitive labor demand at the sticky wage
    E: float  # excess cognitive employment
    Z: float  # cognitive shortfall of employment behind demand
    G_C: float  # cognitive overhang
    B_N: float  # all-other shortfall


@dataclass(frozen=True)
class _Flows:
    """Steps 5-6: quits, layoffs, openings, matching and job-finding rates."""

    qC: float
    qN: float
    D_C: float
    v_C: float
    v_N: float
    H_C: float
    H_N: float
    fC: float
    fN: float


class _Simulator:
    def __init__(self, sc: Scenario, p: Params) -> None:
        self.sc, self.p = sc, p
        self.ss = steady_state(p)
        self.paths = AIPaths(p, sc)
        self.xi_m = sc.xi ** (1.0 / 12.0) if sc.xi > 0 else 0.0
        # state
        self.lC, self.lN = p.lC0, p.lN0
        self.uC, self.uN = self.ss.UC, self.ss.UN
        self.fC_prev, self.fN_prev = self.ss.fC, self.ss.fN
        self.gapC_prev = 0.0
        self.dlnA = 0.0

    def _wages(self, econ: TaskEconomy, inst_next: Instances) -> _WageBlock:
        p, ss = self.p, self.ss
        lC_star, lN_star = targets(p, econ.inst)
        fe = econ.at_employment(lC_star, lN_star)
        lC_star_next, lN_star_next = targets(p, inst_next)
        N_C = self.lC + max(0.0, self.uC - ss.UC)
        clearing = econ.at_employment(N_C, self.lN)
        gapC = self.xi_m * self.gapC_prev + (1.0 - self.xi_m) * (clearing.x_C - fe.x_C)
        x_C = fe.x_C + gapC
        lC_d = econ.cognitive_demand(x_C, self.lN).lC
        return _WageBlock(
            fe=fe,
            lC_star=lC_star,
            lN_star=lN_star,
            gapC=gapC,
            x_C=x_C,
            lC_d=lC_d,
            E=max(0.0, self.lC - lC_d),
            Z=max(0.0, lC_d - self.lC),
            G_C=max(0.0, math.log(self.lC) - math.log(lC_star_next)),
            B_N=max(0.0, math.log(lN_star_next) - math.log(self.lN)),
        )

    def _flows(self, w: _WageBlock) -> _Flows:
        ss, sc, p = self.ss, self.sc, self.p
        qC = ss.qXC + ss.qTC * self.fC_prev / ss.fC
        qN = ss.qXN + ss.qTN * self.fN_prev / ss.fN
        D_C = max(0.0, w.E - qC * self.lC)
        v_C = (max(0.0, qC * self.lC - w.E) + sc.thetaH * w.Z) / ss.piC
        v_N = (qN + sc.thetaH * w.B_N) * self.lN / ss.piN
        S_C = self.uC + sc.mu * self.uN
        S_N = sc.mu * self.uC + self.uN
        H_C = matches(ss.chi, p.iota, S_C, v_C)
        H_N = matches(ss.chi, p.iota, S_N, v_N)
        return _Flows(
            qC=qC,
            qN=qN,
            D_C=D_C,
            v_C=v_C,
            v_N=v_N,
            H_C=H_C,
            H_N=H_N,
            fC=H_C / S_C + sc.mu * H_N / S_N,
            fN=sc.mu * H_C / S_C + H_N / S_N,
        )

    def _record(self, econ: TaskEconomy, w: _WageBlock, f: _Flows) -> dict[str, float]:
        """Step 8: the actual economy at realized employment."""
        p = self.p
        act = econ.at_employment(self.lC, self.lN)
        # the cognitive wage paid is the sticky wage, or the MPL where employment
        # trails demand
        x_C_paid = act.x_C if w.Z > 0 else w.x_C
        dlnw_C, dlnw_N = x_C_paid + self.dlnA, act.x_N + self.dlnA
        wage_bill = math.exp(dlnw_C) * self.lC + math.exp(dlnw_N) * self.lN
        return {
            "md": econ.inst.md,
            "a": econ.inst.a,
            "dlnA": self.dlnA,
            "dlnY": act.yhat + self.dlnA,
            "yhat": act.yhat,
            "x_r": act.x_r,
            "dlnK": act.dlnK,
            "sL": act.sL,
            "dlnw_C": dlnw_C,
            "dlnw_C_sticky": w.x_C + self.dlnA,
            "mpl_C": act.x_C + self.dlnA,
            "dlnw_N": dlnw_N,
            "dlnw_common": w.fe.x_C + self.dlnA,
            "dlnw_avg": math.log(wage_bill / (self.lC + self.lN)),
            "dln_labor_income": math.log(wage_bill / p.L_emp0),
            "dln_cog_wage_bill": math.log(math.exp(dlnw_C) * self.lC / p.lC0),
            "tfp": tfp_gap(p, econ.inst, self.dlnA),
            "lC": self.lC,
            "lN": self.lN,
            "UC": self.uC,
            "UN": self.uN,
            "lC_star": w.lC_star,
            "lN_star": w.lN_star,
            "lC_d": w.lC_d,
            "qC": f.qC,
            "qN": f.qN,
            "D_C": f.D_C,
            "v_C": f.v_C,
            "v_N": f.v_N,
            "H_C": f.H_C,
            "H_N": f.H_N,
            "fC": f.fC,
            "fN": f.fN,
            "G_C": w.G_C,
            "B_N": w.B_N,
            "E": w.E,
            "Z": w.Z,
            "x_r_fe": w.fe.x_r,
            "dlnY_fe": w.fe.yhat + self.dlnA,
            "sL_fe": w.fe.sL,
        }

    def _advance(self, dlnY: float, w: _WageBlock, f: _Flows) -> None:
        """Steps 9 and 7: the ideas step (eq. 42) and the stock updates (eq. 36-37)."""
        p = self.p
        dg = p.g * (math.exp(p.lam * dlnY - p.one_minus_phiR * self.dlnA) - 1.0)
        self.dlnA += p.h * dg
        lC, lN, uC, uN = self.lC, self.lN, self.uC, self.uN
        self.lC = (1.0 - f.qC) * lC - f.D_C + f.H_C
        self.lN = (1.0 - f.qN) * lN + f.H_N
        self.uC = uC + f.qC * lC + f.D_C - f.fC * uC
        self.uN = uN + f.qN * lN - f.fN * uN
        self.fC_prev, self.fN_prev = f.fC, f.fN
        self.gapC_prev = w.gapC

    def month(self, t: float) -> dict[str, float]:
        pt = self.paths.at(t)
        inst = instances(self.p, pt)
        inst_next = instances(self.p, self.paths.at(t + self.p.h))
        econ = TaskEconomy(self.p, inst, self.dlnA, self.sc.eps)
        w = self._wages(econ, inst_next)
        f = self._flows(w)
        row = self._record(econ, w, f)
        row["m"], row["d"] = pt.m, pt.d
        self._advance(row["dlnY"], w, f)
        return row


def simulate(
    sc: Scenario,
    p: Params = DEFAULT_PARAMS,
    t_end: float = 2030.0,
) -> pd.DataFrame:
    """Simulate scenario `sc` monthly from p.t0 through t_end (inclusive).

    Returns a DataFrame indexed by date (years, p.t0 + k/12) with gaps in logs
    (not percent) and head counts as shares of the labor force L = 1.
    """
    sim = _Simulator(sc, p)
    n_months = round((t_end - p.t0) / p.h) + 1
    times = [round(p.t0 + k * p.h, 6) for k in range(n_months)]
    rows = [sim.month(p.t0 + k * p.h) for k in range(n_months)]
    df = pd.DataFrame(rows, index=pd.Index(times, name="t"))
    df.attrs["scenario"] = sc
    df.attrs["params"] = p
    return df
