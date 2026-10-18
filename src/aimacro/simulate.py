"""Monthly simulation of the model with unemployment (Appendix A, Table A.1),
generalized so that either group of occupations can shed workers.

Each month the group that sits above its next full-mobility target is the
*shedding* group: its wage is sticky (eq. 30), firms lay off the workers they do
not demand at that wage (eq. 31), and its openings replace quits and close any
gap of demand beyond employment.  The other group is the *absorbing* group: it is
paid its marginal product and posts openings for a fraction theta^H of its
shortfall (eq. 32).  With a shock on the cognitive group only, the cognitive group
always sheds and the all-other group always absorbs, which is the paper's model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .calibration import DEFAULT_PARAMS, Group, Params, Scenario, other
from .labor import matches, steady_state
from .paths import ScenarioPaths
from .static import Economy, Instances, TaskEconomy, instances, targets, tfp_gap

GROUPS: tuple[Group, Group] = ("C", "N")


@dataclass(frozen=True)
class _WageBlock:
    """Steps 2-4: targets, the common wage, roles, the sticky wage and demand."""

    fe: Economy  # full-employment economy (Proposition 1)
    l_star: dict[Group, float]
    shed: Group
    gap: float  # the shedding group's log wage discount to the common wage
    x_shed: float  # sticky wage gap of the shedding group (A-deflated)
    l_d: float  # labor demand in the shedding group at the sticky wage
    E: float  # excess employment in the shedding group
    Z: float  # employment trailing demand in the shedding group
    G: float  # overhang of the shedding group against next month's target
    B: float  # shortfall of the absorbing group against next month's target


@dataclass(frozen=True)
class _Flows:
    """Steps 5-6: quits, layoffs, openings, matching and job-finding rates."""

    q: dict[Group, float]
    D: dict[Group, float]
    v: dict[Group, float]
    H: dict[Group, float]
    f: dict[Group, float]


class _Simulator:
    def __init__(self, sc: Scenario, p: Params) -> None:
        self.sc, self.p = sc, p
        ss = steady_state(p)
        self.ss = ss
        self.paths = ScenarioPaths(p, sc)
        self.U_bar: dict[Group, float] = {"C": ss.UC, "N": ss.UN}
        self.pi: dict[Group, float] = {"C": ss.piC, "N": ss.piN}
        self.qX: dict[Group, float] = {"C": ss.qXC, "N": ss.qXN}
        self.qT: dict[Group, float] = {"C": ss.qTC, "N": ss.qTN}
        self.f_bar: dict[Group, float] = {"C": ss.fC, "N": ss.fN}
        self.xi_m: dict[Group, float] = {
            g: sc.xi(g) ** (1.0 / 12.0) if sc.xi(g) > 0 else 0.0 for g in GROUPS
        }
        # state
        self.l: dict[Group, float] = {"C": p.lC0, "N": p.lN0}
        self.u: dict[Group, float] = {"C": ss.UC, "N": ss.UN}
        self.f_prev: dict[Group, float] = dict(self.f_bar)
        self.gap_prev: dict[Group, float] = {"C": 0.0, "N": 0.0}
        self.dlnA = 0.0

    def _wages(self, econ: TaskEconomy, inst_next: Instances) -> _WageBlock:
        p = self.p
        l_star = targets(p, econ.inst)
        fe = econ.at_employment(l_star["C"], l_star["N"])
        l_star_next = targets(p, inst_next)
        over = {g: math.log(self.l[g]) - math.log(l_star_next[g]) for g in GROUPS}
        # the shedding group is the one whose target is falling: on the paper's
        # paths the cognitive target only falls and the other only rises; with two
        # shocks the roles switch only when one displacement overtakes the other
        fall = {g: math.log(l_star_next[g]) - math.log(l_star[g]) for g in GROUPS}
        shed: Group = "C" if fall["C"] <= fall["N"] else "N"
        ab = other(shed)
        attached = dict(self.l)
        attached[shed] = self.l[shed] + max(0.0, self.u[shed] - self.U_bar[shed])
        clearing = econ.at_employment(attached["C"], attached["N"])
        xi_m = self.xi_m[shed]
        gap = xi_m * self.gap_prev[shed] + (1.0 - xi_m) * (clearing.x(shed) - fe.x_C)
        x_shed = fe.x_C + gap
        l_d = econ.demand_at_wage(shed, x_shed, self.l[ab]).emp(shed)
        return _WageBlock(
            fe=fe,
            l_star=l_star,
            shed=shed,
            gap=gap,
            x_shed=x_shed,
            l_d=l_d,
            E=max(0.0, self.l[shed] - l_d),
            Z=max(0.0, l_d - self.l[shed]),
            G=max(0.0, over[shed]),
            B=max(0.0, math.log(l_star_next[ab]) - math.log(self.l[ab])),
        )

    def _flows(self, w: _WageBlock) -> _Flows:
        sc, p, ss = self.sc, self.p, self.ss
        shed, ab = w.shed, other(w.shed)
        q: dict[Group, float] = {
            g: self.qX[g] + self.qT[g] * self.f_prev[g] / self.f_bar[g] for g in GROUPS
        }
        D: dict[Group, float] = {"C": 0.0, "N": 0.0}
        D[shed] = max(0.0, w.E - q[shed] * self.l[shed])
        v: dict[Group, float] = {"C": 0.0, "N": 0.0}
        v[shed] = (
            max(0.0, q[shed] * self.l[shed] - w.E) + sc.thetaH_into(shed) * w.Z
        ) / self.pi[shed]
        v[ab] = (q[ab] + sc.thetaH_into(ab) * w.B) * self.l[ab] / self.pi[ab]
        mu_C, mu_N = (
            sc.mu_from("C"),
            sc.mu_from("N"),
        )  # discount outside the origin group
        S: dict[Group, float] = {
            "C": self.u["C"] + mu_N * self.u["N"],
            "N": mu_C * self.u["C"] + self.u["N"],
        }
        H: dict[Group, float] = {g: matches(ss.chi, p.iota, S[g], v[g]) for g in GROUPS}
        f: dict[Group, float] = {
            "C": H["C"] / S["C"] + mu_C * H["N"] / S["N"],
            "N": mu_N * H["C"] / S["C"] + H["N"] / S["N"],
        }
        return _Flows(q=q, D=D, v=v, H=H, f=f)

    def _record(self, econ: TaskEconomy, w: _WageBlock, f: _Flows) -> dict[str, float]:
        """Step 8: the actual economy at realized employment."""
        p = self.p
        shed, ab = w.shed, other(w.shed)
        act = econ.at_employment(self.l["C"], self.l["N"])
        # the shedding group is paid the sticky wage, or its MPL where employment
        # trails demand; the absorbing group is paid its MPL
        x_paid: dict[Group, float] = {ab: act.x(ab)}
        x_paid[shed] = act.x(shed) if w.Z > 0 else w.x_shed
        dlnw = {g: x_paid[g] + self.dlnA for g in GROUPS}
        bill = {g: math.exp(dlnw[g]) * self.l[g] for g in GROUPS}
        # carry the absorbing group's actual discount so that a role switch starts
        # its sticky wage from the wage it is being paid
        self.gap_prev[ab] = act.x(ab) - w.fe.x_C
        self.gap_prev[shed] = w.gap
        inst = econ.inst
        return {
            "dlnA": self.dlnA,
            "dlnY": act.yhat + self.dlnA,
            "yhat": act.yhat,
            "x_r": act.x_r,
            "dlnK": act.dlnK,
            "sL": act.sL,
            "dlnw_C": dlnw["C"],
            "dlnw_N": dlnw["N"],
            "mpl_C": act.x_C + self.dlnA,
            "mpl_N": act.x_N + self.dlnA,
            "dlnw_common": w.fe.x_C + self.dlnA,
            "x_sticky": w.x_shed + self.dlnA,
            "dlnw_avg": math.log((bill["C"] + bill["N"]) / (self.l["C"] + self.l["N"])),
            "dln_labor_income": math.log((bill["C"] + bill["N"]) / p.L_emp0),
            "dln_cog_wage_bill": math.log(bill["C"] / p.lC0),
            "dln_other_wage_bill": math.log(bill["N"] / p.lN0),
            "tfp": tfp_gap(p, inst, self.dlnA),
            "lC": self.l["C"],
            "lN": self.l["N"],
            "UC": self.u["C"],
            "UN": self.u["N"],
            "lC_star": w.l_star["C"],
            "lN_star": w.l_star["N"],
            "shed_C": 1.0 if shed == "C" else 0.0,
            "l_d_shed": w.l_d,
            "qC": f.q["C"],
            "qN": f.q["N"],
            "D_C": f.D["C"],
            "D_N": f.D["N"],
            "v_C": f.v["C"],
            "v_N": f.v["N"],
            "H_C": f.H["C"],
            "H_N": f.H["N"],
            "fC": f.f["C"],
            "fN": f.f["N"],
            "G": w.G,
            "B": w.B,
            "E": w.E,
            "Z": w.Z,
            "x_r_fe": w.fe.x_r,
            "dlnY_fe": w.fe.yhat + self.dlnA,
            "sL_fe": w.fe.sL,
            "md_C": inst.md_C,
            "md_N": inst.md_N,
            "a_C": inst.a_C,
            "a_N": inst.a_N,
        }

    def _advance(self, dlnY: float, f: _Flows) -> None:
        """Steps 9 and 7: the ideas step (eq. 42) and the stock updates (eq. 36-37)."""
        p = self.p
        dg = p.g * (math.exp(p.lam * dlnY - p.one_minus_phiR * self.dlnA) - 1.0)
        self.dlnA += p.h * dg
        emp, pool = dict(self.l), dict(self.u)
        for g in GROUPS:
            self.l[g] = (1.0 - f.q[g]) * emp[g] - f.D[g] + f.H[g]
            self.u[g] = pool[g] + f.q[g] * emp[g] + f.D[g] - f.f[g] * pool[g]
        self.f_prev = dict(f.f)

    def month(self, t: float) -> dict[str, float]:
        ptC, ptN = self.paths.at(t)
        inst = instances(self.p, ptC, ptN)
        inst_next = instances(self.p, *self.paths.at(t + self.p.h))
        econ = TaskEconomy(self.p, inst, self.dlnA, self.sc.eps)
        w = self._wages(econ, inst_next)
        f = self._flows(w)
        row = self._record(econ, w, f)
        row.update({"m_C": ptC.m, "d_C": ptC.d, "m_N": ptN.m, "d_N": ptN.d})
        self._advance(row["dlnY"], f)
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
