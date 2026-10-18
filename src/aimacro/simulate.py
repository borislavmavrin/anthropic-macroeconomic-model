"""Monthly simulation of the model with unemployment (Appendix A, Table A.1).

Each month the nine steps of Appendix A are evaluated in order.  The state
carried from month to month is employment and the unemployment pool by group,
last month's job-finding rates, last month's cognitive wage discount to the
common wage, and the ideas gap.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calibration import DEFAULT_PARAMS, Params, Scenario
from .labor import matches, steady_state
from .paths import AIPaths
from .static import instances, solve_cognitive_demand, solve_economy, targets, tfp_gap


def simulate(sc: Scenario, p: Params = DEFAULT_PARAMS, t_end: float = 2030.0) -> pd.DataFrame:
    """Simulate scenario `sc` monthly from p.t0 through t_end (inclusive).

    Returns a DataFrame indexed by date (years, p.t0 + k/12) with gaps in logs
    (not percent) and head counts as shares of the labor force L = 1.
    """
    ss = steady_state(p)
    paths = AIPaths(p, sc)
    n_months = int(round((t_end - p.t0) / p.h)) + 1
    xi_m = sc.xi ** (1.0 / 12.0) if sc.xi > 0.0 else 0.0
    one_minus_phiR = p.one_minus_phiR

    # state
    lC, lN = p.lC0, p.lN0
    UC, UN = ss.UC, ss.UN
    fC_prev, fN_prev = ss.fC, ss.fN
    gapC_prev = 0.0        # ln(w_{C,t-1} / w_{t-1})
    dlnA = 0.0

    rows = []
    for k in range(n_months):
        t = p.t0 + k * p.h

        # 1. Paths at t and t+1
        inst = instances(p, paths.at(t))
        inst_next = instances(p, paths.at(t + p.h))

        # 2-3. Full-employment economy (Proposition 1): rental gap, common wage, targets
        lC_star, lN_star = targets(p, inst)
        fe = solve_economy(p, inst, lC_star, lN_star, dlnA, sc.eps)
        x_common = fe.x_C
        lC_star_next, lN_star_next = targets(p, inst_next)

        # 4. Gaps and the cognitive wage
        G_C = max(0.0, math.log(lC) - math.log(lC_star_next))
        B_N = max(0.0, math.log(lN_star_next) - math.log(lN))
        N_C = lC + max(0.0, UC - ss.UC)
        clearing = solve_economy(p, inst, N_C, lN, dlnA, sc.eps)
        gapC = xi_m * gapC_prev + (1.0 - xi_m) * (clearing.x_C - x_common)
        x_C = x_common + gapC
        demand = solve_cognitive_demand(p, inst, x_C, lN, dlnA, sc.eps)
        lC_d = demand.lC
        E = max(0.0, lC - lC_d)
        Z = max(0.0, lC_d - lC)

        # 5. Separations and openings
        qC = ss.qXC + ss.qTC * fC_prev / ss.fC
        qN = ss.qXN + ss.qTN * fN_prev / ss.fN
        D_C = max(0.0, E - qC * lC)
        v_C = (max(0.0, qC * lC - E) + sc.thetaH * Z) / ss.piC
        v_N = (qN + sc.thetaH * B_N) * lN / ss.piN

        # 6. Matching
        S_C = UC + sc.mu * UN
        S_N = sc.mu * UC + UN
        H_C = matches(ss.chi, p.iota, S_C, v_C)
        H_N = matches(ss.chi, p.iota, S_N, v_N)
        fC = H_C / S_C + sc.mu * H_N / S_N
        fN = sc.mu * H_C / S_C + H_N / S_N

        # 8. Reporting: the actual economy at realized employment
        act = solve_economy(p, inst, lC, lN, dlnA, sc.eps)
        mpl_C = act.x_C
        x_C_paid = mpl_C if Z > 0.0 else x_C     # MPL where employment trails demand
        dlnY = act.yhat + dlnA
        dlnw_N = act.x_N + dlnA
        dlnw_C = x_C_paid + dlnA
        emp = lC + lN
        wage_bill = math.exp(dlnw_C) * lC + math.exp(dlnw_N) * lN
        tfp = tfp_gap(p, inst, dlnA)

        rows.append(dict(
            t=t, m=inst.md / paths.d(t) if paths.d(t) > 0 else np.nan, d=paths.d(t), md=inst.md, a=inst.a,
            dlnA=dlnA, dlnY=dlnY, yhat=act.yhat,
            x_r=act.x_r, dlnK=act.dlnK, sL=act.sL,
            dlnw_C=dlnw_C, dlnw_C_sticky=x_C + dlnA, mpl_C=mpl_C + dlnA, dlnw_N=dlnw_N,
            dlnw_common=x_common + dlnA, dlnw_avg=math.log(wage_bill / emp),
            dln_labor_income=math.log(wage_bill / p.L_emp0),
            dln_cog_wage_bill=math.log(math.exp(dlnw_C) * lC / p.lC0),
            tfp=tfp,
            lC=lC, lN=lN, UC=UC, UN=UN, lC_star=lC_star, lN_star=lN_star, lC_d=lC_d,
            qC=qC, qN=qN, D_C=D_C, v_C=v_C, v_N=v_N, H_C=H_C, H_N=H_N, fC=fC, fN=fN,
            G_C=G_C, B_N=B_N, E=E, Z=Z,
            x_r_fe=fe.x_r, dlnY_fe=fe.yhat + dlnA, sL_fe=fe.sL,
        ))

        # 9. Ideas: research uplift equals the GDP gap; monthly step on eq. (42)
        dg = p.g * (math.exp(p.lam * dlnY - one_minus_phiR * dlnA) - 1.0)
        dlnA = dlnA + p.h * dg

        # 7. Stocks
        lC_next = (1.0 - qC) * lC - D_C + H_C
        lN_next = (1.0 - qN) * lN + H_N
        UC_next = UC + qC * lC + D_C - fC * UC
        UN_next = UN + qN * lN - fN * UN
        lC, lN, UC, UN = lC_next, lN_next, UC_next, UN_next
        fC_prev, fN_prev = fC, fN
        gapC_prev = gapC

    df = pd.DataFrame(rows).set_index("t")
    df.index = np.round(df.index.to_numpy(), 6)
    df.attrs["scenario"] = sc
    df.attrs["params"] = p
    return df
