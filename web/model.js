/*
 * A JavaScript port of the aimacro model (src/aimacro): calibration, paths,
 * the static task economy, the labor-market steady state and the monthly
 * simulation.  It mirrors the Python line by line so that the browser can re-run
 * the model as the dials move.  The Python package remains the reference; this
 * port agrees with it to rounding error.
 */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- calibration
  const SC_OVER_SL = 0.624; // cognitive share of employment
  const M_BAR_C = SC_OVER_SL;
  const M_BAR_N = 1 - SC_OVER_SL;
  const T_ANCHOR = 2026.5;
  const T_TARGET = 2030.0;

  function makeParams(over) {
    const p = Object.assign(
      {
        sigma: 0.5,
        sL0: 0.6,
        sC_over_sL: SC_OVER_SL,
        r_bar: 0.115,
        delta: 0.05,
        t0: 2024.0,
        h: 1 / 12,
        lam: 1.0,
        one_minus_phiR: 2.86,
        g: 0.0167,
        n: 0.0033,
        q_bar_annual: 0.11,
        qT_share: 0.55,
        qC_rel: 0.69,
        qN_rel: 1.52,
        U_bar: 0.0384,
        mu_bar: 0.17,
        iota: 1.27,
        pi_bar_mean: 0.65,
      },
      over || {},
    );
    p.sK0 = 1 - p.sL0;
    p.L_emp0 = 1 - p.U_bar;
    p.lC0 = p.sC_over_sL * p.L_emp0;
    p.lN0 = (1 - p.sC_over_sL) * p.L_emp0;
    p.q_bar = p.q_bar_annual / 12;
    p.qC_bar = p.qC_rel * p.q_bar;
    p.qN_bar = p.qN_rel * p.q_bar;
    p.gA = p.sL0 * p.g;
    p.m_bar = (g) => (g === "C" ? p.sC_over_sL : 1 - p.sC_over_sL);
    p.l0 = (g) => (g === "C" ? p.lC0 : p.lN0);
    return p;
  }
  const DEFAULT_PARAMS = makeParams();

  function makeShock(o) {
    return Object.assign(
      {
        xi: 0.5,
        m_anchor: 0.14,
        d_anchor: 0.1,
        t_anchor: T_ANCHOR,
        t_target: T_TARGET,
      },
      o,
    );
  }

  /** The AI shock mirrored on the all-other group (calibration.on_other_group). */
  function onOtherGroup(ai) {
    const scale = M_BAR_N / M_BAR_C;
    return Object.assign({}, ai, {
      m_target: ai.m_target * scale,
      m_anchor: ai.m_anchor * scale,
    });
  }

  const REPLACED_SPLIT = Math.sqrt(0.99);
  const FULL_GAIN = Math.log(10);

  const AI_PRESETS = {
    modest: makeShock({ m_target: 0.2, d_target: 0.2, a_anchor: 0.3, g_a: 0.0, psi: 0.5, rho: 0.5, mu: 0.17, thetaH: 0.1 }),
    substantial: makeShock({ m_target: 0.3, d_target: 0.4, a_anchor: 0.35, g_a: 0.028, psi: 0.75, rho: 0.25, mu: 0.08, thetaH: 0.25 }),
    extreme: makeShock({ m_target: 0.5, d_target: 0.6, a_anchor: 0.45, g_a: 0.1, psi: 0.9, rho: 0.0, mu: 0.04, thetaH: 0.5 }),
    agi: makeShock({
      m_target: REPLACED_SPLIT * M_BAR_C,
      d_target: REPLACED_SPLIT,
      a_anchor: 0.45,
      g_a: (FULL_GAIN - 0.45) / (T_TARGET - T_ANCHOR),
      psi: 1.0,
      rho: 0.0,
      mu: 0.04,
      thetaH: 0.5,
    }),
  };
  const ROBOT_PRESETS = {
    modest: onOtherGroup(AI_PRESETS.modest),
    substantial: onOtherGroup(AI_PRESETS.substantial),
    extreme: onOtherGroup(AI_PRESETS.extreme),
  };

  const other = (g) => (g === "C" ? "N" : "C");
  const GROUPS = ["C", "N"];
  const shockOf = (sc, g) => (g === "C" ? sc.ai : sc.robots);
  // the paper's modest frictions stand in when neither group is shocked; they
  // then never matter, because nobody is displaced
  const NO_SHOCK_FRICTIONS = AI_PRESETS.modest;
  function shockFor(sc, g) {
    return shockOf(sc, g) || shockOf(sc, other(g)) || NO_SHOCK_FRICTIONS;
  }

  // ---------------------------------------------------------------- numerics
  /** scipy.optimize.brentq, ported from scipy's C implementation. */
  function brentq(f, xa, xb, xtol, rtol, maxiter) {
    xtol = xtol === undefined ? 2e-12 : xtol;
    rtol = rtol === undefined ? 8.881784197001252e-16 : rtol;
    maxiter = maxiter === undefined ? 100 : maxiter;
    let xpre = xa, xcur = xb, xblk = 0, fblk = 0, spre = 0, scur = 0;
    let fpre = f(xpre), fcur = f(xcur);
    if (fpre === 0) return xpre;
    if (fcur === 0) return xcur;
    if (Math.sign(fpre) === Math.sign(fcur)) throw new Error("brentq: f(a) and f(b) must have different signs");
    for (let i = 0; i < maxiter; i++) {
      if (fpre !== 0 && fcur !== 0 && Math.sign(fpre) !== Math.sign(fcur)) {
        xblk = xpre;
        fblk = fpre;
        spre = scur = xcur - xpre;
      }
      if (Math.abs(fblk) < Math.abs(fcur)) {
        xpre = xcur; xcur = xblk; xblk = xpre;
        fpre = fcur; fcur = fblk; fblk = fpre;
      }
      const delta = (xtol + rtol * Math.abs(xcur)) / 2;
      const sbis = (xblk - xcur) / 2;
      if (fcur === 0 || Math.abs(sbis) < delta) return xcur;
      if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
        let stry;
        if (xpre === xblk) {
          stry = (-fcur * (xcur - xpre)) / (fcur - fpre);
        } else {
          const dpre = (fpre - fcur) / (xpre - xcur);
          const dblk = (fblk - fcur) / (xblk - xcur);
          stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre));
        }
        if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
          spre = scur;
          scur = stry;
        } else {
          spre = sbis;
          scur = sbis;
        }
      } else {
        spre = sbis;
        scur = sbis;
      }
      xpre = xcur;
      fpre = fcur;
      if (Math.abs(scur) > delta) xcur += scur;
      else xcur += sbis > 0 ? delta : -delta;
      fcur = f(xcur);
    }
    return xcur;
  }

  const expit = (x) => 1 / (1 + Math.exp(-x));

  function logaddexp(a, b) {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log1p(Math.exp(-Math.abs(a - b)));
  }

  // ---------------------------------------------------------------- paths
  function logisticSlope(ceiling, xa, xt, years) {
    return Math.log(((ceiling - xa) / xa) * (xt / (ceiling - xt))) / years;
  }
  function logisticMidpoint(ceiling, xt, kappa, tt) {
    return tt + Math.log(ceiling / xt - 1) / kappa;
  }

  function noShock(m_bar) {
    return { m: 0, d: 0, a: 0, psi: 0, rho: 0, m_bar, m_frac_c: 1, d_c: 1 };
  }

  function shockPaths(shock, m_bar, t0) {
    const years = shock.t_target - shock.t_anchor;
    const kappa_m = logisticSlope(m_bar, shock.m_anchor, shock.m_target, years);
    const t_m = logisticMidpoint(m_bar, shock.m_target, kappa_m, shock.t_target);
    const kappa_d = logisticSlope(1, shock.d_anchor, shock.d_target, years);
    const t_d = logisticMidpoint(1, shock.d_target, kappa_d, shock.t_target);
    const a0 = shock.a_anchor - shock.g_a * (shock.t_anchor - t0);
    return (t) => ({
      m: m_bar * expit(kappa_m * (t - t_m)),
      d: expit(kappa_d * (t - t_d)),
      a: a0 + shock.g_a * (t - t0),
      psi: shock.psi,
      rho: shock.rho,
      m_bar,
      m_frac_c: expit(-kappa_m * (t - t_m)),
      d_c: expit(-kappa_d * (t - t_d)),
    });
  }

  function scenarioPaths(p, sc) {
    const C = sc.ai ? shockPaths(sc.ai, p.m_bar("C"), p.t0) : null;
    const N = sc.robots ? shockPaths(sc.robots, p.m_bar("N"), p.t0) : null;
    return (t) => [C ? C(t) : noShock(p.m_bar("C")), N ? N(t) : noShock(p.m_bar("N"))];
  }

  // ---------------------------------------------------------------- static
  const LAMBDA_FLOOR = 1e-200;
  const W_UNDERFLOW = -1e4;

  function groupMass(p, pt) {
    const s = p.sigma;
    const md = pt.m * pt.d;
    const e_a = Math.exp(-(1 - s) * pt.a);
    const c = 1 - pt.rho * pt.psi - (1 - pt.psi) * e_a;
    const x = (1 - pt.m_frac_c) * pt.d;
    const one_minus_x = pt.m_frac_c + (1 - pt.m_frac_c) * pt.d_c;
    const lam = Math.max(pt.m_bar * (one_minus_x + x * (1 - c)), LAMBDA_FLOOR);
    return [lam, pt.psi * md * (e_a - pt.rho), md * (1 - e_a)];
  }

  function instances(p, ptC, ptN) {
    const [lamC, bC, lossC] = groupMass(p, ptC);
    const [lamN, bN, lossN] = groupMass(p, ptN);
    return {
      Lambda_C: lamC,
      Lambda_N: lamN,
      B: p.sK0 + p.sL0 * (bC + bN),
      tfp_loss: lossC + lossN,
      md_C: ptC.m * ptC.d,
      md_N: ptN.m * ptN.d,
      a_C: ptC.a,
      a_N: ptN.a,
      Lambda: (g) => (g === "C" ? lamC : lamN),
    };
  }

  function targets(p, inst) {
    const total = inst.Lambda_C + inst.Lambda_N;
    return { C: (p.L_emp0 * inst.Lambda_C) / total, N: (p.L_emp0 * inst.Lambda_N) / total };
  }

  function solveCapitalMarket(F, w_hi) {
    let lo = -50;
    while (F(lo) >= 0) {
      lo *= 2;
      if (lo < W_UNDERFLOW) return lo;
    }
    let gap = 1e-8;
    let hi = w_hi - gap;
    while (F(hi) <= 0) {
      gap *= 0.1;
      hi = w_hi - gap;
    }
    return brentq(F, lo, hi, 1e-14, 1e-13, 500);
  }

  function taskEconomy(p, inst, dlnA, eps) {
    const s = p.sigma;
    const k = (1 - s) / s;

    function lnC(g, emp) {
      return Math.log(emp / p.l0(g)) + Math.log(p.m_bar(g)) - Math.log(inst.Lambda(g));
    }

    function close(slope, term) {
      const ceiling = 1 - term;
      const w_hi = Math.log(ceiling);
      const x_r_of = (w) => Math.log((ceiling - Math.exp(w)) / inst.B) / (1 - s);
      const F = (w) => {
        const x_r = x_r_of(w);
        const sK = ceiling - Math.exp(w);
        const dlnK = Math.log(sK / p.sK0) + slope + w / k + dlnA - x_r;
        return dlnK - eps * x_r;
      };
      const w = eps === Infinity ? Math.log(ceiling - inst.B) : solveCapitalMarket(F, w_hi);
      return [w, x_r_of(w), slope + w / k];
    }

    function economy(x, x_r, yhat, emp) {
      const sK = inst.B * Math.exp((1 - s) * x_r);
      const dlnK = Math.log(sK / p.sK0) + yhat + dlnA - x_r;
      return {
        x_C: x.C,
        x_N: x.N,
        x_r,
        yhat,
        sL: 1 - sK,
        dlnK,
        lC: emp.C,
        lN: emp.N,
        x: (g) => (g === "C" ? x.C : x.N),
        emp: (g) => (g === "C" ? emp.C : emp.N),
      };
    }

    function atEmployment(lC, lN) {
      const c = { C: lnC("C", lC), N: lnC("N", lN) };
      const lnQ =
        Math.log(p.sL0) +
        logaddexp(Math.log(inst.Lambda_C) - k * c.C, Math.log(inst.Lambda_N) - k * c.N);
      const [, x_r, yhat] = close(-lnQ / k, 0);
      return economy({ C: (yhat - c.C) / s, N: (yhat - c.N) / s }, x_r, yhat, { C: lC, N: lN });
    }

    function demandAtWage(shed, x_shed, l_other) {
      const ab = other(shed);
      const c_ab = lnC(ab, l_other);
      const term = p.sL0 * inst.Lambda(shed) * Math.exp((1 - s) * x_shed);
      const ln_lam_ab = Math.log(p.sL0) + Math.log(inst.Lambda(ab));
      const [, x_r, yhat] = close(c_ab - ln_lam_ab / k, term);
      const ln_l_shed =
        Math.log(p.l0(shed)) + Math.log(inst.Lambda(shed)) - Math.log(p.m_bar(shed)) + yhat - s * x_shed;
      const x = {};
      x[shed] = x_shed;
      x[ab] = (yhat - c_ab) / s;
      const emp = {};
      emp[shed] = Math.exp(ln_l_shed);
      emp[ab] = l_other;
      return economy(x, x_r, yhat, emp);
    }

    return { inst, atEmployment, demandAtWage };
  }

  function tfpGap(p, inst, dlnA) {
    const s = p.sigma;
    const labor = p.sL0 * (1 - inst.tfp_loss) * Math.exp(-(1 - s) * dlnA);
    return -Math.log(p.sK0 + labor) / (1 - s);
  }

  // ---------------------------------------------------------------- labor
  function matches(chi, iota, S, v) {
    if (S <= 0 || v <= 0) return 0;
    return (chi * S * v) / Math.pow(Math.pow(S, iota) + Math.pow(v, iota), 1 / iota);
  }

  function fillingRate(chi, iota, hiresPerSearch) {
    const x = Math.pow(hiresPerSearch / chi, iota);
    return chi * Math.pow(1 - x, 1 / iota);
  }

  function steadyState(p) {
    const qC = p.qC_bar, qN = p.qN_bar;
    const HC = qC * p.lC0, HN = qN * p.lN0;
    const mu = p.mu_bar, U = p.U_bar;
    const fC_of = (uC) => {
      const uN = U - uC;
      return HC / (uC + mu * uN) + (mu * HN) / (mu * uC + uN);
    };
    const UC = brentq((u) => fC_of(u) * u - HC, 1e-9, U - 1e-9, 1e-15);
    const UN = U - UC;
    const SC = UC + mu * UN, SN = mu * UC + UN;
    const fC = HC / SC + (mu * HN) / SN;
    const fN = (mu * HC) / SC + HN / SN;
    const meanPi = (chi) =>
      (p.lC0 * fillingRate(chi, p.iota, HC / SC) + p.lN0 * fillingRate(chi, p.iota, HN / SN)) / p.L_emp0;
    const chi_lo = Math.max(HC / SC, HN / SN) * (1 + 1e-9);
    const chi = brentq((c) => meanPi(c) - p.pi_bar_mean, chi_lo, 5.0, 1e-14);
    return {
      UC, UN, fC, fN, chi,
      piC: fillingRate(chi, p.iota, HC / SC),
      piN: fillingRate(chi, p.iota, HN / SN),
      qXC: (1 - p.qT_share) * qC,
      qTC: p.qT_share * qC,
      qXN: (1 - p.qT_share) * qN,
      qTN: p.qT_share * qN,
    };
  }

  // ---------------------------------------------------------------- simulate
  function simulate(sc, p, tEnd) {
    p = p || DEFAULT_PARAMS;
    tEnd = tEnd === undefined ? 2030 : tEnd;
    const ss = steadyState(p);
    const paths = scenarioPaths(p, sc);
    const eps = sc.eps === undefined ? 3.0 : sc.eps;
    const U_bar = { C: ss.UC, N: ss.UN };
    const pi = { C: ss.piC, N: ss.piN };
    const qX = { C: ss.qXC, N: ss.qXN };
    const qT = { C: ss.qTC, N: ss.qTN };
    const f_bar = { C: ss.fC, N: ss.fN };
    const xi_m = {};
    for (const g of GROUPS) {
      const xi = shockFor(sc, g).xi;
      xi_m[g] = xi > 0 ? Math.pow(xi, 1 / 12) : 0;
    }
    const mu_from = (g) => shockFor(sc, g).mu;
    const thetaH_into = (g) => shockFor(sc, other(g)).thetaH;

    let l = { C: p.lC0, N: p.lN0 };
    let u = { C: ss.UC, N: ss.UN };
    let f_prev = Object.assign({}, f_bar);
    const gap_prev = { C: 0, N: 0 };
    let dlnA = 0;

    const nMonths = Math.round((tEnd - p.t0) / p.h) + 1;
    const out = {};
    const push = (row) => {
      for (const key in row) (out[key] || (out[key] = [])).push(row[key]);
    };

    for (let km = 0; km < nMonths; km++) {
      const t = p.t0 + km * p.h;
      const [ptC, ptN] = paths(t);
      const inst = instances(p, ptC, ptN);
      const instNext = instances(p, ...paths(t + p.h));
      const econ = taskEconomy(p, inst, dlnA, eps);

      // --- wages: targets, roles, the sticky wage and demand
      const l_star = targets(p, inst);
      const fe = econ.atEmployment(l_star.C, l_star.N);
      const l_star_next = targets(p, instNext);
      const fall = {};
      const over = {};
      for (const g of GROUPS) {
        over[g] = Math.log(l[g]) - Math.log(l_star_next[g]);
        fall[g] = Math.log(l_star_next[g]) - Math.log(l_star[g]);
      }
      const shed = fall.C <= fall.N ? "C" : "N";
      const ab = other(shed);
      const attached = Object.assign({}, l);
      attached[shed] = l[shed] + Math.max(0, u[shed] - U_bar[shed]);
      const clearing = econ.atEmployment(attached.C, attached.N);
      const xm = xi_m[shed];
      const gap = xm * gap_prev[shed] + (1 - xm) * (clearing.x(shed) - fe.x_C);
      const x_shed = fe.x_C + gap;
      const l_d = econ.demandAtWage(shed, x_shed, l[ab]).emp(shed);
      const E = Math.max(0, l[shed] - l_d);
      const Z = Math.max(0, l_d - l[shed]);
      const G = Math.max(0, over[shed]);
      const Bshort = Math.max(0, Math.log(l_star_next[ab]) - Math.log(l[ab]));

      // --- flows: quits, layoffs, openings, matching
      const q = {};
      for (const g of GROUPS) q[g] = qX[g] + (qT[g] * f_prev[g]) / f_bar[g];
      const D = { C: 0, N: 0 };
      D[shed] = Math.max(0, E - q[shed] * l[shed]);
      const v = { C: 0, N: 0 };
      v[shed] = (Math.max(0, q[shed] * l[shed] - E) + thetaH_into(shed) * Z) / pi[shed];
      v[ab] = ((q[ab] + thetaH_into(ab) * Bshort) * l[ab]) / pi[ab];
      const mu_C = mu_from("C"), mu_N = mu_from("N");
      const S = { C: u.C + mu_N * u.N, N: mu_C * u.C + u.N };
      const H = {};
      for (const g of GROUPS) H[g] = matches(ss.chi, p.iota, S[g], v[g]);
      const f = {
        C: H.C / S.C + (mu_C * H.N) / S.N,
        N: (mu_N * H.C) / S.C + H.N / S.N,
      };

      // --- record the actual economy at realized employment
      const act = econ.atEmployment(l.C, l.N);
      const x_paid = {};
      x_paid[ab] = act.x(ab);
      x_paid[shed] = Z > 0 ? act.x(shed) : x_shed;
      const dlnw = { C: x_paid.C + dlnA, N: x_paid.N + dlnA };
      const bill = { C: Math.exp(dlnw.C) * l.C, N: Math.exp(dlnw.N) * l.N };
      gap_prev[ab] = act.x(ab) - fe.x_C;
      gap_prev[shed] = gap;
      const dlnY = act.yhat + dlnA;
      push({
        t: Math.round(t * 1e6) / 1e6,
        dlnA,
        dlnY,
        yhat: act.yhat,
        x_r: act.x_r,
        dlnK: act.dlnK,
        sL: act.sL,
        dlnw_C: dlnw.C,
        dlnw_N: dlnw.N,
        mpl_C: act.x_C + dlnA,
        mpl_N: act.x_N + dlnA,
        dlnw_common: fe.x_C + dlnA,
        x_sticky: x_shed + dlnA,
        dlnw_avg: Math.log((bill.C + bill.N) / (l.C + l.N)),
        dln_labor_income: Math.log((bill.C + bill.N) / p.L_emp0),
        dln_cog_wage_bill: Math.log(bill.C / p.lC0),
        dln_other_wage_bill: Math.log(bill.N / p.lN0),
        tfp: tfpGap(p, inst, dlnA),
        lC: l.C,
        lN: l.N,
        UC: u.C,
        UN: u.N,
        lC_star: l_star.C,
        lN_star: l_star.N,
        shed_C: shed === "C" ? 1 : 0,
        l_d_shed: l_d,
        qC: q.C,
        qN: q.N,
        D_C: D.C,
        D_N: D.N,
        v_C: v.C,
        v_N: v.N,
        H_C: H.C,
        H_N: H.N,
        fC: f.C,
        fN: f.N,
        G,
        B: Bshort,
        E,
        Z,
        x_r_fe: fe.x_r,
        dlnY_fe: fe.yhat + dlnA,
        sL_fe: fe.sL,
        md_C: inst.md_C,
        md_N: inst.md_N,
        a_C: inst.a_C,
        a_N: inst.a_N,
        m_C: ptC.m,
        d_C: ptC.d,
        m_N: ptN.m,
        d_N: ptN.d,
      });

      // --- advance: the ideas step and the stock updates
      const dg = p.g * (Math.exp(p.lam * dlnY - p.one_minus_phiR * dlnA) - 1);
      dlnA += p.h * dg;
      const emp = Object.assign({}, l), pool = Object.assign({}, u);
      l = {};
      u = {};
      for (const g of GROUPS) {
        l[g] = (1 - q[g]) * emp[g] - D[g] + H[g];
        u[g] = pool[g] + q[g] * emp[g] + D[g] - f[g] * pool[g];
      }
      f_prev = f;
    }
    return out;
  }

  const api = {
    DEFAULT_PARAMS,
    M_BAR_C,
    M_BAR_N,
    T_ANCHOR,
    T_TARGET,
    AI_PRESETS,
    ROBOT_PRESETS,
    makeParams,
    makeShock,
    onOtherGroup,
    simulate,
    steadyState,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AIMacro = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
