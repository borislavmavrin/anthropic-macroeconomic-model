# anthropic-macroeconomic-model

A Python implementation of the model in Korinek, Jones, Sacher, Cotter and McCrory (2026),
*Economic Scenarios for Transformative AI*, Anthropic Institute Working Paper 2026-02 (full citation
[below](#source-paper)), calibrated with the paper's Table 1 and extended in three ways beyond the paper: the horizon runs to
2050, an AGI scenario joins the paper's three AI scenarios, and the model is generalized so that
robots can automate the all-other (physical and interpersonal) occupations too, with three
robots-only scenarios that mirror the paper's three AI scenarios and all twelve AI x robot
combinations.

## Source paper

The model, its calibration and the acceptance tests come from:

> Korinek, Anton, Charles I. Jones, Szymon Sacher, Tess Cotter, and Peter McCrory. 2026.
> "Economic Scenarios for Transformative AI." The Anthropic Institute Working Paper No. 2026-02,
> September 2026.

The authors' scenario explorer, which implements the same model through 2030, is at
<https://anthropic.com/institute/econ-scenarios>; consult The Anthropic Institute for the current
version of the paper. Copyright in the paper remains with its authors and the Institute. The code
here is an independent implementation of the equations in the paper's Appendix A with the
calibration of its Table 1, checked against the numbers the paper prints in its Tables 3, 5 and 6.
Everything beyond the paper (the horizon past 2030, the AGI scenario, robotization of all-other
work and the combined scenarios) is this repository's own extension and is not endorsed by the
paper's authors.

```bibtex
@techreport{korinek2026scenarios,
  title       = {Economic Scenarios for Transformative {AI}},
  author      = {Korinek, Anton and Jones, Charles I. and Sacher, Szymon and Cotter, Tess and McCrory, Peter},
  institution = {The Anthropic Institute},
  type        = {Working Paper},
  number      = {2026-02},
  year        = {2026},
  month       = sep,
  url         = {https://anthropic.com/institute/econ-scenarios},
}
```

## Quick start

```bash
uv sync                      # Python 3.12 virtualenv with numpy, scipy, pandas, matplotlib, pytest
uv run pytest                # 151 acceptance tests: Tables 3, 5, 6 and the Section 3.2 steady state
uv run aimacro               # simulate all 19 scenarios to 2050, print tables, write output/
```

`uv run aimacro` writes to `output/`:

| file | content |
|---|---|
| `table_2030.csv`, `table_2036.csv`, `table_2050.csv` | all 19 scenarios in the layout of the paper's Table 3 |
| `grid_<year>.csv` | GDP, average wage, labor share and unemployment for every AI x robot combination |
| `annual_<scenario>.csv` | the same rows at the start of every year 2026-2050 |
| `monthly_<scenario>.csv` | every endogenous variable of Table A.1, monthly |
| `scenarios*.png`, `labor_market*.png`, `shock_paths.png` | figures, see below |

Python API:

```python
from aimacro.calibration import DEFAULT_PARAMS
from aimacro.scenarios import SCENARIOS
from aimacro.simulate import simulate
from aimacro.report import snapshot, table

# monthly DataFrame, and a dict of its Table-3 rows at the start of 2030
df = simulate(SCENARIOS["substantial"], DEFAULT_PARAMS, t_end=2036.0)
snapshot(df, 2030.0)

# robustness variants, robots only, and a combination
simulate(SCENARIOS["extreme"].with_(eps=1.0, xi=0.9), t_end=2030.0)
simulate(SCENARIOS["robots_extreme"], t_end=2050.0)
simulate(SCENARIOS["extreme+robots_extreme"], t_end=2050.0)

# The catalog, all in aimacro.scenarios: AI_SCENARIOS (4), ROBOT_SCENARIOS (3),
# COMBINED_SCENARIOS (12), DIAGONAL_SCENARIOS (the 3 matched pairs),
# PAPER_SCENARIOS (the paper's 3) and SCENARIOS (all 19).
```

## Interactive web page

`web/index.html` puts every dial of both shocks on sliders and re-runs the model to 2050 as they
move: affected tasks and diffusion by 2030, the gain per task and its growth, the automation
share, the reinstatement ratio, the frictions for displaced workers, wage rigidity and the
capital-supply elasticity. Quick presets load the paper's scenarios, AGI and the robot ladder,
"Copy AI dials" mirrors the AI shock onto robots, and "Pin for comparison" keeps one setting on
every chart as dashed lines. Open the file in a browser; it needs no server or build step.

`web/model.js` is a JavaScript port of `calibration.py`, `paths.py`, `static.py`, `labor.py` and
`simulate.py`. It reproduces the Python simulation to about 1e-13 on the paper's scenarios, AGI,
robots-only and mixed combinations. The one exception is a matched pair (the same shock on both
groups), where the two targets fall at identical rates and rounding decides which group is
labelled as shedding; outcomes then differ by at most 0.15 percent. The Python package remains
the reference: change the model there first, then mirror the change in the port.

## What is implemented

The package follows Appendix A of the paper (Table A.1, 44 equations per month) exactly:

* `calibration.py`: every parameter of Table 1. A `Shock` holds the five scenario objects of
  Section 2.1.2 for one group of occupations plus the frictions for the workers it displaces; a
  `Scenario` is an AI shock on the cognitive group, a robot shock on the all-other group (either
  may be absent) and the capital-supply elasticity.
* `scenarios/`: one module per scenario, each defining its `SCENARIO`: the paper's three AI
  scenarios (`modest.py`, `substantial.py`, `extreme.py`), `agi.py`, and the three robots-only
  scenarios (`robots_modest.py`, `robots_substantial.py`, `robots_extreme.py`), each the paper's
  AI shock of the same name mirrored on the all-other group. The package `__init__` derives the
  twelve combinations and assembles the catalog.
* `paths.py`: logistic paths for the affected mass and diffusion through a shock's anchor and
  target values (eq. 8, 8'), a linear path for the gain per task, constant automation share and
  reinstatement ratio, for each group.
* `static.py`: the CES task economy with a surviving labor-task mass per group. `TaskEconomy`
  is system (39): `at_employment` returns, for given employment in the two groups, the wages that
  make firms demand it, the rental-rate gap that clears the capital market and output (at the
  full-mobility targets this is Proposition 1); `demand_at_wage` inverts it for the shedding
  group's labor demand at its sticky wage. The capital market is solved in the log of labor's
  residual expenditure share, which stays finite when labor's share goes to zero. `tfp_gap` is
  equation (45) summed over both groups.
* `labor.py`: the normal-times steady state (eq. 38) and the den Haan-Ramey-Watson matching function.
* `simulate.py`: the monthly loop in the nine steps of Appendix A with symmetric roles: each
  month the group whose full-mobility target is falling is the shedding group (sticky wage,
  eq. 30; layoffs, eq. 31) and the other is the absorbing group (paid its marginal product, posts
  openings for its shortfall, eq. 32); then quits, matching (eq. 33-35), stock updates
  (eq. 36-37), reporting and the ideas step (eq. 42). With an AI shock only, the cognitive group
  always sheds and the all-other group always absorbs, which is the paper's model.
* `report.py`: the rows of Table 3 at any date, plus the all-other group's employment,
  unemployment and wage bill. Growth rates are log changes over the preceding twelve months, as
  in the paper's Table 3 note.

## Acceptance tests

`tests/test_paper_predictions.py` is the only test file. Each test checks one number printed in
the paper against the simulation at the start of 2030:

* Table 3: 20 rows x 3 scenarios, plus the No-AI column.
* Table 5: capital-supply elasticity 1, 3, 6 and infinity for the substantial and extreme scenarios.
* Table 6: cognitive-wage rigidity 0, 0.5, 0.75 and 0.9 for the substantial and extreme scenarios.
* Section 3.2: the steady-state split of the unemployment pool, the filling rates and the
  matching efficiency.

A value passes if it is within one unit of the paper's last printed digit, or 1 percent of the
printed value for the largest magnitudes. All 151 pass. Table 4 (outcomes implied by the survey)
needs the respondent-level survey data, which the paper does not publish, so it is not tested.
The AGI, robot and combined scenarios have no counterpart in the paper and therefore no tests;
because the generalized model reduces exactly to the paper's when the robot shock is absent, the
151 tests also guard the extension against regressions.

Two details in the paper are ambiguous in print and were resolved by what reproduces its tables:

* The normal unemployment pool is 3.84 percent of the labor force (the CPS figure quoted in
  Section 3.2); Table 1 rounds it to 3.8. The 1.76 / 2.08 split reported on p. 21 requires 3.84.
* In the measured-TFP formula (eq. 45) the ideas term multiplies only the labor part of the
  price index, as written in Table A.1; this is what gives the extreme scenario's 13.4 percent.

Reported quantities follow the paper's conventions: the labor share is one minus the capital
share from the price index, labor income uses the cognitive wage actually paid (the sticky wage,
or the marginal product where employment trails demand), and all "percent above the no-AI path"
figures are exponentiated log gaps.

## The extended horizon: 2036 and 2050

This section covers the paper's three scenarios; the AGI scenario follows in the next one.

The paper simulates 2024-2030 and explicitly declines to look further because it does not model
robotics or physical tasks. This repository keeps the model unchanged and lets the exogenous
paths run on past 2030 under their own equations: the affected mass and diffusion saturate at
their ceilings (all cognitive work, every instance), the gain per task keeps rising linearly
(0.30 / 0.62 / 1.40 log points in 2036 and 0.30 / 1.01 / 2.80 in 2050), and the automation share
and reinstatement ratio stay constant. The results below are therefore an extrapolation of the
paper's assumptions, not a forecast endorsed by its authors. `uv run aimacro` simulates to 2050 by
default; `--end` sets any other date and `--dates` picks the years printed as tables.

Start of 2036 and start of 2050 (the paper's 2030 values are in Table 3 and in `output/table_2030.csv`):

| | modest 2036 | modest 2050 | substantial 2036 | substantial 2050 | extreme 2036 | extreme 2050 |
|---|---|---|---|---|---|---|
| GDP, pct above no-AI path | 7.0 | 29.1 | 57.2 | 133.1 | 167.6 | 392.8 |
| GDP growth, pct per year | 3.2 | 2.9 | 8.0 | 3.8 | 9.5 | 5.3 |
| Average wage, pct above no-AI path | 2.9 | 11.2 | 16.1 | 62.1 | 50.1 | 213.6 |
| Cognitive wage | 1.8 | 8.0 | -6.0 | 21.2 | -86.8 | -87.7 |
| All-other wage | 4.6 | 15.1 | 37.9 | 91.1 | 103.2 | 282.5 |
| Net return to capital, pct | 6.9 | 7.9 | 9.0 | 10.6 | 11.8 | 14.1 |
| Labor share, pct of income | 57.6 | 51.6 | 43.5 | 41.8 | 31.6 | 37.7 |
| Cognitive employment, pct since mid-2026 | -2.7 | -10.9 | -21.6 | -33.1 | -57.6 | -70.4 |
| Unemployment, cognitive workers, pct | 3.1 | 3.0 | 6.7 | 2.8 | 21.9 | 10.0 |
| Unemployment, all workers, pct | 4.0 | 3.9 | 5.4 | 3.6 | 9.7 | 5.1 |
| Measured TFP, pct above no-AI path | 3.0 | 11.7 | 20.2 | 49.2 | 57.7 | 123.1 |
| Ideas stock, pct above no-AI path | 0.4 | 3.5 | 2.8 | 16.6 | 8.9 | 39.2 |

Features of the extended paths worth knowing before quoting them:

* The transition is a hump. GDP growth peaks at 17.9 percent in early 2031 (extreme), 9.0 percent
  in late 2033 (substantial) and 3.6 percent in 2040 (modest), then falls back toward the no-AI
  rate as the logistic paths saturate. By 2050 only the linear gain path and the ideas channel
  still add to growth.
* Unemployment also peaks and recedes: 33 percent of cognitive workers and 17 percent of all
  workers in 2032-33 in the extreme scenario, 7 and 6 percent in 2034 in the substantial one.
  By 2050 all three scenarios are back near the normal 3.8 percent as the all-other occupations
  absorb the displaced workers and the sticky cognitive wage finishes adjusting.
* In the extreme scenario the ever-rising gain per task automates nearly all cognitive work, so
  the cognitive wage settles about 87 percent below its no-AI path while the economy-wide average
  wage triples relative to that path by 2050. The labor share bottoms out near 31 percent in the
  mid-2030s and recovers slightly as capital accumulates and the all-other wage rises. In the
  substantial scenario the cognitive wage falls through the mid-2030s and then recovers to
  21 percent above its no-AI path by 2050 once the group has shrunk by a third.
* The ideas channel, "relatively minor" through 2030, becomes material by 2050: the ideas stock is
  17 and 39 percent above its no-AI path in the substantial and extreme scenarios.

## An AGI scenario: 99 percent of cognitive work replaced

A fourth scenario, `agi`, goes beyond the paper. It asks what the model implies if AI replaces
99 percent of cognitive workers by the paper's scenario date. The paper's five scenario objects
are set as follows, everything else as in the extreme scenario:

| object | value | meaning |
|---|---|---|
| affected mass, 2030 | 0.621 (= sqrt(0.99) x 0.624) | AI reaches essentially all cognitive tasks |
| diffusion share, 2030 | 0.995 (= sqrt(0.99)) | and is used on essentially every instance; the product is 99 percent of cognitive instances |
| automation share | 1.0 | every AI-performed instance is done by AI alone, none augmented |
| reinstatement ratio | 0 | no new tasks for cognitive workers |
| gain per task | 0.45 at mid-2026, 2.30 (= ln 10, tenfold) in 2030, rising linearly after | AGI does the task ten times more productively |
| search discount, posting speed, wage rigidity, capital elasticity | 0.04, 0.50, 0.5, 3 | the extreme scenario's frictions |

"Replaced" is an input, the share of cognitive task instances performed by AI; how far cognitive
employment actually falls is the model's output. The paths follow the same equation 8 as the
other scenarios, which makes the logistics very steep (midpoints in 2027, most of the transition
in 2028-2029). Two numerical notes: the affected mass and diffusion split the 99 percent
symmetrically because a logistic cannot sit on its ceiling, and the surviving cognitive task mass
is computed from the logistic complements so it does not cancel to zero once both paths saturate.

| | 2030 | 2036 | 2050 |
|---|---|---|---|
| GDP, pct above no-AI path | 188 | 524 | 823 |
| GDP growth, pct per year | 30.5 | 9.0 | 3.3 |
| Average wage, pct above no-AI path | 158 | 394 | 599 |
| Cognitive wage | -93 | -100 | -100 |
| All-other wage | 187 | 397 | 599 |
| Net return to capital, pct | 11.7 | 14.7 | 16.7 |
| Labor share, pct of income | 37.6 | 44.4 | 45.0 |
| Cognitive employment, pct since mid-2026 | -88.5 | -99.0 | -100 |
| Unemployment, cognitive workers, pct | 81.6 | 91.4 | 100 |
| Unemployment, all workers, pct | 32.7 | 10.0 | 4.9 |
| Measured TFP, pct above no-AI path | 83 | 160 | 207 |
| Ideas stock, pct above no-AI path | 3.8 | 28.2 | 76.1 |

What the model says about such a world:

* The transition is brutal and fast. Economy-wide unemployment peaks at 35 percent in 2029 and
  GDP growth at 42 percent a year in the same year; cognitive employment is down 88 percent by
  2030 and 99 percent by 2036, with the displaced workers taking a decade to be absorbed by the
  other occupations (unemployment is back near 5 percent only by 2050).
* Cognitive wages go to zero, not to a low level. With tasks as gross complements, the one
  percent of cognitive instances left to humans would be valuable, but their mass shrinks toward
  nothing as the logistics saturate while workers remain attached to the group, so the marginal
  product of a cognitive worker vanishes. The cognitive group's unemployment rate reads 100
  percent by 2050 because the group itself has essentially disappeared.
* Everyone else gains enormously: the all-other wage is 2.9 times its no-AI path in 2030 and
  7 times in 2050, and average wages rise with it once the cognitive group is small.
* The gains are bounded by the weak link. Even with a tenfold gain per task in 2030 and a far
  larger one later, measured TFP is 83 percent above path in 2030 and about triple it in 2050,
  because the tasks AI does not perform (all physical and interpersonal work) cap output under
  gross complementarity. GDP is nevertheless 9 times its no-AI path by 2050.
* The labor share falls to 37 percent in 2029 and then recovers to 45 percent as the all-other
  wage rises and the capital stock catches up. The ideas channel becomes large: ideas growth
  peaks above 5 percent a year and the ideas stock is 76 percent above path in 2050.

Caveats stack on top of the paper's own: the model has no robotics, so "all other" occupations
are untouched by assumption; the capital-supply schedule and the matching function are
calibrated to normal times and are being pushed to unemployment rates with no precedent; and the
linear gain path continues past tenfold without limit (the output effect of that is small,
because of the weak link, but the per-task number itself is not meaningful by the 2040s).

## Robotization of all-other work, and combinations with AI

The paper keeps all-other occupations (manual and interpersonal work) out of AI's reach and
uses them as the sink that absorbs displaced cognitive workers. This repository lets a second
shock, robots, automate all-other tasks with the same five objects AI has (affected mass,
diffusion, gain per task, automation share, reinstatement ratio) and the same frictions for the
displaced. The production side generalizes directly: each group has a surviving mass of labor
tasks, capital takes over the automated instances of both, and the full-mobility targets split
employment in proportion to the two surviving masses. The labor market needs one new rule, since
either group may now shrink: each month the group whose target is falling sheds workers under a
sticky wage with layoffs, and the other absorbs them. With AI alone this is exactly the paper.

### The three robots-only scenarios

The paper gives no robot parameters, so each robot scenario is the paper's AI scenario of the
same name applied to all-other work, modeled in exactly the same way: the same five objects, the
same mid-2026 anchors and the same 2030 scenario date, with the same frictions for the displaced.
The only adjustment is to the affected mass, which the paper states in units of the economy's
labor tasks: it is rescaled by the ratio of the two groups' sizes (0.376 / 0.624) so that robots
reach the same share of all-other work that AI reaches of cognitive work. Without the rescaling
the extreme scenario's mass of 0.5 would exceed the whole all-other group.

| object | robots_modest | robots_substantial | robots_extreme |
|---|---|---|---|
| affected mass, 2030 (share of all-other work) | 0.121 (0.32) | 0.181 (0.48) | 0.301 (0.80) |
| affected mass, mid-2026 anchor | 0.084 (0.22) | 0.084 (0.22) | 0.084 (0.22) |
| diffusion share, 2030 (mid-2026 anchor 0.10) | 0.2 | 0.4 | 0.6 |
| gain per task, mid-2026 (log points) and slope per year | 0.30, 0 | 0.35, 0.028 | 0.45, 0.10 |
| automation share | 0.50 | 0.75 | 0.90 |
| reinstatement ratio | 0.50 | 0.25 | 0 |
| search discount of displaced workers | 0.17 | 0.08 | 0.04 |
| posting speed of the hiring group | 0.10 | 0.25 | 0.50 |

The labor supply is fixed, as in the paper: there is no reservation wage, so workers stay
attached to the market however low their wage. Robots-only results at the start of 2030, 2036
and 2050:

| | modest 2030 / 2036 / 2050 | substantial | extreme |
|---|---|---|---|
| GDP, pct above no-AI path | 1.0 / 4.2 / 16.5 | 5.0 / 29.4 / 60.0 | 16.7 / 65.1 / 125.0 |
| Average wage | 0.4 / 1.8 / 6.6 | 1.4 / 9.6 / 31.4 | 6.1 / 25.9 / 80.9 |
| Cognitive wage | 0.7 / 2.9 / 8.8 | 3.6 / 20.2 / 42.2 | 17.6 / 45.0 / 95.4 |
| All-other wage | -0.1 / -0.2 / 1.8 | -2.6 / -19.7 / -10.4 | -22.0 / -92.8 / -93.7 |
| Labor share, pct of income | 59.7 / 58.5 / 55.0 | 57.7 / 50.5 / 49.8 | 51.4 / 44.6 / 48.6 |
| All-other employment, pct since mid-2026 | -0.7 / -4.1 / -16.5 | -5.4 / -29.4 / -44.5 | -25.7 / -63.8 / -79.3 |
| Unemployment, all-other workers, pct | 5.6 / 5.9 / 5.6 | 7.4 / 10.1 / 5.2 | 22.3 / 26.5 / 14.7 |
| Unemployment, all workers, pct | 3.9 / 3.9 / 3.6 | 4.2 / 4.4 / 2.8 | 8.9 / 6.3 / 3.1 |
| Measured TFP, pct above no-AI path | 0.4 / 1.8 / 6.8 | 1.8 / 11.5 / 26.1 | 7.8 / 29.7 / 56.9 |

The robots-only picture is the paper's picture with the groups swapped. Displaced all-other
workers move into cognitive occupations, whose employment is up 10 to 48 percent by 2050 and
whose wage rises; the all-other wage falls, slightly and then recovers in the modest case, and
to almost nothing in the extreme case. Economy-wide unemployment peaks at the same time as under
the matching AI scenario but lower (3.9 percent in 2035, 4.8 in 2033 and 11.4 in 2032 against
4.0, 5.9 and 16.8 for the AI ladder), because the cognitive group is the larger of the two, so
it absorbs the displaced more easily, and because the same share of a smaller group is a smaller
mass of workers. GDP gains are smaller than AI's at every rung for the same reason: all-other
work is 38 percent of the wage bill against 62 for cognitive work.

### AI and robots combined

All twelve combinations are simulated; the figures show the three matched pairs. Frictions are
per direction: displaced cognitive workers search under the AI shock's discount and are hired at
its posting speed, displaced all-other workers under the robot shock's. Matched pairs at the
start of 2030, 2036 and 2050:

| | modest pair 2030 / 2036 / 2050 | substantial pair | extreme pair |
|---|---|---|---|
| GDP, pct above no-AI path | 2.7 / 12 / 54 | 15 / 132 / 417 | 88 / 836 / 5692 |
| Average wage | 1.1 / 4.6 / 18 | 2.9 / 21 / 123 | 6.2 / -76 / -23 |
| Labor share, pct of income | 59.1 / 56.1 / 46.1 | 53.6 / 31.3 / 25.8 | 33.9 / 1.5 / 0.8 |
| Net return to capital, pct | 6.6 / 7.1 / 8.8 | 7.4 / 11.3 / 15.2 | 10.3 / 20.2 / 34.8 |
| Unemployment, all workers, pct | 3.8 / 3.8 / 3.8 | 3.9 / 3.9 / 3.9 | 3.9 / 3.9 / 3.9 |
| Measured TFP, pct above no-AI path | 1.2 / 4.9 / 20 | 5.0 / 36 / 99 | 23 / 128 / 315 |

The full 4 x 3 grid at the start of 2050 (robots across, AI down; the `none` row and column are
the single-shock scenarios):

GDP, percent above the no-AI path

| AI \ robots | none | modest | substantial | extreme |
|---|---|---|---|---|
| none | | 16 | 60 | 125 |
| modest | 29 | 54 | 123 | 243 |
| substantial | 133 | 196 | 417 | 947 |
| extreme | 393 | 634 | 1781 | 5692 |
| agi | 823 | 1413 | 4989 | 22988 |

Labor share, percent of income

| AI \ robots | none | modest | substantial | extreme |
|---|---|---|---|---|
| none | | 55.0 | 49.8 | 48.6 |
| modest | 51.6 | 46.1 | 39.8 | 37.4 |
| substantial | 41.8 | 35.0 | 25.8 | 19.6 |
| extreme | 37.7 | 28.6 | 13.8 | 0.8 |
| agi | 45.0 | 35.1 | 17.6 | 0.3 |

What the combinations show:

* **A matched pair displaces nobody.** Because the robot shock is the AI shock mirrored exactly,
  the two groups' surviving task masses shrink in the same proportion at every date, so the
  full-mobility targets of equation 13 never move: employment stays where it is, unemployment
  stays at its normal 3.8 to 3.9 percent throughout, and the two groups' wages move together.
  The matched pair is the paper's single-group mechanism applied to the whole economy, with no
  labor reallocation at all. Off the diagonal the group hit by the larger shock sheds workers
  much as it does alone: the extreme AI + modest robots pair peaks at 16.7 percent unemployment
  in 2032 against 16.8 for extreme AI alone.
* **The two shocks are far more than additive in output.** With AI alone the tasks robots would
  do are the weak link that caps output under gross complementarity, and vice versa. Hit both
  groups and the cap lifts: the extreme pair's 2050 GDP gain is 5692 percent against 393 for AI
  alone and 125 for robots alone. The same mechanism is why measured TFP, bounded at about triple
  the no-AI level in any single-shock scenario, reaches four times it in the extreme pair.
* **Labor's share can go to zero.** Once nearly all tasks in both groups are automated, the
  surviving human tasks shrink toward nothing while every worker still has to be employed in
  them (fixed labor supply, no reservation wage). Their marginal product falls, and all income
  accrues to capital: the extreme pair's labor share is 1.5 percent in 2036 and 0.8 in 2050, and
  wages are 76 percent below their no-AI path in 2036 and still 23 percent below in 2050 despite
  GDP being 58 times higher. This is the model's answer under the choices made here, not a
  forecast.
* **Mid-ladder pairs look like a bigger AI scenario.** The substantial pair reaches in 2050 a
  GDP gain slightly above extreme AI alone (417 against 393 percent), with a lower labor share
  (26 against 38 percent) and no unemployment episode at all against a peak of 16.8 percent.

Caveats, on top of the paper's own and the AGI section's: the robot ladder is the AI ladder
mirrored, not an estimate, and in particular robots reach their scenario values in 2030 like AI
rather than lagging it; the frictions for all-other workers moving into cognitive occupations are
taken from the AI ladder although the retraining involved is likely harder; and the model has no
mechanism (reservation wage, demand feedback, redistribution) that would stop a wage from falling
while output multiplies.

## Follow-up work: grounding the robot shock in evidence

The robot ladder above is an assumption mirrored from the AI ladder. Anthropic's Economic
Research note ["What work can robots do?"](https://www.anthropic.com/research/what-work-can-robots-do)
(Russell Legate-Yang and Maxim Massenkoff, September 30, 2026) measures exactly the objects the robot
shock needs, task by task across about 900 O*NET occupations, and should replace the mirrored
values. What it provides, and what each piece would change in this repository:

* **The affected mass and its ceiling.** Physical work is 46 percent of US working hours, and
  robots demonstrated today can do three-quarters of physical tasks, 34 percent of all hours,
  though mostly in purpose-built or structured settings (23 percent of hours in factory-type
  environments, 10 percent in structured facilities such as warehouses, 1 percent in unstructured
  ones such as city roads; 12 percent of hours robots cannot do at all). In the model's units this
  pins down the robot shock's ceiling on the affected mass well below "all of all-other work": the
  interpersonal and dexterity-heavy parts of the group (nursing, general repair, personal care,
  hairdressing, scaffolding) are the least exposed. Making the ceiling a parameter of the shock,
  set from the exposure share rather than the group's whole wage bill, is the first change.
* **The diffusion path and its timing.** The note separates capability from cost: robots are
  cost-competitive for only 0.3 percent of job tasks today; a 20 percent cost decline would make
  them competitive for 2.8 million workers, which at the historical 3 percent a year takes about
  seven years; half of physical work becomes cost-competitive by 2085 at baseline and by 2050 in
  the fast scenario, and automating 90 percent of physical work takes 53 years even then. These
  timelines are far slower than the robot ladder here, whose extreme rung has robots performing
  48 percent of all-other task instances by 2030 (80 percent of tasks affected, 60 percent
  diffusion) and whose modest rung (6 percent of instances) is closest to the note's fast
  scenario. The diffusion logistic should be re-anchored to the note's cost-parity path, or
  replaced by a cost-driven rule: an instance switches to the robot when its annual robot cost,
  falling at a chosen rate, drops below its labor cost, which the note estimates per task.
* **The gain per task and the automation share.** The note's cost estimates (for example about
  45,000 dollars a year for a packing robot against 49,000 in labor; welding robots at about five
  times the cost of a welder) give the unit-cost decline per robot-performed instance directly,
  which is what the gain parameter means. Its finding that robots replace rather than assist
  physical tasks argues for automation shares above the AI ladder's 0.5 to 0.9 used here.
* **Who is displaced, and where they can go.** Highly exposed workers are 20 percentage points
  less likely to be female, 55 points less likely to hold a bachelor's degree and earn about 30
  dollars an hour less, and nine of the ten most exposed occupations are vehicle operators. That
  argues for a search discount for displaced all-other workers moving into cognitive work that is
  lower, not equal, to the AI ladder's, and for splitting the all-other group into a manual part
  robots reach and an interpersonal part they largely do not, with frictions between them: the
  third-island extension discussed above.
* **Barriers the model lacks.** Capabilities block about 70 percent of physical tasks (manipulation
  alone half of them), regulation about 14 percent (concentrated in healthcare, protective service
  and education), and human preferences about a quarter. The paper's model has only a diffusion
  share to stand in for all of these; a regulation or preference ceiling per occupation group
  would make the robot scenarios' upper rungs explicit assumptions about those barriers falling.

The note's own caveat applies here too: cost parity does not by itself imply high unemployment or
rapid growth, because bottleneck tasks remain. That is the weak-link mechanism this model already
has, which is why the single-shock scenarios' gains are bounded and why grounding the robot shock
is the step that would make the combined scenarios more than an illustration.

## Figures

`uv run aimacro` writes eight figures to `output/` (the PNGs are kept in git; the CSVs next to
them are not). Colours follow the rung of the ladder: blue modest, green substantial, red extreme,
purple AGI; dashed grey lines are the economy without AI.

### The paper's horizon, 2025-2030

The paper's three scenarios only: the panels of its Figures 2-4 plus labor income, capital income,
the ideas stock and the all-other group, each labelled with its January 2030 value. These are the
numbers the acceptance tests check.

![The three scenarios, 2025-2030](output/scenarios_2025_2030.png)

### The AI scenarios to 2050

The same panels for the four AI scenarios. The vertical line marks the start of 2030, where the
paper stops; everything to its right is the extrapolation described above.

![The AI scenarios, 2025-2050](output/scenarios.png)

### The robots-only scenarios to 2050

![The robots-only scenarios, 2025-2050](output/scenarios_robots.png)

### AI and robots combined, matched rungs, to 2050

The extreme pair sets the scale of most panels; the other two pairs are best read in
`output/table_2036.csv` and `output/table_2050.csv`.

![The matched pairs, 2025-2050](output/scenarios_combined.png)

### The exogenous shock paths

The objects each shock is made of, AI on cognitive work in the top row and robots on all-other
work in the bottom row: the affected mass and the diffusion share follow logistics through the
mid-2026 anchor and the scenario date, 2030 for both, then saturate at their
ceilings; their product is the share of all task instances the technology performs; the gain per
instance rises linearly without a ceiling.

![Exogenous shock paths](output/shock_paths.png)

### The labor-market block

One row per scenario: employment in the two groups against the full-mobility targets of
equation 13 (dashed); the unemployment pools by group of origin against their normal levels;
job-finding rates by origin; and the monthly flows of layoffs in each group, hires and quits.
There is one figure per family: `labor_market.png` (AI), `labor_market_robots.png` (robots only)
and `labor_market_combined.png` (matched pairs).

![Labor market flows, AI scenarios](output/labor_market.png)

![Labor market flows, matched pairs](output/labor_market_combined.png)
