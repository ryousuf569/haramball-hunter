# Defender calibration

Everything in `defenders/` that has a number in it got that number from one of the
studies below. This file records what was calibrated, how, why, and what came out.

Two different kinds of calibration live here, and they are worth keeping apart:

1. **Data fits.** Parameters measured off real StatsBomb freeze frames. These
   answer "what do real low blocks actually do". Covered in sections 1 to 6.
2. **Sim sweeps.** Parameters with no direct real-world counterpart, tuned by
   running the sim itself across a grid and looking at the outcome. These answer
   "what value makes the model behave". Covered in sections 7 and 8.

A note on coordinates. StatsBomb uses a 120x80 pitch with the attacking team
going left to right, so defenders defend x=0. The sim uses 105x68 with defenders
defending x=105. Everything in this folder is in the **calibration frame**
(105x68, defenders at x=0) unless a section says otherwise. `defenders.py` mirrors
with `PITCH_X - x` at the boundary, and `calibrate_defender_formation.py` mirrors
its inputs up front because it feeds the sim directly.

---

## 1. The frame dataset

**How.** `calibration_graphing.py` pulls StatsBomb competition 55, season 282, and
takes the first 8 matches by `match_id`. For every event that has a freeze frame
attached it keeps the frame only if the event team is also the possession team, so
the "defenders" in the frame are genuinely the team out of possession. Positions
are scaled from 120x80 to 105x68.

**Why these filters.** We want settled low blocks, not transitions and not
counter-attacks. Four filters do that work: at least 8 non-keeper defenders
visible (otherwise the line structure is guesswork), the ball inside 70m
(`BALL_X_MAX`), the possession at least 5 seconds old (`MIN_POSSESSION_SECONDS`,
which drops the chaotic first seconds after a turnover), and the defensive mean x
inside 52.5m (`BLOCK_DEPTH_MAX`, which drops high presses). At least 3 attackers
must be visible so an attacking line can be defined at all.

**Results.** 2032 frames across 8 matches, and 36054 player rows (10160 back-line,
7604 midfield, 18290 attacker). The event mix is what you would expect from
settled possession: 670 ball receipts, 617 passes, 568 carries. Within each frame
the 5 deepest non-keeper defenders are called the back line and the next 4 the
midfield line. The attacking line is the mean x of the 4 most advanced attackers.
Frames go to `low_block_frames.csv`, players to `low_block_defenders.csv`, and
summary statistics to `low_block_summary_stats.csv`.

---

## 2. Block depth

**How.** Least squares on the frame table, predicting `back_line_x` from ball
position and attacker position. Two models were fitted: ball only, and ball plus
attacking line.

**Why.** `defenders.py` needs a rule that says how deep to sit given what it can
see. The obvious candidate is ball position, since that is the usual hand-wavy
description of a low block. The question was whether ball position alone is
actually enough.

**Results.** It is not.

| model | fit | R2 |
| --- | --- | --- |
| ball only | `back_line_x = 0.2037 * ball_x + 29.57` | 0.157 |
| ball + att | `back_line_x = -0.1067 * ball_x + 0.8144 * att_line_x + 16.070` | 0.666 |

Ball position alone explains 16 percent of the variance. Adding the attacking line
takes that to 67 percent, in-sample RMSE 3.90m. The block tracks the attackers,
not the ball.

**On the ball_x coefficient.** It is worth being explicit that the `ball_x` term is
not interpretable. Its sign flips between the two models, from +0.204 on its own to
-0.107 once `att_line_x` is present. That is textbook multicollinearity:
`corr(ball_x, att_line_x) = 0.646`, so the two predictors are largely carrying the
same information and the split between them is unstable. The marginal correlations
tell the real story, `corr(att_line_x, back_line_x) = 0.801` against
`corr(ball_x, back_line_x) = 0.397`. We keep `ball_x` in the model because it
improves prediction and because the sim has the value available for free, but the
negative coefficient should be read as a small correction term and not as
"defenders drop when the ball advances". Section 3 checks whether the fit as a
whole survives on unseen matches, which is the claim we actually care about.

Coefficients go to `depth_fit_coefficients.csv`. `defenders.py` uses them in
`calculate_depth_ref` with two modifications: inputs are clipped to the 5th/95th
percentile ranges in `input_ranges_q05_q95.csv` (att_line 16.09 to 42.25, ball_x
13.26 to 57.48) so extrapolation cannot run away, and the inputs are read from the
back of a 5-tick history deque so the block reacts with a lag instead of teleporting.
`HARAM_DEPTH_OFFSET = 15.0` is then added on top, which is a deliberate design
choice for the haramball scenario and not something the data supports.

---

## 3. Held-out validation of the depth fit

**How.** `validate_depth_fit.py` refits the same two models on a subset of matches
and scores them on matches the fit never saw. Splitting is by `match_id`, never by
frame. Consecutive freeze frames inside one possession are near-duplicates of each
other, so a frame-level split would put the same possession on both sides and
report an optimistically low error. Two schemes are run: a fixed split with 6
training matches and 2 held-out matches, and leave-one-match-out over all 8. The
baseline predicts the **training** mean, so it is held out on the same terms as the
model. Skill is `1 - (RMSE / baseline RMSE)^2`.

**Why.** The R2 and RMSE in section 2 are in-sample. They describe how well the
model fits the frames it was fitted on, which is not the same as evidence it
predicts anything. Without this section "calibrated from real data" is an
unchecked claim.

**Results.** Leave-one-match-out, pooled across folds, all figures in metres:

| target | model | RMSE | baseline | skill |
| --- | --- | --- | --- | --- |
| back_line_x | ball only | 6.324 | 6.854 | 0.149 |
| back_line_x | ball + att | 4.032 | 6.854 | 0.654 |
| mid_line_x | ball only | 5.818 | 6.043 | 0.073 |
| mid_line_x | ball + att | 4.165 | 6.043 | 0.525 |

The fixed 6/2 split agrees closely, 3.956m and 4.116m for the two-predictor model.

Three things come out of this. First, the fit generalises: in-sample RMSE was
3.905m for the back line and 4.111m for the midfield line, held-out is 4.032m and
4.165m. A two-predictor linear model on 2032 frames is not overfitting, and now
that is a measurement rather than an assertion. Second, per-fold RMSE is tight,
3.66m to 4.45m across all 8 folds, so no single match is carrying the fit. Third,
ball position alone is close to worthless out of sample, skill 0.149 and 0.073, and
on the fixed split it is actually negative for the back line at -0.019, meaning it
does worse than predicting the mean. That is the honest complement to the
multicollinearity note above. `ball_x` is not just an unstable coefficient, it
carries almost no independent signal about block depth.

Output goes to `depth_fit_holdout.csv`. The script reads the committed
`low_block_frames.csv`, so it runs without StatsBomb API access.

---

## 4. Lateral shift

**How.** Regress the back-line and midfield y centroids on ball y, both centred on
the pitch midline at 34m, and read off the slope. The slope is the gain: 0 means
the line ignores the ball laterally, 1 means it slides across with the ball
one-for-one.

**Why.** A real block shifts sideways toward the ball but does not follow it fully,
because the far-side players hold width to cover a switch. The gain measures where
in that range real teams sit.

**Results.** Back line gain 0.41, midfield gain 0.29. Restricting to central ball
positions between y=10 and y=58 barely moves it, 0.43 and 0.31, so the number is
not being driven by touchline events. The midfield line shifting less than the
back line is the interesting part, and consistent with the midfield holding
horizontal compactness while the back line tracks across.

Note that `defenders.py` currently applies gain 0.19 to the back line and 0.31 to
the midfield. The midfield value matches. The back-line value does not reproduce
from this dataset and is roughly half the fitted gain. See "Open gaps" below.

---

## 5. Line separation, width, and spacing

**How.** Straight descriptive statistics off `low_block_frames.csv`, written to
`low_block_summary_stats.csv` and `input_ranges_q05_q95.csv`.

**Why.** These are the shape constants `defenders.py` needs: how far apart the two
lines sit, how wide each line stretches, and what player-to-player gaps look like.
The compactness snapback in `apply_compactness_snapback` needs a low and high gap
band, and picking those from intuition is exactly the kind of thing that gets
questioned.

**Results.**

| quantity | mean | sd | q05 | q95 |
| --- | --- | --- | --- | --- |
| line_sep (mid minus back) | 15.58 | 3.19 | 10.74 | 21.24 |
| back_width | 30.15 | 7.53 | 16.89 | 41.60 |
| mid_width | 29.75 | 8.24 | 15.29 | 42.01 |
| back_gap_min | 2.56 | 1.88 | 0.19 | 6.23 |
| back_gap_max | 13.39 | 4.41 | 7.23 | 21.40 |

The snapback band in `defenders.py` is `lo=2.5, hi=16.0`, which lines up: 2.5m is
essentially the mean minimum gap observed, and 16m sits just above the mean maximum
gap, so the correction fires on genuinely stretched shapes rather than on normal
ones. The two lines are close to the same width, around 30m each, which is well
inside the 68m pitch and much narrower than the attacking line at 48.8m mean width.

Two constants in `defenders.py` do not match these numbers. `LINE_SEP = 10` against
a fitted 15.58, and the hardcoded backline offsets span y=-20 to y=+20, a fixed 40m
width, against a fitted mean of 30.15m. 40m is roughly the 94th percentile of
observed back-line width. See "Open gaps" below.

---

## 6. Non-unison evidence and per-slot stagger

**How.** Two measurements. First, the standard deviation of x within each line per
frame (`back_x_scatter`, `mid_x_scatter`). A perfectly rigid line that moves in
unison would give exactly 0. Second, each player's x offset from their own line
mean, averaged by slot after sorting the line by y, written to
`per_slot_stagger.csv`.

**Why.** The naive model is a rigid horizontal line that slides forward and back as
one unit. If that were right, the within-line scatter would be near zero and every
slot would have the same offset. This section tests that, and it fails.

**Results.** Within-line x scatter averages 5.44m for the back line and 3.71m for
the midfield, with maxima above 10m. Real lines are not rigid, and the back line is
noticeably less rigid than the midfield.

The per-slot stagger has a clear shape. Numbering slots 0 to 4 across the back line
by y, the mean offsets from the line mean are +1.77, -0.93, -1.42, -0.80, +1.38 in
the calibration frame. The two wide defenders sit about 1.5m deeper than the line
mean and the central defenders about 1m higher, which is the familiar slight U of a
back line tucking its full-backs in. The midfield shows the same pattern at about a
third of the amplitude: +0.47, -0.18, -0.44, +0.20. `defenders.py` carries these
directly as `backline_offset` and `midline_offset`, sign-flipped because the sim
mirrors x.

---

## 7. Ground duel

**How.** The geometry is transferred from the RoboCup 2D soccer server's tackle
model rather than fitted. RoboCup uses `tackle_dist=2.0`, `tackle_width=1.25`,
`tackle_exponent=6`, `tackle_back_dist=0`. Its pitch and timestep are a coarser sim
than ours, so the transferable parts are the aspect ratio (2.0/1.25 = 1.6) and the
super-ellipse exponent, not the absolute sizes. Those scale to `DUEL_A = 1.20`,
`DUEL_B = 0.50` (about A/1.6), and `DUEL_A_BACK = 0.40` for reduced reach behind
the defender. RoboCup rolls once per command; we test every tick, so the roll is
converted to a hazard rate and `p_tick = 1 - exp(-lambda * dt)`, which makes the
outcome independent of timestep.

**Why.** There is no freeze-frame data for tackle outcomes, so this parameter
cannot be fitted the way sections 2 to 6 were. Borrowing a published model with a
stated calibration is more defensible than inventing a number. `LAM_MAX = 2.1`
comes from a target: a perfectly positioned defender should win the ball with
probability 0.65 over half a second, so `lambda = -ln(1 - 0.65) / 0.5 = 2.1`.

**Results.** `duel_calibration_summary.csv` records a sim measurement of how often
the duel geometry actually engages: over 15 episodes of 2000 ticks, the ball was
held for 8940 ticks and a defender was inside the duel ellipse for only 85 of them,
0.95 percent of held ticks, spread over 25 separate contact runs. Contacts are
short: mean 0.34s, median 0.4s, p90 0.6s, max 0.6s. The nearest defender is a median
16.9m from the ball, dropping to 2.12m at the 5th percentile.

That short contact duration is the finding that matters. The 0.65-in-0.5s target
assumed a half-second window, but the median real window is 0.4s and the maximum
observed is 0.6s, so the fitted `LAM_MAX` that would hit 0.65 over the windows we
actually see is 3.09 mean / 2.62 median rather than 2.1. At the current 2.1 the
implied win rate is 0.51, not 0.65. The value was left at 2.1 rather than raised.
The observed turnover rate of 0.667 per episode was already in a reasonable range,
and 0.51 per genuine contact is defensible on its own terms, but the gap between
the stated target and the delivered rate should be stated rather than glossed.

Caveat: the script that produced these two duel CSVs was not committed, only its
outputs. The numbers cannot currently be regenerated from the repo.

---

## 8. Interception threshold

**How.** `intercept_calibration.py` sweeps `INTERCEPT_P_MIN` over 0.50, 0.60, 0.70,
0.80, 0.90, 0.95 and 1.0, running 52 episodes at each value (13 seeds x 4 starting
holders) for up to 1200 ticks. 1.0 is a control: it is unreachable by construction,
so no interception can ever fire and the run isolates what everything else in the
model does on its own.

Cause attribution is done by wrapping `turnover.ground_duel` and
`turnover.intercept_pass` and patching them into the environment module's globals,
rather than inferring the cause from whether the previous ball state was
`in_flight`. That inference merges offsides into the duel bucket and mislabels a
duel that happens to land on a flight tick. Wrapping also lets us log the actual
control probabilities reached on ticks where a defender was geometrically in range,
which is what justifies choosing a threshold instead of just counting outcomes.

**Why.** Unlike the duel, there is no RoboCup interception model to copy. The
threshold decides when a defender close enough to a moving ball is deemed to have
won it, and it has no real-world reading, so it has to be picked by behaviour.

**Results.**

| threshold | intercept | duel | timeout | intercepts/pass | median episode |
| --- | --- | --- | --- | --- | --- |
| 0.50 | 29 | 23 | 0 | 0.0505 | 44.1s |
| 0.60 | 29 | 23 | 0 | 0.0505 | 44.1s |
| 0.70 | 29 | 23 | 0 | 0.0505 | 44.1s |
| 0.80 | 16 | 32 | 4 | 0.0206 | 64.0s |
| 0.90 | 0 | 34 | 18 | 0.0 | 42.3s |
| 0.95 | 0 | 34 | 18 | 0.0 | 42.3s |
| 1.00 | 0 | 34 | 18 | 0.0 | 42.3s |

Two plateaus and one working point. Everything at or below 0.70 gives identical
results, so those thresholds are not binding at all: any interception that fires at
0.70 would also have fired at 0.50. Everything at or above 0.90 is identical to the
1.0 control, so the threshold never fires and interception is effectively disabled.

The control run explains why. The threshold only accepts or rejects, it does not
change the probabilities, which come from the physics. Sampling the best in-range
control probability per episode at threshold 1.0 gives a median of 0.816, a p90 of
0.824, and a maximum of 0.858, and only 29 of 52 episodes ever had an in-range tick
at all. So any threshold above about 0.86 is unreachable by construction, no matter
what the outcome table shows. 0.80 was chosen as the highest threshold that still
fires, sitting below the achievable ceiling, and it is also the only setting that
splits outcomes between interceptions and duels rather than letting one dominate.
It is set in `turnover.py` as `INTERCEPT_P_MIN = 0.80`.

**Superseded in part.** This sweep was run with the geometric gate at RoboCup's
kickable area, 1.085m, and with the pre-section-10 press. Both have since changed:
the gate is now `INTERCEPT_REACH = 2.5`, a closing distance rather than a reaching
one, and the press of section 10 puts defenders much nearer the passing lanes.
Spot-measured over 40 episodes against the 500k checkpoint, a defender is now
geometrically in range on 30.3% of flight ticks rather than 11.7%, and the number
of ticks clearing 0.80 roughly doubles. The achievable-p ceiling that motivated the
choice of 0.80 has moved with it — the maximum observed is now 0.992, not 0.858 —
so 0.80 is no longer near the ceiling and is no longer the highest threshold that
fires. **The sweep should be re-run before `INTERCEPT_P_MIN` is trusted again.**
End to end the effect is currently small (4 interceptions to 5 over those 40
episodes) because duels and offsides dominate the turnover mix, but that is a
statement about one checkpoint, not about the parameter.

---

## 9. Formation sampler

**How.** `calibrate_defender_formation.py` fits a generative model of the whole
5-4-1 shape, so episodes can start from a randomly drawn but realistic block
instead of one fixed pose. It reads the same two CSVs, mirrors them into sim
coordinates (defenders now defend x=105), and keeps only frames with a complete
back five and midfield four, which leaves 1508 frames.

The shape is decomposed in two stages. Per frame it records a centroid (cx, cy) and
a spread (sx, sy). Each player's position is then standardised into
`u = (x - cx) / sx` and `v = (y - cy) / sy`, and the mean and sd of u and v are
taken per slot. To sample, draw a centroid and a spread, then draw each slot's
standardised offset and rescale. The two spreads are drawn through a shared factor
with correlation 0.198 so that a compressed shape compresses in both directions at
once instead of becoming implausibly narrow-but-deep.

**Why.** Fixed starting positions make every episode the same episode. Sampling
independent per-player Gaussians instead would produce shapes no real team has ever
held, because it throws away the fact that all 9 players move together. The
two-stage decomposition keeps the frame-level correlation and only randomises what
genuinely varies.

**Results.** Frame parameters, in sim coordinates, written to
`defender_frame_params.csv`:

| parameter | mean | sd | q05 | q95 |
| --- | --- | --- | --- | --- |
| centroid_x | 60.65 | 5.26 | 53.94 | 69.96 |
| centroid_y | 34.04 | 9.26 | 18.83 | 49.24 |
| spread_x | 9.86 | 1.84 | 7.23 | 13.23 |
| spread_y | 12.45 | 2.14 | 8.73 | 15.74 |
| attline_gap | -13.79 | 4.94 | -22.43 | -5.29 |

Marginals are Gaussian clipped to the 5th/95th range, so a draw cannot land
somewhere no real frame did. `attline_gap` is negative because the attacking front
line sits goalward of the block centroid. The sampler can anchor either on absolute
depth (`centroid_x`) or on that gap relative to a given attacking line.

Per-slot offsets are in `defender_slot_params.csv`, and reproduce the same U shape
as section 6 in standardised units: back-line `mu_u` runs 0.56, 0.80, 0.87, 0.80,
0.54 and midfield `mu_u` runs about -0.91 to -0.84. Slot 9, the lone forward, has no
freeze-frame data behind it because the source only labelled the 9 deepest
defenders, so it gets `defenders.py`'s resting x of 52.0 plus borrowed midfield
spread, and takes no haram depth offset. This is an assumption, not a measurement.
After drawing, each line is re-sorted by y, because an unsorted draw can put two
slots on the wrong side of each other.

The script self-validates by sampling as many shapes as there were real frames and
printing real against sampled mean and sd for centroid x, centroid y, deepest x and
width. Figures go to `defender_shape_fit.png`, `defender_slot_scatter.png` and
`defender_samples.png`.

---

## 10. Press policy

**How.** `press_calibration.py`, using two sources that are not interchangeable.

The **geometry** comes from the 2032 StatsBomb low-block freeze frames of section
1 — the right population by construction, since they are already filtered to
settled low blocks. For each frame it computes the distance from the ball to the
nearest non-keeper block defender (`d1`), how many are within 5m (`n_near`), which
line that nearest defender belongs to, and the largest y-gap in the back line.

The **dynamics** — how long one commitment lasts, and how much ground the presser
covers during it — cannot be measured from freeze frames at all, because a freeze
frame is one instant. Those come from Metrica's `Sample_Game_1` tracking (25fps,
`physics/validation/data/`), restricted to frames whose block depth falls inside
the band the StatsBomb low blocks occupy (back line at sim x >= 59.2, the
StatsBomb q05). **A full match is mostly not a low block**, and unfiltered this
source measures mid-block and counter-pressing instead, which behave differently.
The two sources overlap on `d1`, and the third panel of `press_commitment.png`
plots both profiles so the subset can be checked before its durations are trusted.
Treat everything from Metrica as order-of-magnitude: one match, not eight.

**Why.** The press in `defenders.py` sent whoever was closest to the ball at it,
at full speed, from anywhere on the pitch. Against a learned attacker that
recycles the ball every tick, that produced one midfielder ping-ponging across
the block and a hole where they had been standing. The question was what a real
low block does instead.

**Results.**

*Absolute ball x is the wrong variable, and it inverts.* Profiling engagement
against absolute ball x says low blocks press **less** the closer the ball gets to
their own box, which is backwards. The reason is a selection effect: the frames
with the ball deepest are overwhelmingly balls that have already been played
**through** the block — median 19m goal-side of the back line once the ball is past
x=90 — so of course nobody is near it. The right conditioning variable is the
ball's position **relative to the block's own lines**, which `defenders.py` already
computes every tick. The third panel of `press_engagement.png` shows the trap.

*There is a press band.* Measured against the back line and signed so positive is
goal-side, commitment is the norm from **-20m to +5m** (the front edge is 4.7m in
front of the midfield line, given the measured 15.34m line separation):

| ball vs back line | frames | nearest defender | committed | defenders within 5m |
| --- | --- | --- | --- | --- |
| -20 to -15m | 73 | 1.86m | 75% | 1.01 |
| -15 to -10m | 149 | 2.01m | 79% | 1.08 |
| -10 to -5m | 236 | 2.30m | 80% | 1.08 |
| -5 to 0m | 304 | 3.33m | 71% | 1.00 |
| 0 to +5m | 387 | 4.29m | 58% | 0.77 |
| +5 to +10m | 285 | 5.88m | 39% | 0.41 |
| +10 to +20m | 400 | 9.75m | 14% | 0.14 |
| +20 to +40m | 96 | 16.07m | 1% | 0.01 |

Inside the band a defender is within 5m of the ball on 70% of frames and the
nearest one is 3.19m away; outside it, 21% and 8.97m. The band covers 57% of
low-block possession frames.

*Exactly one defender commits.* Inside the band, 49% of frames have exactly one
defender within 5m, 30% have none and 21% have two or more, for a mean of 0.95.
Committing the whole time, or committing two, is not what real blocks do.

*It costs no shape.* The largest y-gap in the back line is 13.27m when nobody is
committed and 13.30m when someone is — statistically identical. A real press does
not open the block up, which is precisely the failure mode being fixed.

*The press comes from the line the ball is in.* Inside the band the nearest
defender is a back-line player on 72% of frames and a midfielder on 28%, and which
one it is tracks the ball across the band (`press_shape.png`).

*Commitments are short and local.* From Metrica: one defender stays committed for
0.78s at the median and 2.44s at p90, covering 2.44m of ground at the median and
9.14m at p90, with a median net displacement of 2.33m. A press is a step out of
the line, not a chase.

**What went into `defenders.py`.**

| constant | value | from |
| --- | --- | --- |
| `PRESS_BAND_FRONT` | -20.0 | band front edge, table above |
| `PRESS_BAND_BEHIND` | 5.0 | band back edge |
| `PRESS_MAX_EXCURSION` | 10.0 | press path p90, 9.14m |
| `PRESS_LATCH_TICKS` | 25 (2.5s) | commitment p90, 2.44s |

Two structural changes went with them. The presser is chosen by the distance from
each defender's **slot** to the ball rather than from its body, so responsibility
belongs to whoever owns the space rather than to whoever the last press left
nearest — that is what makes the press hand off as the ball moves instead of
towing one player around. And the excursion leash breaks the latch when a presser
is dragged more than 10m from its slot, so a carrier cannot hold a defender out of
shape for the full commit window.

**Validation.** Re-measuring the sim with the same statistics, over 40 episodes
against the 500k checkpoint:

| statistic | sim | real |
| --- | --- | --- |
| commit rate, ball in band | 0.77 | 0.70 |
| commit rate, ball out of band | 0.06 | 0.21 |
| nearest defender to ball, in band | 3.04m | 3.19m |
| mean defenders within 5m, in band | 1.01 | 0.95 |
| — exactly one | 0.53 | 0.49 |
| — two or more | 0.24 | 0.21 |
| back-line gap, nobody committed | 10.20m | 13.27m |
| back-line gap, someone committed | 10.24m | 13.30m |
| presser excursion from slot, p90 | 9.47m | 9.14m (path) |

The block is tighter than the real one in absolute terms (10.2m vs 13.3m back-line
gap), which is the formation sampler of section 9, not the press. What matters
here is that the gap is unchanged between committed and holding, exactly as in the
real data: the press no longer costs shape.

---

## Open gaps

Places where `defenders.py` does not currently match what was fitted. None of these
are silent, they are listed here so the discrepancy is on the record.

- **Back-line lateral gain.** Fitted 0.41, code uses 0.19. The midfield value of
  0.31 matches the fitted 0.29 well, so this looks like the back line specifically.
- **Line separation.** Fitted 15.58m mean, code uses `LINE_SEP = 10`.
- **Back-line width.** Fitted 30.15m mean, code uses hardcoded offsets spanning a
  fixed 40m, which is about the 94th percentile of observed width.
- **Duel hazard rate.** `LAM_MAX = 2.1` targets a 0.65 win rate over 0.5s, but
  measured contact windows are shorter than that, so the delivered rate is 0.51.
  Section 7.
- **Forward slot.** The lone forward in the 5-4-1 has no calibration data at all.
  Its resting x of 52.0 and its spread are assumptions.
- **`attacker_positioning` ignores its `gain` argument.** It is called with 0.7 from
  `compute_defender_targets` but hardcodes 0.4 in the body.
- **Duel calibration script is missing.** Only `duel_calibration_runs.csv` and
  `duel_calibration_summary.csv` were committed, so section 7's sim measurements
  cannot be regenerated.

Sample size is 8 matches from a single competition and season. Section 3 shows the
depth fit holds across those 8 matches, which is the strongest available check, but
it says nothing about other leagues, other eras, or teams that defend differently.

---

## Files

Scripts:

| file | what it does |
| --- | --- |
| `calibration_graphing.py` | Pulls StatsBomb frames and runs sections 1, 2, 4, 5, 6. Needs API access. |
| `validate_depth_fit.py` | Held-out validation of the depth fit, section 3. Runs offline. |
| `intercept_calibration.py` | Interception threshold sweep, section 8. Runs the sim. |
| `press_calibration.py` | Press policy, section 10. Offline: reads the committed low-block CSVs and the Metrica tracking data. |
| `../calibrate_defender_formation.py` | Formation sampler, section 9. Fits when run as main, and is imported by the environment for sampling. |

Data:

| file | contents |
| --- | --- |
| `low_block_frames.csv` | One row per freeze frame, 2032 rows. The source table for everything in sections 2 to 6. |
| `low_block_defenders.csv` | One row per player per frame, 36054 rows. |
| `low_block_summary_stats.csv` | Descriptive statistics for every shape quantity. |
| `input_ranges_q05_q95.csv` | 5th/95th percentile clip ranges used by `defenders.py`. |
| `depth_fit_coefficients.csv` | Depth regression coefficients and in-sample R2/RMSE. |
| `depth_fit_holdout.csv` | Held-out RMSE, baseline RMSE and skill, both split schemes. |
| `per_slot_stagger.csv` | Mean and sd of per-slot x offset from the line mean. |
| `defender_frame_params.csv` | Frame-level centroid, spread and gap parameters for the sampler. |
| `defender_slot_params.csv` | Per-slot standardised offset parameters for the sampler. |
| `duel_calibration_runs.csv` | Per contact-run duration measurements. |
| `duel_calibration_summary.csv` | Duel engagement rates and the fitted lambda. |
| `intercept_calibration_runs.csv` | One row per episode of the threshold sweep. |
| `intercept_calibration_summary.csv` | Per-threshold outcome mix and the achievable-p ceiling. |
| `press_frames.csv` | One row per low-block freeze frame with the press geometry, section 10. |
| `press_spells.csv` | One row per single-defender commitment measured off Metrica tracking. |
| `press_policy_params.csv` | Every number in section 10, tagged by source. |

Figures:

| file | shows |
| --- | --- |
| `depth_response.png` | Depth against ball only, and predicted against actual for the two-predictor fit. |
| `shape_diagnostic.png` | Depth, lateral shift, within-line scatter, width, gaps and attacking-line depth. |
| `intercept_calibration.png` | Outcome mix by threshold, interception rate per pass, and the achievable-p curve. |
| `defender_shape_fit.png` | Block depth, gap to the attacking line, spread correlation and lateral position. |
| `defender_slot_scatter.png` | Per-slot standardised offsets, real frames against fitted mean and sd. |
| `defender_samples.png` | Six sampled 5-4-1 blocks. |
| `press_engagement.png` | Engagement distance and the press band, plus why absolute ball x is the wrong axis. |
| `press_commitment.png` | How many defenders commit, and the StatsBomb/Metrica cross-check. |
| `press_shape.png` | Which line supplies the presser, and what the back line pays for it. |
| `press_duration.png` | Commitment length and ground covered, from Metrica tracking. |

---

## Reproducing

```
python defenders/calibration/calibration_graphing.py     # needs StatsBomb API access
python defenders/calibration/validate_depth_fit.py       # offline, reads committed CSVs
python defenders/calibration/intercept_calibration.py    # runs the sim, slow
python defenders/calibrate_defender_formation.py         # offline, reads committed CSVs
python defenders/calibration/press_calibration.py        # offline, reads committed CSVs + Metrica
```

Only `calibration_graphing.py` needs network access. The other four read CSVs that
are already committed, or run the sim directly.
