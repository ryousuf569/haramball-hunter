# Success-condition calibration

`environment/termination.py` decides when an episode ends in success. This file
records where its numbers came from, why those anchors were chosen, and what came
out. Same split as `defenders/calibration/README.md`: **data fits** are measured
off real football, **sim sweeps** are tuned by running the model. Both kinds are
here, and section 4 is the one that matters most, because it checks whether the
fitted numbers are reachable at all.

A note on coordinates. The sim attacks x=105 on a 105x68 pitch, with the goal at
`GOAL = [105, 34]`. `low_block_frames.csv` is in the calibration frame, where the
attacked goal is at x=0, so `s_r_calibration.py` mirrors x before scoring. Metrica
tracking is in its own metric frame (origin at the centre spot, 106x68); nothing
there is converted, because pitch control only depends on distances between
players, and the goal each team attacks is found from its own keeper's position.

---

## 1. What is being calibrated

`check_shot_opening` ends the episode in success when the attacker on the ball
clears three conditions at once:

| condition | value | set by |
| --- | --- | --- |
| `pcf_in_area(...) >= 0.30` | control of his own space | section 3 |
| `scoring_probability(...) >= 0.779` | in a dangerous position | section 2 |
| `nearest_defender_distance(...)` (> 3m) | not being closed down | not calibrated |

`pcf_in_area` replaced a single-cell lookup. Pitch control evaluated at exactly
the ball's cell is ~1 for whoever is holding it, so a single cell tests almost
nothing. `AREA_RADIUS = 3.0` against 2m cells averages the 3x3 block around the
player, which asks whether he owns the space rather than the point.

---

## 2. S(r) threshold, from real low-block frames

**How.** `s_r_calibration.py` reads `defenders/calibration/low_block_frames.csv`,
the same 2032 StatsBomb freeze frames of settled possession against a low block
that sections 1 to 6 of the defender README are built on. Every frame's ball
position is mirrored into sim coordinates and scored with `scoring_probability`.
The threshold is then swept and each candidate is reported three ways: fraction of
real frames it passes, frames per match, and the distance from goal it corresponds
to (S(r) inverted analytically).

**Why this anchor.** S(r) as implemented is not calibrated xG. It spans only
0.49 to 0.88 across the whole pitch (0.49 is the far corner, 100m out), so no
absolute value on it means anything on its own, and a threshold picked by
intuition, e.g. "0.2 because that sounds like a decent chance", sits below the
curve's floor and passes every position on the pitch. What can be anchored is
*rarity*: a threshold that passes X% of real settled low-block possession fires
about as often as real teams get the ball into positions that dangerous.

**Results.** 2032 frames over 8 matches, 254 frames per match. Ball distance to
goal is p50 40.8m, p90 57.2m, minimum 4.8m.

| S(r) threshold | distance | % of real frames | frames per match |
| --- | --- | --- | --- |
| 0.699 | 23.1m | 11.32 | 28.75 |
| 0.713 | 20.4m | 8.46 | 21.50 |
| 0.727 | 17.8m | 6.10 | 15.50 |
| 0.741 | 15.5m | 4.53 | 11.50 |
| 0.770 | 11.4m | 2.41 | 6.12 |
| **0.779** | **10.2m** | **1.82** | **4.62** |
| 0.798 | 8.1m | 0.39 | 1.00 |

For reference, on this curve: penalty spot (11m) is S = 0.773, edge of the box
(16.5m) is 0.735, 30m out is 0.668.

`TARGET_FRAC = 0.02` was chosen and 0.779 falls out of it: ~4.6 qualifying
moments per match, which is the order of shots a team gets from settled
possession against a low block. That target is a design choice, not a
measurement, and it is a single constant at the top of the script. The full
sweep is in `s_r_calibration_sweep.csv` so a different rate can be read straight
off it.

---

## 3. `pcf_in_area` threshold, from Metrica tracking

**How.** `pcf_success_calibration.py` uses Metrica's public Sample_Game_1, the
same match `physics/validation/ppcf_validation.py` validates our pitch control
against, and reuses that script's loaders so the frames are built identically.
Every event with a real player on the ball at its start frame is taken (PASS,
BALL LOST, SHOT, CHALLENGE), 1311 after dropping the velocity-smoother warmup and
frames where a keeper is untracked. For each, `physics/ppcf.py` is run on the real
22-player frame over the same 9-cell pattern `pcf_in_area` uses, centred on the
player, with the ball at his feet.

**Why two anchors.** "In control of his space" is not a label in any dataset, so
it has to be inferred from what the player did next. Two candidates were measured.

**Anchor 1, retention. This one fails, and that is worth recording.** Label each
moment by whether the next event still belongs to the same team, then ask what
control level predicts keeping the ball.

| pcf_in_area >= | moments | retention |
| --- | --- | --- |
| 0.30 | 1156 | 0.605 |
| 0.50 | 625 | 0.677 |
| 0.70 | 340 | 0.768 |
| 0.90 | 169 | 0.793 |

Retention climbs from 0.61 to 0.79 and flattens; no threshold reaches 90% on a
sample of 30 or more moments. Mean control is 0.599 for moments that kept the
ball against 0.475 for moments that lost it. Controlling your own 3m barely moves
the odds of the next event still being yours, because that depends mostly on the
pass. The sweep is written out anyway as `pcf_calibration_sweep.csv`.

**Anchor 2, shots. This is the one used.** It asks the same question the gate
asks: how much of his own space did a player have when he chose to shoot?

| event type | n | mean | p25 | p50 | p75 |
| --- | --- | --- | --- | --- | --- |
| PASS | 797 | 0.595 | 0.391 | 0.550 | 0.809 |
| BALL LOST | 257 | 0.536 | 0.350 | 0.475 | 0.707 |
| SHOT | 24 | 0.430 | 0.298 | 0.402 | 0.478 |
| CHALLENGE | 233 | 0.403 | 0.310 | 0.392 | 0.480 |

`PICK_PCTL = 25` gives **0.298**, rounded to 0.30 in the code: three quarters of
real shots were taken with at least this much control of the 3m around the
shooter.

**The number is low, and the reason is structural.** Shots have *less* area
control than passes, and about the same as contested challenges. Control rises
with distance from goal:

| distance to goal | n | mean pcf_in_area |
| --- | --- | --- |
| 0-10m | 13 | 0.319 |
| 10-20m | 36 | 0.348 |
| 20-30m | 96 | 0.420 |
| 30-40m | 132 | 0.552 |
| 40-50m | 177 | 0.551 |
| 50-60m | 240 | 0.518 |
| 60-70m | 220 | 0.534 |
| 70-80m | 154 | 0.570 |

Correlation with distance to goal is r = +0.286. "Owns his area" and "is in a
dangerous position" are opposing conditions in real football: near goal nobody
owns their space. So this term should be read as a floor that rejects a player
who is completely swamped, not as a test of full control. The S(r) term is what
carries the danger requirement.

---

## 4. Reachability in the sim

The thresholds above are fitted to real football. Whether the sim can produce
them is a separate question, and it was measured by running episodes with the
success gate patched off so nothing terminates early: 12 seeds, up to 300 ticks
each, 1805 ticks with an attacker on the ball.

| quantity | sim p50 | sim p95 | sim max | real p50 |
| --- | --- | --- | --- | --- |
| S(r) at the holder | 0.580 | 0.589 | 0.647 | 0.629 |
| pcf_in_area at the holder | 0.972 | 0.992 | 0.995 | 0.481 |

Two consequences, both of which should be stated rather than discovered later:

1. **No episode can currently end in success.** The sim's best S(r) over 1805
   on-ball ticks is 0.647, about 35m from goal, against a threshold of 0.779 at
   10m. The current attacker is the throwaway scripted baseline, which circulates
   the ball and never enters the box, so this is the honest answer for that
   policy rather than a broken gate. It does mean every episode ends in a
   turnover or runs out until the attackers are learned.
2. **The `pcf_in_area` condition cannot discriminate in the sim at any
   threshold.** 99.9% of on-ball ticks clear 0.30, and the sim's median of 0.972
   is nothing like the real 0.481, because the low block never presses the
   carrier in midfield. The gap is a property of the defender policy, not of the
   threshold.

---

## Open gaps

- **The 3m defender-distance condition is not calibrated.** It is also largely
  redundant with `AREA_RADIUS = 3.0`, since both ask about the same circle.
- **`nearest_defender_distance` returns a bool, not a distance.** It evaluates
  `> 3.0` internally, so the threshold is not visible at the call site with the
  other two.
- **S(r) uses `exp(-0.14 * sqrt(d))`.** That curve is very flat: 0.83 at 5m, 0.59
  at 52m, 0.49 at 100m. Every threshold in section 2 is meaningful only relative
  to this curve, and none of them are probabilities in the xG sense.
- **Sample sizes.** The shot anchor rests on 24 shots in one match. The low-block
  frames are 8 matches from a single competition and season.
- **Game-state mismatch.** Metrica shots come from all game states, not
  specifically from settled possession against a low block.
- **Both fits describe the current environment.** The sim numbers in section 4
  are measured against the throwaway baseline attacker; they should be re-measured
  once the attackers are learned.
- **The section 4 script was not committed.** The reachability numbers cannot
  currently be regenerated from the repo, only re-derived.

---

## Files

Scripts:

| file | what it does |
| --- | --- |
| `s_r_calibration.py` | Section 2. Offline, reads the committed low-block CSV. |
| `pcf_success_calibration.py` | Section 3. Needs the Metrica data in `physics/validation/data/`. |

Data:

| file | contents |
| --- | --- |
| `s_r_calibration_samples.csv` | One row per real frame: distance, S(r) at the ball, S(r) at the most advanced attacker. |
| `s_r_calibration_sweep.csv` | Per candidate threshold: implied distance, share of frames, frames per match. |
| `pcf_calibration_samples.csv` | One row per Metrica on-ball moment: type, `pcf_in_area`, distance to goal, kept/lost. |
| `pcf_calibration_sweep.csv` | Anchor 1: moments above each threshold and their retention rate. |
| `pcf_calibration_by_type.csv` | Anchor 2: `pcf_in_area` distribution per event type. |

Figures:

| file | shows |
| --- | --- |
| `s_r_calibration.png` | S(r) distribution over real frames with the pick, and qualifying frames per match against threshold. |
| `pcf_calibration.png` | `pcf_in_area` distribution with shots highlighted, the retention curve, and control against distance to goal. |

---

## Reproducing

```
python environment/calibration/s_r_calibration.py            # offline
python environment/calibration/pcf_success_calibration.py    # needs Metrica data
```

`s_r_calibration.py` reads a CSV that is already committed. `pcf_success_calibration.py`
needs Metrica's Sample_Game_1 in `physics/validation/data/`; fetch it with
`python physics/validation/ppcf_validation.py --download`.
