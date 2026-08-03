# PPCF validation against Laurie Shaw's implementation

`physics/ppcf.py` is a from-scratch pitch control model. This folder checks it
against an independent implementation of the same model — Laurie Shaw's
[LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
`Metrica_PitchControl.py`, following Spearman (2018) — on real tracking frames.

## Running it

```bash
python physics/validation/ppcf_validation.py --download   # first time only, ~66 MB
python physics/validation/ppcf_validation.py              # 20 frames, ~2 min
python physics/validation/ppcf_validation.py --frames 40
```

## Files

| File | What it is |
|---|---|
| `Metrica_PitchControl.py` | Shaw's model, **verbatim**. What we validate against. |
| `Metrica_IO.py` | Shaw's data loader, verbatim. |
| `Metrica_Velocities.py` | Shaw's velocity estimator, verbatim. Kept for reference — not imported, see caveats. |
| `ppcf_validation.py` | The experiment. |
| `data/` | Metrica's public sample data, downloaded on demand. |
| `results/` | CSVs and figures. |

## Method

20 pass events are sampled evenly across Metrica Sample Game 1. For each one,
both models are evaluated on the same 50x32 grid of points, with the same players,
at the same instant — 32,000 cells in total. We then scatter our attacking-team
control probability against his, cell by cell.

**Both models get the same constants.** Rather than typing them twice, the
parameter dict handed to Shaw's code is built directly from the constants in
`physics/ppcf.py` and `physics/tti.py`, so they cannot drift apart:

| Constant | Value | Read from |
|---|---|---|
| max player speed | 5.0 m/s | `tti.v_max` |
| reaction time | 0.54 s | `tti.reaction_time` |
| arrival-time sigma | 0.45 s | `tti.intercept_uncertainty` |
| λ attacking | 4.30 | `ppcf.attacker_control_rate` |
| λ defending | 7.40 (κ = 1.72) | `ppcf.defender_control_rate` |
| integration timestep | 0.08 s | `ppcf.integration_timestep` |
| integration horizon | 10 s | `ppcf.integration_horizon` |

Three features of Shaw's code have no counterpart in ours, so they are switched
off to keep the two looking at the same problem:

- **Goalkeeper λ boost.** He gives keepers 3x the control rate. We have no
  goalkeeper concept, so the keeper is just another defender.
- **Offside removal.** He can drop offside attackers from the calculation. We have
  no offside concept, so they stay in for both.
- **Ball travel time.** He starts the integral when the ball would arrive. We
  always start at t = 0 — `PPCF_grid`'s `ball_pos` argument only fills the
  `players['i_p']` interception cache that `turnover.py` reads, and never enters
  the integral. So Shaw is given `ball_start_pos=None`, which makes him do the
  same.

His head-start veto shortcut (which returns a hard 0 or 1 when one team arrives
far enough ahead) is also disabled, because we always run the integral.

## Results

20 frames, 32,000 cells:

| | |
|---|---|
| **R²** | **0.9939** |
| **Mean absolute deviation** | **0.0135** |
| RMSE | 0.0336 |
| Bias (ours − Shaw) | +0.0014 |
| p95 \|deviation\| | 0.0713 |
| max \|deviation\| | 0.5270 |
| cells agreeing within 0.05 | 91.8% |
| best fit | ours = 1.000 × shaw + 0.001 |

Per-frame R² ranges from 0.975 to 0.999.

![scatter](results/ppcf_scatter.png)

The fit has slope 1.000 and essentially zero bias, so there is no systematic
stretch or offset between the two surfaces — the residuals are symmetric noise
around zero, and the difference maps show they sit in thin bands along the
boundaries between attacking and defending control, not in the settled areas.

![surfaces](results/ppcf_surfaces.png)

**What the remaining 0.0135 is.** With every constant matched, one modelling
difference is left, and it is not something you can set with a parameter: how long
a player takes to reach a point.

- Shaw drifts the player at their current velocity for the reaction time, then
  runs them at top speed to the target. There is no acceleration limit, and
  turning around is free.
- We model acceleration explicitly (`a_max = 7 m/s²` in `physics/tti.py`),
  including braking and reversing when a player is moving away from the target.

Ours is the more conservative of the two: a stationary player needs ~0.2 s longer
to cover a moderate distance, and a player running the wrong way is penalised for
having to stop first. That shifts control boundaries by a fraction of a metre,
which is exactly where the residuals show up. The large `max |deviation|` values
are single cells sitting right on a boundary that the two models place slightly
differently.

## Outputs

| File | Contents |
|---|---|
| `results/ppcf_pointwise.csv` | Every cell: event, frame, x, y, both PPCF values, difference |
| `results/ppcf_per_frame.csv` | R², MAD and friends for each frame separately |
| `results/ppcf_summary.csv` | The overall numbers in the table above |
| `results/ppcf_scatter.png` | Scatter + residual histogram |
| `results/ppcf_surfaces.png` | Two example frames: his surface, ours, the difference |

## Caveats

- **Velocities use a moving-average smoother, not Savitzky-Golay.** Shaw defaults
  to Savitzky-Golay, which needs scipy; this repo does not depend on scipy, so the
  moving-average branch he also provides is used instead. Both models are fed the
  identical velocities, so this cannot tilt the comparison — it only shifts the
  common input slightly.
- **Four of Shaw's helpers are reimplemented.** `Metrica_IO` and
  `Metrica_Velocities` were written against pandas 0.x and call
  `Series.idxmax(2)`, which raises on pandas 2.x. `second_half_index`,
  the playing-direction flip, `calc_player_velocities` and `find_goalkeeper` in
  `ppcf_validation.py` are his logic with that fixed. `Metrica_PitchControl.py` —
  the model under test — is used exactly as he wrote it.
- **Frames in the first second of either half are skipped.** The velocity smoother
  has no history there, so every player reads as stationary — a data artefact
  rather than a state worth comparing on.
- **Shaw prints "Integration failed to converge" on a couple of cells.** With his
  veto shortcut disabled and a 0.08 s timestep, a few cells do not reach 0.99
  total probability inside the 10 s horizon. It is his own diagnostic, it affects
  a handful of cells out of 32,000, and our model stops at the same horizon.
