"""Do real carriers survive a defender at 1.2m as long as our duel model says?

The 500k run's constrained arms lost the ball to duels 103 times in 150
episodes. defenders/turnover.py's LAM_MAX is the one physics constant taken
from RoboCup rather than fitted to football, so it is worth checking against
the Metrica tracking data already in physics/validation/data.

SUPERSEDED, kept as the record of what prompted the refit. This script imports
the duel constants live, so re-running it now prints the CALIBRATED model against
the same Metrica exposure, while the commentary at the bottom still describes the
2.1 / 1.2m model that motivated it. The fit itself is
defenders/calibration/duel_calibration.py and README section 7; it uses Metrica's
labelled ground-duel EVENTS, which this script did not look at, so it can measure
a dispossession hazard rather than only the exposure carriers accept.

Run: python models/081226-500k/analysis/duel_reality_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "physics", "validation"))

os.environ.setdefault("MPLBACKEND", "Agg")

from defenders.turnover import (  # noqa: E402
    ADV_FLOOR, DUEL_A, DUEL_B, DUEL_EXP, LAM_MAX,
)
from physics.validation.dribble_speed_calibration import (  # noqa: E402
    load, possession_mask, spells, stack_team,
)

DT = 0.04          # Metrica is 25 Hz
SIM_DT = 0.1       # our tick
CONTEST_R = DUEL_A  # the duel's hard cutoff, 1.2m


def model_survival(dist, closing=1.0, dt=SIM_DT):
    """Per-tick loss probability our model assigns to a contested carry."""
    p_geom = max(0.0, 1.0 - (dist / DUEL_A) ** DUEL_EXP)
    if p_geom <= 0:
        return 0.0
    adv = float(np.clip(closing / 5.0, -1.0, 1.0))
    mult = ADV_FLOOR + (1.0 - ADV_FLOOR) * 0.5 * (adv + 1.0)
    return 1.0 - np.exp(-LAM_MAX * p_geom * mult * dt)


def main():
    home, away = load()
    hid, hx, hy, hvx, hvy, _ = stack_team(home, "Home")
    aid, ax, ay, avx, avy, _ = stack_team(away, "Away")
    ball = tuple(home[f"ball_{c}"].to_numpy(dtype=float)
                 for c in ("x", "y", "vx", "vy"))

    rows = []
    for (x, y, vx, vy), (ox, oy) in (((hx, hy, hvx, hvy), (ax, ay)),
                                     ((ax, ay, avx, avy), (hx, hy))):
        poss = possession_mask(x, y, vx, vy, ball)
        for j in range(x.shape[1]):
            for a, b in spells(poss[:, j], min_len=2):
                cx, cy = x[a:b, j], y[a:b, j]
                d = np.hypot(ox[a:b] - cx[:, None], oy[a:b] - cy[:, None])
                d = np.where(np.isnan(d), np.inf, d)
                nearest = d.min(axis=1)
                if not np.isfinite(nearest).any():
                    continue
                rows.append((len(cx) * DT, float(np.nanmin(nearest)),
                             float(np.mean(nearest <= CONTEST_R) * len(cx) * DT)))

    dur = np.array([r[0] for r in rows])
    closest = np.array([r[1] for r in rows])
    contested = np.array([r[2] for r in rows])

    print("=" * 78)
    print(f"REAL CARRIES, Metrica game 1: {len(rows):,} possession spells")
    print("=" * 78)
    print(f"  duration          median {np.median(dur):.2f}s  "
          f"p90 {np.percentile(dur, 90):.2f}s  max {dur.max():.2f}s")
    print(f"  closest opponent  median {np.median(closest):.2f}m  "
          f"p10 {np.percentile(closest, 10):.2f}m")

    tight = contested > 0
    print(f"\n  carries with an opponent inside {CONTEST_R:.1f}m at some point: "
          f"{tight.mean():.1%} ({tight.sum():,})")
    if tight.any():
        ct = contested[tight]
        print(f"  time spent inside {CONTEST_R:.1f}m, on those carries:")
        print(f"     median {np.median(ct):.2f}s  p75 {np.percentile(ct,75):.2f}s  "
              f"p90 {np.percentile(ct,90):.2f}s  max {ct.max():.2f}s")

    print("\n" + "=" * 78)
    print("WHAT OUR MODEL SAYS ABOUT THE SAME SITUATION")
    print("=" * 78)
    print(f"{'gap (m)':>8s} {'p(loss)/tick':>13s} {'median survival':>16s} "
          f"{'p(lost within 1s)':>18s}")
    for dist in (0.4, 0.6, 0.8, 1.0, 1.15):
        p = model_survival(dist)
        if p <= 0:
            continue
        med = np.log(0.5) / np.log(1 - p) * SIM_DT
        print(f"{dist:8.2f} {p:13.4f} {med:15.2f}s {1-(1-p)**10:18.3f}")

    print("\n  The model has a hard cutoff at 1.2m and nothing outside it, so a")
    print("  carrier is either untouchable or in serious trouble. Real carriers")
    print("  spend substantial time inside that radius: a quarter of all spells")
    print("  go in, for a median 0.6s and up to 3.8s. The model gives a carrier")
    print("  at 1.0m an 81% chance of losing the ball within the first second.")
    print("  (These spells end for any reason, pass or shot or tackle, so this")
    print("  is not a survival rate. It is the exposure real carriers accept.)")
    print(f"\n  Carrier top speed 4.05 m/s against a 5.0 m/s defender, so once")
    print(f"  the gap is closed the carrier cannot reopen it.")


if __name__ == "__main__":
    main()
