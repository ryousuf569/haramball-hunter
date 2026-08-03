import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from environment.termination import GOAL, scoring_probability

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMES_CSV = os.path.join(os.path.dirname(OUT_DIR), "..", "defenders", "calibration",
                          "low_block_frames.csv")
PITCH_X = 105.0

# A settled attack against a low block works the ball into a genuinely dangerous
# spot a handful of times a match. 2% of frames is ~5 per match at this dataset's
# 254 frames/match, which is the order of shots a team gets from settled play.
# The whole sweep is written out, so this target is a dial, not a fact.
TARGET_FRAC = 0.02

# reference distances: penalty spot, edge of the box, edge of the D
MARKS = [(11.0, "penalty spot"), (16.5, "edge of box"), (20.0, "edge of the D"),
         (30.0, "30m"), (40.0, "40m")]


def load_frames():
    with open(FRAMES_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    # mirror into sim coordinates, where the attacked goal is at x=105
    ball = np.array([[PITCH_X - float(r["ball_x"]), float(r["ball_y"])] for r in rows])
    att = np.array([[PITCH_X - float(r["highest_att_x"]), float(r["ball_y"])]
                    for r in rows])
    matches = [r["match_id"] for r in rows]
    return rows, ball, att, matches


def s_to_distance(s):
    # invert S(r) = (0.93 * exp(-0.14 * sqrt(d)))^0.48
    x = (s ** (1 / 0.48)) / 0.93
    return float((np.log(x) / -0.14) ** 2) if 0 < x < 1 else 0.0


def main():
    rows, ball, att, matches = load_frames()
    n_matches = len(set(matches))
    s_ball = scoring_probability(ball)
    s_att = scoring_probability(att)
    d_ball = np.linalg.norm(ball - GOAL, axis=1)

    print("%d frames, %d matches (%.0f frames/match)"
          % (len(rows), n_matches, len(rows) / n_matches))
    print("ball distance to goal: p50 %.1fm  p90 %.1fm  p98 %.1fm  min %.1fm"
          % (*np.percentile(d_ball, [50, 90, 98]), d_ball.min()))

    samples_path = os.path.join(OUT_DIR, "s_r_calibration_samples.csv")
    with open(samples_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["match_id", "event_type", "ball_dist_m", "s_r_ball", "s_r_top_attacker"])
        for r, d, sb, sa in zip(rows, d_ball, s_ball, s_att):
            w.writerow([r["match_id"], r["type"], round(float(d), 2),
                        round(float(sb), 4), round(float(sa), 4)])
    print("wrote", samples_path)

    # sweep: what each candidate threshold would cost in real frames
    lo, hi = np.percentile(s_ball, [50, 99.9])
    candidates = np.round(np.linspace(lo, hi, 40), 4)
    sweep = []
    for t in candidates:
        frac = float((s_ball >= t).mean())
        sweep.append(dict(threshold=t,
                          distance_m=round(s_to_distance(float(t)), 2),
                          frac_frames=round(frac, 4),
                          frames_per_match=round(frac * len(rows) / n_matches, 2),
                          frac_frames_top_attacker=round(float((s_att >= t).mean()), 4)))

    sweep_path = os.path.join(OUT_DIR, "s_r_calibration_sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    print("wrote", sweep_path)

    pick = min((s for s in sweep if s["frac_frames"] <= TARGET_FRAC),
               key=lambda s: s["threshold"], default=sweep[-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(s_ball, bins=50, color="#2a78d6", alpha=0.85)
    axes[0].axvline(pick["threshold"], color="#d1495b", lw=1.6,
                    label="pick %.3f (%.0fm)" % (pick["threshold"], pick["distance_m"]))
    axes[0].set_xlabel("S(r) at the ball, real low-block frames")
    axes[0].set_ylabel("frames")
    axes[0].legend(fontsize=8)

    axes[1].plot([s["threshold"] for s in sweep],
                 [s["frames_per_match"] for s in sweep], color="#eb6834", marker="o", ms=3)
    axes[1].axvline(pick["threshold"], color="#d1495b", lw=1.6)
    axes[1].axhline(TARGET_FRAC * len(rows) / n_matches, color="#888", lw=1, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("S(r) threshold")
    axes[1].set_ylabel("qualifying frames per match")
    fig.suptitle("S(r) against 2032 real settled low-block frames", y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "s_r_calibration.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print("wrote", fig_path)

    print("\nS(r) at reference distances:")
    for d, name in MARKS:
        print("  %-14s %5.1fm -> S = %.3f"
              % (name, d, scoring_probability(np.array([[GOAL[0] - d, GOAL[1]]]))[0]))

    print("\nwhat each threshold costs in real frames:")
    for s in sweep[::4]:
        print("  S >= %.3f  (%5.1fm)  %5.2f%% of frames  %5.2f per match"
              % (s["threshold"], s["distance_m"], 100 * s["frac_frames"],
                 s["frames_per_match"]))

    print("\npick at <= %.0f%% of real frames: S(r) >= %.3f  (%.1fm from goal, "
          "%.2f frames/match)" % (100 * TARGET_FRAC, pick["threshold"],
                                  pick["distance_m"], pick["frames_per_match"]))


if __name__ == "__main__":
    main()
