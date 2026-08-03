import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

# ppcf_validation owns the Metrica loading (and puts Metrica_IO on the path)
sys.path.insert(0, os.path.join(REPO_ROOT, "physics", "validation"))
import physics.validation.ppcf_validation as pv
import Metrica_PitchControl as mpc

import physics.ppcf as ppcf
from environment.termination import AREA_DI, AREA_DJ, AREA_RADIUS
from environment.grid import PC_CELL_SIZE

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# every event type that has a real player on the ball at its start frame
ON_BALL_TYPES = ("PASS", "BALL LOST", "SHOT", "CHALLENGE")
PICK_PCTL = 25          # keep the control level 75% of real shots reached
TARGET_RETENTION = 0.90  # what anchor 1 was looking for, and never finds

# the cells pcf_in_area averages, as metre offsets from the player
AREA_OFFSETS = np.stack([AREA_DI * PC_CELL_SIZE, AREA_DJ * PC_CELL_SIZE], axis=1)


def collect():
    home, away, events, gk = pv.load_match()
    params = mpc.default_model_params()
    half_starts = [home.index[0], pv.second_half_index(home)]

    onball = events[events.Type.isin(ON_BALL_TYPES)
                    & events["Start X"].notna() & events["Start Y"].notna()]

    rows = []
    for eid in onball.index:
        row = events.loc[eid]
        frame = int(row["Start Frame"])
        if frame not in home.index or frame not in away.index:
            continue
        if any(0 <= frame - s < pv.WARMUP for s in half_starts):
            continue  # velocity smoother has no history yet
        if np.isnan(home.loc[frame, "Home_%s_x" % gk[0]]):
            continue
        if np.isnan(away.loc[frame, "Away_%s_x" % gk[1]]):
            continue

        attacking, defending = pv.teams_at_event(events, home, away, gk, eid, params)
        players = pv.to_our_players(attacking, defending)

        # the goal this team attacks is the one its own keeper is not standing in
        own_gk_x = (home.loc[frame, "Home_%s_x" % gk[0]] if row.Team == "Home"
                    else away.loc[frame, "Away_%s_x" % gk[1]])
        goal = np.array([-np.sign(own_gk_x) * pv.FIELD[0] / 2, 0.0])

        carrier = np.array([row["Start X"], row["Start Y"]], dtype=float)
        targets = carrier + AREA_OFFSETS
        per_player = ppcf.PPCF_grid(targets, players, carrier)
        area = float(per_player[:, players["team"] == "attacker"].sum(axis=1).mean())

        # kept the ball = he used it and the next event is still this team's
        nxt = events.index.get_loc(eid) + 1
        kept = (row.Type in ("PASS", "SHOT") and nxt < len(events)
                and events.iloc[nxt].Team == row.Team)

        rows.append({"event_id": int(eid), "frame": frame, "type": row.Type,
                     "team": row.Team, "pcf_in_area": round(area, 4),
                     "dist_to_goal_m": round(float(np.linalg.norm(carrier - goal)), 2),
                     "kept": int(bool(kept))})

    return rows


def main():
    rows = collect()
    v = np.array([r["pcf_in_area"] for r in rows])
    kept = np.array([r["kept"] for r in rows], dtype=bool)
    types = np.array([r["type"] for r in rows])

    print("\n%d on-ball moments over %d cells within %.0fm"
          % (len(rows), len(AREA_OFFSETS), AREA_RADIUS))

    samples_path = os.path.join(OUT_DIR, "pcf_calibration_samples.csv")
    with open(samples_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", samples_path)

    # per event type: what control real players had while doing each thing
    by_type = []
    for t in ON_BALL_TYPES:
        s = v[types == t]
        if not s.size:
            continue
        by_type.append(dict(event_type=t, n=int(s.size),
                            mean=round(float(s.mean()), 4),
                            p25=round(float(np.percentile(s, 25)), 4),
                            p50=round(float(np.percentile(s, 50)), 4),
                            p75=round(float(np.percentile(s, 75)), 4)))

    # anchor 1: does control predict keeping the ball? (it does not)
    sweep = []
    for t in np.round(np.arange(0.30, 1.001, 0.02), 2):
        m = v >= t
        sweep.append(dict(threshold=t, n_moments=int(m.sum()),
                          frac_moments=round(float(m.mean()), 4),
                          retention=round(float(kept[m].mean()), 4) if m.sum() else ""))

    sweep_path = os.path.join(OUT_DIR, "pcf_calibration_sweep.csv")
    with open(sweep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)

    type_path = os.path.join(OUT_DIR, "pcf_calibration_by_type.csv")
    with open(type_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_type[0]))
        w.writeheader()
        w.writerows(by_type)
    print("wrote", sweep_path)
    print("wrote", type_path)

    shots = v[types == "SHOT"]
    pick = round(float(np.percentile(shots, PICK_PCTL)), 3)

    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    axes[0].hist(v[types != "SHOT"], bins=25, color="#c9d4e0",
                 label="other on-ball moments")
    axes[0].hist(shots, bins=25, color="#eb6834", label="shots (n=%d)" % shots.size)
    axes[0].axvline(pick, color="#d1495b", lw=1.6,
                    label="p%d of shots = %.3f" % (PICK_PCTL, pick))
    axes[0].set_xlabel("pcf_in_area at the player on the ball")
    axes[0].set_ylabel("moments")
    axes[0].legend(fontsize=8)

    axes[1].plot([s["threshold"] for s in sweep],
                 [s["retention"] if s["retention"] != "" else np.nan for s in sweep],
                 color="#2a78d6", marker="o", ms=3, label="retention above threshold")
    axes[1].plot([s["threshold"] for s in sweep],
                 [s["frac_moments"] for s in sweep],
                 color="#8a8880", lw=1, ls="--", label="share of moments")
    axes[1].axhline(TARGET_RETENTION, color="#888", lw=1, ls=":")
    axes[1].axvline(pick, color="#d1495b", lw=1.6)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("pcf_in_area threshold")
    axes[1].set_ylabel("rate")
    axes[1].legend(fontsize=8, loc="lower left")
    # the reason the two gate conditions fight each other, in one panel
    dist = np.array([r["dist_to_goal_m"] for r in rows])
    bands = np.arange(0, 90, 10)
    idx = np.digitize(dist, bands) - 1
    centres = [bands[b] + 5 for b in range(len(bands) - 1) if (idx == b).any()]
    means = [v[idx == b].mean() for b in range(len(bands) - 1) if (idx == b).any()]
    axes[2].scatter(dist[types != "SHOT"], v[types != "SHOT"], s=5, alpha=0.25,
                    color="#c9d4e0")
    axes[2].scatter(dist[types == "SHOT"], v[types == "SHOT"], s=18, color="#eb6834",
                    label="shots")
    axes[2].plot(centres, means, color="#2a78d6", lw=1.8, marker="o", ms=4,
                 label="mean per 10m band")
    axes[2].set_xlabel("distance to goal (m)")
    axes[2].set_ylabel("pcf_in_area")
    axes[2].legend(fontsize=8, loc="lower right")

    fig.suptitle("pcf_in_area against Metrica Sample_Game_1", y=1.02)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "pcf_calibration.png")
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print("wrote", fig_path)

    print("\ncontrol of own space, by what the player did with the ball:")
    print("  %-11s %5s %7s %7s %7s %7s" % ("type", "n", "mean", "p25", "p50", "p75"))
    for b in by_type:
        print("  %-11s %5d %7.3f %7.3f %7.3f %7.3f"
              % (b["event_type"], b["n"], b["mean"], b["p25"], b["p50"], b["p75"]))

    print("\nanchor 1, retention (this is the one that fails):")
    for s in sweep[::5]:
        print("  >= %.2f  %4d moments  retention %s"
              % (s["threshold"], s["n_moments"],
                 "%.3f" % s["retention"] if s["retention"] != "" else "-"))
    ok = [s for s in sweep if s["retention"] != "" and s["n_moments"] >= 30
          and s["retention"] >= TARGET_RETENTION]
    print("  no threshold reaches %.0f%% retention on >= 30 moments"
          % (100 * TARGET_RETENTION) if not ok else
          "  lowest reaching %.0f%%: %.2f" % (100 * TARGET_RETENTION,
                                              min(s["threshold"] for s in ok)))
    print("  mean pcf_in_area kept %.3f vs lost %.3f -- barely separated"
          % (v[kept].mean(), v[~kept].mean()))

    print("\nanchor 2, shots: pcf_in_area >= %.3f" % pick)
    print("  %d%% of real shots were taken with at least this much control of the "
          "%.0fm around the shooter" % (100 - PICK_PCTL, AREA_RADIUS))
    print("  it passes %.1f%% of all on-ball moments"
          % (100 * float((v >= pick).mean())))

    # control and danger pull against each other, and this is by how much
    print("\ncontrol of own space by distance to goal:")
    for b in range(len(bands) - 1):
        m = idx == b
        if m.sum():
            print("  %2d-%2dm  n=%4d  mean pcf_in_area %.3f"
                  % (bands[b], bands[b] + 10, m.sum(), v[m].mean()))
    print("  correlation with distance to goal: r = %+.3f"
          % np.corrcoef(dist, v)[0, 1])


if __name__ == "__main__":
    main()
