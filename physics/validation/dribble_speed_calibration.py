"""DRIBBLE_SPEED, fitted to Metrica tracking.

engine.py caps a ball carrier at DRIBBLE_SPEED * V_MAX. That constant started as
a guess (0.82) picked to stop a scripted carry-and-shoot converting 98% against
the low block, which is not a good enough reason for a physics constant.

Metrica's event file has no CARRY type, so carries come from tracking: a player
is on the ball when he is nearest it, inside POSSESSION_RADIUS, AND moving with
it (proximity alone counts a pass flying over a sprinter as a very fast carry).

The transferable quantity is the RATIO of carrying speed to free-running speed,
not either absolute -- real players top out near 9 m/s and engine.V_MAX is 5.0.
Same reasoning turnover.py uses for the RoboCup duel geometry.

Two anchors are measured. The first fails, and section 1 of the printout records
why, because the failure is the whole reason the second one is built the way it
is. Run: python physics/validation/dribble_speed_calibration.py
"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(VALIDATION_DIR))
for _p in (REPO_ROOT, VALIDATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Metrica_IO as mio          # noqa: E402
# Metrica_Velocities is not imported, for the reason press_calibration.py gives:
# it needs scipy for its Savitzky-Golay smoother and scipy is not a project
# dependency. add_speed below is a centred difference over the same window.

DATA_DIR = os.path.join(VALIDATION_DIR, "data")
OUT_DIR = os.path.join(VALIDATION_DIR, "results")
GAME_ID = 1
FPS = 25
VEL_WINDOW = 5             # frames, 0.2s, centred

POSSESSION_RADIUS = 1.5    # a carrier keeps the ball inside about a stride
REL_SPEED_MAX = 2.0        # a carried ball travels WITH the carrier
MAX_SPEED = 11.0           # above this is a tracking glitch

# --- what counts as a run ------------------------------------------------------
# The unit of comparison. Anchor 1 compares every frame, which is a selection
# bias: carrying frames are all active, while "free" frames are 3.1M of a match
# that is mostly walking. Restricting both sides to high-intensity efforts is
# what makes the comparison like-for-like.
RUN_SPEED_MIN = 4.0        # m/s, the floor of a genuine effort
RUN_MIN_FRAMES = 8         # ~0.32s
CARRY_RUN_FRAC = 0.60      # possession fraction over the run to call it a carry
FREE_RUN_FRAC = 0.05       # ...and to call it ball-free
MIN_RUNS_PER_PLAYER = 5

ANCHOR_Q = 0.95            # the sim constant is a CAP, so anchor high
QUANTILES = (0.50, 0.75, 0.90, 0.95, 0.99)
OLD_GUESS = 0.82


def player_ids(team, prefix):
    return sorted({c[:-2] for c in team.columns
                   if c.startswith(prefix) and c.endswith("_x")})


def centred_velocity(series):
    return (series.shift(-VEL_WINDOW // 2)
            - series.shift(VEL_WINDOW // 2)) / (VEL_WINDOW / FPS)


def add_speed(team, prefix):
    """Player and ball velocity by centred difference. Direction of play does
    not matter to a speed, so the second-half flip
    mio.to_single_playing_direction would do (and which raises on pandas 2.x)
    is not needed here."""
    names = player_ids(team, prefix) + (["ball"] if "ball_x" in team else [])
    for pid in names:
        vx, vy = centred_velocity(team[f"{pid}_x"]), centred_velocity(team[f"{pid}_y"])
        team[f"{pid}_vx"], team[f"{pid}_vy"] = vx, vy
        team[f"{pid}_speed"] = np.hypot(vx, vy)
    return team


def load():
    home, away, _events = mio.read_match_data(DATA_DIR, GAME_ID)
    return (add_speed(mio.to_metric_coordinates(home), "Home"),
            add_speed(mio.to_metric_coordinates(away), "Away"))


def stack_team(team, prefix):
    ids = player_ids(team, prefix)
    get = lambda suf: team[[f"{p}{suf}" for p in ids]].to_numpy(dtype=float)
    return ids, get("_x"), get("_y"), get("_vx"), get("_vy"), get("_speed")


def possession_mask(x, y, vx, vy, ball):
    """(n_frames, n_players) bool: nearest to the ball, close to it, and moving
    with it."""
    bx, by, bvx, bvy = ball
    d = np.hypot(x - bx[:, None], y - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    rel = np.hypot(vx - bvx[:, None], vy - bvy[:, None])
    rel = np.where(np.isnan(rel), np.inf, rel)

    ranked = np.where((d <= POSSESSION_RADIUS) & (rel <= REL_SPEED_MAX), d, np.inf)
    nearest = np.argmin(ranked, axis=1)
    rows = np.arange(len(d))
    has = np.isfinite(ranked[rows, nearest])

    out = np.zeros(x.shape, dtype=bool)
    out[rows[has], nearest[has]] = True
    return out


def spells(flags, min_len=1):
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]] + 1
    return [(a, b) for a, b in zip(starts.tolist(), stops.tolist())
            if b - a >= min_len]


def analyse(home, away):
    ball = tuple(home[f"ball_{c}"].to_numpy(dtype=float)
                 for c in ("x", "y", "vx", "vy"))

    frame_carry, frame_free, run_rows = [], [], []
    for team, prefix in ((home, "Home"), (away, "Away")):
        ids, x, y, vx, vy, s = stack_team(team, prefix)
        poss = possession_mask(x, y, vx, vy, ball)

        for j, pid in enumerate(ids):
            speed = s[:, j]
            ok = np.isfinite(speed) & (speed <= MAX_SPEED)

            # anchor 1: every frame, split by possession
            frame_carry.append(speed[ok & poss[:, j]])
            frame_free.append(speed[ok & ~poss[:, j]])

            # anchor 2: high-intensity runs, classified by possession fraction
            for a, b in spells(ok & (speed >= RUN_SPEED_MIN), RUN_MIN_FRAMES):
                frac = float(poss[a:b, j].mean())
                if frac >= CARRY_RUN_FRAC:
                    kind = "carry"
                elif frac <= FREE_RUN_FRAC:
                    kind = "free"
                else:
                    continue
                run_rows.append({"player": pid, "kind": kind,
                                 "frames": b - a, "duration_s": (b - a) / FPS,
                                 "peak_speed": float(np.nanmax(speed[a:b])),
                                 "mean_speed": float(np.nanmean(speed[a:b])),
                                 "possession_frac": frac})

    return (np.concatenate(frame_carry), np.concatenate(frame_free),
            pd.DataFrame(run_rows))


def per_player(runs):
    rows = []
    for pid, g in runs.groupby("player"):
        c = g[g["kind"] == "carry"]["peak_speed"]
        f = g[g["kind"] == "free"]["peak_speed"]
        if len(c) < MIN_RUNS_PER_PLAYER or len(f) < MIN_RUNS_PER_PLAYER:
            continue
        rows.append({"player": pid, "carry_runs": len(c), "free_runs": len(f),
                     "carry_peak": float(c.quantile(ANCHOR_Q)),
                     "free_peak": float(f.quantile(ANCHOR_Q)),
                     "carry_median": float(c.median()),
                     "free_median": float(f.median()),
                     "ratio": float(c.quantile(ANCHOR_Q) / f.quantile(ANCHOR_Q))})
    return pd.DataFrame(rows)


def figures(fc, ff, runs, per, fitted):
    os.makedirs(OUT_DIR, exist_ok=True)
    carry = runs[runs["kind"] == "carry"]["peak_speed"].to_numpy()
    free = runs[runs["kind"] == "free"]["peak_speed"].to_numpy()

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.7))

    bins = np.linspace(0, 11, 55)
    ax[0].hist(ff, bins=bins, density=True, alpha=0.55, color="#4C72B0",
               label=f"free frames (n={ff.size:,})")
    ax[0].hist(fc, bins=bins, density=True, alpha=0.55, color="#C44E52",
               label=f"carry frames (n={fc.size:,})")
    ax[0].set_xlabel("speed (m/s)")
    ax[0].set_ylabel("density")
    ax[0].set_title("Anchor 1, every frame: FAILS\n"
                    "free is mostly walking, so carrying looks faster")
    ax[0].legend(fontsize=8)

    bins = np.linspace(RUN_SPEED_MIN, 11, 45)
    ax[1].hist(free, bins=bins, density=True, alpha=0.55, color="#4C72B0",
               label=f"free runs (n={free.size})")
    ax[1].hist(carry, bins=bins, density=True, alpha=0.55, color="#C44E52",
               label=f"carry runs (n={carry.size})")
    for arr, col in ((free, "#4C72B0"), (carry, "#C44E52")):
        ax[1].axvline(np.quantile(arr, ANCHOR_Q), color=col, ls="--", lw=1.5)
    ax[1].set_xlabel("peak speed of the run (m/s)")
    ax[1].set_ylabel("density")
    ax[1].set_title(f"Anchor 2, matched runs (>= {RUN_SPEED_MIN} m/s)\n"
                    f"dashed = p{int(ANCHOR_Q*100)}")
    ax[1].legend(fontsize=8)

    ax[2].scatter(per["free_peak"], per["carry_peak"], s=40, alpha=0.85,
                  color="#55A868", edgecolor="k", linewidth=0.4)
    lo = min(per["free_peak"].min(), per["carry_peak"].min()) - 0.4
    hi = max(per["free_peak"].max(), per["carry_peak"].max()) + 0.4
    ax[2].plot([lo, hi], [lo, hi], "k:", lw=1, label="parity")
    ax[2].plot([lo, hi], [fitted * lo, fitted * hi], "-", color="#C44E52",
               lw=1.8, label=f"fitted {fitted:.2f}")
    ax[2].plot([lo, hi], [OLD_GUESS * lo, OLD_GUESS * hi], "--", color="grey",
               lw=1.4, label=f"previous guess {OLD_GUESS}")
    ax[2].set_xlim(lo, hi); ax[2].set_ylim(lo, hi)
    ax[2].set_xlabel(f"free-run peak, p{int(ANCHOR_Q*100)} (m/s)")
    ax[2].set_ylabel(f"carry-run peak, p{int(ANCHOR_Q*100)} (m/s)")
    ax[2].set_title(f"Per player, n={len(per)}")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dribble_speed_fit.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for kind, col in (("free", "#4C72B0"), ("carry", "#C44E52")):
        g = runs[runs["kind"] == kind]
        ax[0].hist(g["duration_s"], bins=np.arange(0.3, 6.0, 0.2), alpha=0.6,
                   density=True, color=col, label=kind)
    ax[0].set_xlabel("run duration (s)")
    ax[0].set_ylabel("density")
    ax[0].set_title("Run lengths are comparable,\nso the peaks are too")
    ax[0].legend()

    qs = np.linspace(0.05, 0.99, 40)
    ax[1].plot(qs, np.quantile(carry, qs) / np.quantile(free, qs), lw=2,
               color="#8172B2", label="matched runs")
    ax[1].plot(qs, np.quantile(fc, qs) / np.quantile(ff, qs), lw=1.6,
               color="grey", ls=":", label="every frame (biased)")
    ax[1].axhline(fitted, color="#C44E52", lw=1.5, label=f"fitted {fitted:.2f}")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].axvline(ANCHOR_Q, color="k", ls=":", lw=1)
    ax[1].set_xlabel("quantile")
    ax[1].set_ylabel("carry / free")
    ax[1].set_title("The frame-wise anchor sits above 1.0\nat every quantile")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dribble_runs.png"), dpi=140)
    plt.close(fig)


def main():
    home, away = load()
    fc, ff, runs = analyse(home, away)
    per = per_player(runs)

    carry = runs[runs["kind"] == "carry"]["peak_speed"].to_numpy()
    free = runs[runs["kind"] == "free"]["peak_speed"].to_numpy()

    print("\n1. ANCHOR 1 -- every frame, split by possession. THIS ONE FAILS.")
    print(f"   carry frames {fc.size:,}   free frames {ff.size:,}")
    print(f"   {'quantile':>10} {'carrying':>10} {'free':>8} {'ratio':>8}")
    for q in QUANTILES:
        a, b = np.quantile(fc, q), np.quantile(ff, q)
        print(f"   {q:10.2f} {a:10.2f} {b:8.2f} {a/b:8.3f}")
    print("   Every ratio is above 1.0, i.e. 'carrying makes you faster'. It is")
    print("   selection bias: carrying only happens while active, whereas the")
    print("   free sample is a whole match and is mostly walking.")

    print(f"\n2. ANCHOR 2 -- peak speed of runs over {RUN_SPEED_MIN} m/s. USED.")
    print(f"   carry runs {len(carry)}   free runs {len(free)}   "
          f"players {len(per)}")
    print(f"   {'quantile':>10} {'carrying':>10} {'free':>8} {'ratio':>8}")
    for q in QUANTILES:
        a, b = np.quantile(carry, q), np.quantile(free, q)
        print(f"   {q:10.2f} {a:10.2f} {b:8.2f} {a/b:8.3f}")

    pooled = float(np.quantile(carry, ANCHOR_Q) / np.quantile(free, ANCHOR_Q))
    median_player = float(per["ratio"].median())
    fitted = round(median_player, 2)
    print(f"\n   pooled p{int(ANCHOR_Q*100)} ratio     {pooled:.3f}")
    print(f"   median per-player ratio {median_player:.3f}  "
          f"(IQR {per['ratio'].quantile(.25):.3f}-{per['ratio'].quantile(.75):.3f})")
    print(f"   carry-run peak p{int(ANCHOR_Q*100)} {np.quantile(carry, ANCHOR_Q):.2f} m/s vs "
          f"free {np.quantile(free, ANCHOR_Q):.2f} m/s")
    print(f"\n-> DRIBBLE_SPEED = {fitted:.2f}   (was {OLD_GUESS}, a guess)")
    print(f"   at engine V_MAX = 5.0 that is a carrier cap of {fitted*5:.2f} m/s")

    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(os.path.join(OUT_DIR, "dribble_runs.csv"), index=False)
    per.to_csv(os.path.join(OUT_DIR, "dribble_speed_per_player.csv"), index=False)
    pd.DataFrame([{
        "anchor_quantile": ANCHOR_Q, "run_speed_min": RUN_SPEED_MIN,
        "possession_radius": POSSESSION_RADIUS, "rel_speed_max": REL_SPEED_MAX,
        "pooled_ratio": pooled, "median_player_ratio": median_player,
        "fitted_dribble_speed": fitted, "previous_guess": OLD_GUESS,
        "n_carry_runs": len(carry), "n_free_runs": len(free),
        "n_players": len(per),
        "frame_anchor_p95_ratio": float(np.quantile(fc, 0.95)
                                        / np.quantile(ff, 0.95)),
    }]).to_csv(os.path.join(OUT_DIR, "dribble_speed_params.csv"), index=False)

    figures(fc, ff, runs, per, fitted)
    print(f"\nwrote {OUT_DIR}\\dribble_speed_fit.png, dribble_runs.png "
          f"and 3 csvs")


if __name__ == "__main__":
    main()
