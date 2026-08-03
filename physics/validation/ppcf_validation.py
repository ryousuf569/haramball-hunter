"""
Validate physics/ppcf.py against Laurie Shaw's implementation of the same model.

Shaw's Metrica_PitchControl.py (from LaurieOnTracking) is an independent
implementation of Spearman's pitch control model. Ours is a from-scratch,
vectorised one. Run both on the same real frames and see whether they agree.

Method:
1. Take N pass events from Metrica's public tracking data.
2. For each one, evaluate both models on the same 50x32 grid of points, with the
   same players, at the same instant.
3. Scatter our value against his, cell by cell. Report R2 and mean absolute
   deviation.

Both models are given the same constants. Rather than typing them twice, the
parameter dict handed to Shaw's code is built directly from the constants in
physics/ppcf.py and physics/tti.py, so the two cannot drift apart.

Metrica_PitchControl.py, the model under test, is used exactly as Shaw wrote it.
His data loaders needed small fixes to run on pandas 2.x.

Data: https://github.com/metrica-sports/sample-data -> physics/validation/data/
Run once with --download to fetch it.

Usage:
    python physics/validation/ppcf_validation.py --download
    python physics/validation/ppcf_validation.py              # 20 frames
    python physics/validation/ppcf_validation.py --frames 40
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(VALIDATION_DIR))
for _p in (REPO_ROOT, VALIDATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import Metrica_IO as mio
import Metrica_PitchControl as mpc

import physics.ppcf as ppcf
import physics.tti as tti
from schema import player_dt

DATA_DIR = os.path.join(VALIDATION_DIR, "data")
OUT_DIR = os.path.join(VALIDATION_DIR, "results")
GAME_ID = 1
FIELD = (106.0, 68.0)   # Metrica's pitch, metres, origin at the centre spot
N_GRID_X = 50           # Shaw's default surface resolution

# warm/cool diverging pair for the surfaces, one hue for density
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "def_att", ["#154a87", BLUE, "#b9c4cf", "#e8e6e1", "#f0bfa8", ORANGE, "#93381a"])
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#f4f6f9", "#b7cfec", BLUE, "#123f74"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": INK_3,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 140,
})


# Shaw's params, every value read from our modules so both run the same numbers
def matched_params():
    p = mpc.default_model_params()

    p["max_player_speed"] = tti.v_max                    # 5 m/s
    p["reaction_time"] = tti.reaction_time               # 0.54 s
    p["tti_sigma"] = tti.intercept_uncertainty           # 0.45 s
    p["lambda_att"] = ppcf.attacker_control_rate         # 4.3
    p["lambda_def"] = ppcf.defender_control_rate         # 4.3 * 1.72
    p["int_dt"] = ppcf.integration_timestep              # 0.08 s
    p["max_int_time"] = ppcf.integration_horizon         # 10 s

    # Shaw gives keepers 3x the control rate; we have no goalkeeper concept
    p["lambda_gk"] = p["lambda_def"]

    # he short-circuits the integral on a big head start; we always integrate
    p["time_to_control_att"] = 1e6
    p["time_to_control_def"] = 1e6

    return p


# Shaw's loaders call Series.idxmax(2), which raises on pandas 2.x; fixed below
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
FILES = ("RawTrackingData_Home_Team", "RawTrackingData_Away_Team", "RawEventsData")


def download_data():
    folder = os.path.join(DATA_DIR, "Sample_Game_%d" % GAME_ID)
    os.makedirs(folder, exist_ok=True)
    for stem in FILES:
        name = "Sample_Game_%d_%s.csv" % (GAME_ID, stem)
        dest = os.path.join(folder, name)
        if os.path.exists(dest):
            print("  have %s" % name)
            continue
        print("  fetching %s" % name)
        urllib.request.urlretrieve(
            "%s/Sample_Game_%d/%s" % (BASE_URL, GAME_ID, name), dest)


# first frame of the second half, Shaw's Period.idxmax(2) made pandas-2 safe
def second_half_index(df):
    return df.index[df.Period == df.Period.max()][0]


# Shaw defaults to Savitzky-Golay (needs scipy); this is his moving-average branch
def calc_player_velocities(team, window=7, maxspeed=12.0):
    ids = np.unique([c[:-2] for c in team.columns if c[:4] in ("Home", "Away")])
    dt = team["Time [s]"].diff()
    half = second_half_index(team)
    kernel = np.ones(window) / window
    for p in ids:
        vx = team[p + "_x"].diff() / dt
        vy = team[p + "_y"].diff() / dt
        speed = np.sqrt(vx ** 2 + vy ** 2)
        vx[speed > maxspeed] = np.nan          # position glitches
        vy[speed > maxspeed] = np.nan
        for part in (slice(None, half), slice(half, None)):
            vx.loc[part] = np.convolve(vx.loc[part], kernel, mode="same")
            vy.loc[part] = np.convolve(vy.loc[part], kernel, mode="same")
        team[p + "_vx"], team[p + "_vy"] = vx, vy
    return team


# player furthest from halfway at kickoff, Shaw's rule made pandas-2 safe
def find_goalkeeper(team):
    x_cols = [c for c in team.columns
              if c[-2:].lower() == "_x" and c[:4] in ("Home", "Away")]
    return team.iloc[0][x_cols].abs().idxmax().split("_")[1]


def load_match():
    if not os.path.isdir(os.path.join(DATA_DIR, "Sample_Game_%d" % GAME_ID)):
        raise SystemExit("No data in %s. Run with --download first." % DATA_DIR)
    home = mio.tracking_data(DATA_DIR, GAME_ID, "Home")
    away = mio.tracking_data(DATA_DIR, GAME_ID, "Away")
    events = mio.read_event_data(DATA_DIR, GAME_ID)
    for f in (home, away, events):
        mio.to_metric_coordinates(f)
    # flip the second half so each team attacks the same way all match
    for f in (home, away, events):
        i = second_half_index(f)
        cols = [c for c in f.columns if c[-1].lower() in ("x", "y")]
        f.loc[i:, cols] *= -1
    home = calc_player_velocities(home)
    away = calc_player_velocities(away)
    return home, away, events, (find_goalkeeper(home), find_goalkeeper(away))


WARMUP = 25   # frames; the velocity smoother has no history before this


# N pass events spread evenly, so the sample is reproducible and spans both halves
def select_events(events, home, away, gk, n):
    passes = events[(events.Type == "PASS")
                    & events["Start X"].notna() & events["Start Y"].notna()]
    half_starts = [home.index[0], second_half_index(home)]
    usable = []
    for eid, row in passes.iterrows():
        frame = int(row["Start Frame"])
        if frame not in home.index or frame not in away.index:
            continue
        # skip the start of a half, where everyone reads as stationary
        if any(0 <= frame - s < WARMUP for s in half_starts):
            continue
        # both keepers must be tracked for the frame to be usable
        if np.isnan(home.loc[frame, "Home_%s_x" % gk[0]]):
            continue
        if np.isnan(away.loc[frame, "Away_%s_x" % gk[1]]):
            continue
        usable.append(eid)
    if len(usable) < n:
        raise SystemExit("only %d usable events, asked for %d" % (len(usable), n))
    picks = np.unique(np.linspace(0, len(usable) - 1, n).round().astype(int))
    return [usable[i] for i in picks]


# cell centres of Shaw's default surface, 50 x 32 cells over the pitch
def build_grid():
    ny = int(N_GRID_X * FIELD[1] / FIELD[0])
    dx, dy = FIELD[0] / N_GRID_X, FIELD[1] / ny
    xgrid = np.arange(N_GRID_X) * dx - FIELD[0] / 2 + dx / 2
    ygrid = np.arange(ny) * dy - FIELD[1] / 2 + dy / 2
    X, Y = np.meshgrid(xgrid, ygrid, indexing="ij")
    return xgrid, ygrid, np.stack([X.ravel(), Y.ravel()], axis=1)


XGRID, YGRID, TARGETS = build_grid()


# Shaw's attacking-team pitch control at each target point
def shaw_ppcf(targets, attacking, defending, params):
    out = np.empty(len(targets))
    for i, t in enumerate(targets):
        # ball_start_pos=None treats the ball as already there, as our model does
        out[i] = mpc.calculate_pitch_control_at_target(
            t, attacking, defending, None, params)[0]
    return out


# our attacking-team pitch control, straight out of physics/ppcf.py
def our_ppcf(targets, players):
    per_player = ppcf.PPCF_grid(targets, players)
    return per_player[:, players["team"] == "attacker"].sum(axis=1)


# Shaw's player objects to our structured array, so both see the same inputs
def to_our_players(attacking, defending):
    everyone = list(attacking) + list(defending)
    arr = np.zeros(len(everyone), dtype=player_dt)
    arr["id"] = np.arange(len(everyone))
    arr["position"] = np.array([p.position for p in everyone], dtype=float)
    arr["velocity"] = np.array([p.velocity for p in everyone], dtype=float)
    arr["team"] = ["attacker"] * len(attacking) + ["defender"] * len(defending)
    return arr


# offside attackers stay in, since we have no offside concept to match
def teams_at_event(events, home, away, gk, eid, params):
    row = events.loc[eid]
    frame = int(row["Start Frame"])
    if row.Team == "Home":
        attacking = mpc.initialise_players(home.loc[frame], "Home", params, gk[0])
        defending = mpc.initialise_players(away.loc[frame], "Away", params, gk[1])
    else:
        attacking = mpc.initialise_players(away.loc[frame], "Away", params, gk[1])
        defending = mpc.initialise_players(home.loc[frame], "Home", params, gk[0])
    return attacking, defending

# R2 uses his values as the reference, so an offset counts against it
def metrics(shaw, ours):
    shaw, ours = np.asarray(shaw), np.asarray(ours)
    residual = ours - shaw
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((shaw - shaw.mean()) ** 2))
    slope, intercept = np.polyfit(shaw, ours, 1)
    return dict(
        n_cells=int(shaw.size),
        r2=1.0 - ss_res / ss_tot,
        pearson_r=float(np.corrcoef(shaw, ours)[0, 1]),
        mad=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(residual ** 2))),
        bias=float(np.mean(residual)),
        p95_abs_dev=float(np.percentile(np.abs(residual), 95)),
        max_abs_dev=float(np.max(np.abs(residual))),
        within_0p05=float(np.mean(np.abs(residual) < 0.05)),
        slope=float(slope),
        intercept=float(intercept),
    )

def fig_scatter(shaw, ours, stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))

    ax = axes[0]
    hb = ax.hexbin(shaw, ours, gridsize=70, cmap=DENSITY, bins="log",
                   mincnt=1, linewidths=0)
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1.2, ls="--", zorder=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Shaw PPCF")
    ax.set_ylabel("our PPCF")
    ax.set_title("Our pitch control vs Shaw's, cell by cell", loc="left", pad=18)
    ax.text(0.0, 1.015, "same frames, same players, same constants",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
    ax.text(0.04, 0.96,
            "R$^2$ = %.4f\nMAD = %.4f\nRMSE = %.4f\nn = %s cells"
            % (stats["r2"], stats["mad"], stats["rmse"],
               format(stats["n_cells"], ",")),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=INK_3, lw=0.6))
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("cells (log)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)

    ax = axes[1]
    ax.hist(ours - shaw, bins=90, color=BLUE, zorder=2)
    ax.axvline(0, color=INK_3, lw=1.0, ls="--", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("our PPCF - Shaw PPCF")
    ax.set_ylabel("cells")
    ax.set_title("Residuals", loc="left", pad=18)
    ax.text(0.0, 1.015, "%.1f%% of cells agree to within 0.05"
            % (100 * stats["within_0p05"]),
            transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
    ax.text(0.03, 0.96, "bias %+.4f\np95 |dev| %.4f\nmax |dev| %.4f"
            % (stats["bias"], stats["p95_abs_dev"], stats["max_abs_dev"]),
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=INK_3, lw=0.6))

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_surfaces(frames, path, n_show=2):
    nx, ny = len(XGRID), len(YGRID)
    extent = [-FIELD[0] / 2, FIELD[0] / 2, -FIELD[1] / 2, FIELD[1] / 2]
    picks = frames[:n_show]
    fig, axes = plt.subplots(len(picks), 3, figsize=(12.5, 3.6 * len(picks)),
                             squeeze=False)
    for r, f in enumerate(picks):
        a = f["shaw"].reshape(nx, ny)
        b = f["ours"].reshape(nx, ny)
        diff = b - a
        lim = max(0.05, float(np.abs(diff).max()))
        panels = [(a, "Shaw", 0, 1), (b, "ours", 0, 1),
                  (diff, "ours - Shaw", -lim, lim)]
        for c, (surf, title, lo, hi) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(surf.T, origin="lower", extent=extent, cmap=DIVERGING,
                           vmin=lo, vmax=hi, interpolation="nearest")
            pos = f["players"]["position"]
            att = f["players"]["team"] == "attacker"
            ax.scatter(pos[att, 0], pos[att, 1], s=22, c=ORANGE,
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.scatter(pos[~att, 0], pos[~att, 1], s=22, c=BLUE, marker="s",
                       edgecolors="white", linewidths=0.7, zorder=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, loc="left")
            if c == 0:
                ax.set_ylabel("frame %d\n%s attacking"
                              % (f["frame"], f["pass_team"]), fontsize=8)
            # columns 0 and 1 share the 0-1 scale, so it is only drawn once
            if c > 0:
                cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.88)
                cb.set_label("PPCF" if c == 1 else "difference",
                             color=INK_2, fontsize=7.5)
                cb.outline.set_visible(False)
                cb.ax.tick_params(labelsize=7)
    fig.suptitle("Pitch control surfaces  (orange = attacking control, "
                 "blue = defending)", x=0.012, ha="left", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Validate our PPCF against Shaw's")
    ap.add_argument("--frames", type=int, default=20,
                    help="number of pass events to evaluate (default 20)")
    ap.add_argument("--download", action="store_true",
                    help="fetch the Metrica sample data first")
    args = ap.parse_args()

    if args.download:
        print("Downloading Metrica sample data...")
        download_data()

    os.makedirs(OUT_DIR, exist_ok=True)
    params = matched_params()

    print("Loading Metrica Sample Game %d..." % GAME_ID)
    home, away, events, gk = load_match()
    event_ids = select_events(events, home, away, gk, args.frames)

    print("\nshared constants")
    print("  max speed        %.2f m/s" % params["max_player_speed"])
    print("  reaction time    %.2f s" % params["reaction_time"])
    print("  tti sigma        %.2f s" % params["tti_sigma"])
    print("  lambda attack    %.2f" % params["lambda_att"])
    print("  lambda defend    %.2f   (kappa %.2f)"
          % (params["lambda_def"], params["lambda_def"] / params["lambda_att"]))
    print("  integration dt   %.2f s" % params["int_dt"])
    print("\nevaluating %d events on a %dx%d grid (%.2f x %.2f m cells)"
          % (len(event_ids), len(XGRID), len(YGRID),
             XGRID[1] - XGRID[0], YGRID[1] - YGRID[0]))

    frames = []
    for i, eid in enumerate(event_ids):
        row = events.loc[eid]
        attacking, defending = teams_at_event(events, home, away, gk, eid, params)
        players = to_our_players(attacking, defending)
        f = dict(event_id=int(eid), frame=int(row["Start Frame"]),
                 pass_team=row.Team, players=players,
                 shaw=shaw_ppcf(TARGETS, attacking, defending, params),
                 ours=our_ppcf(TARGETS, players))
        frames.append(f)
        s = metrics(f["shaw"], f["ours"])
        print("  [%2d/%2d] event %-5d frame %-6d %-4s  R2 %.4f  MAD %.4f"
              % (i + 1, len(event_ids), f["event_id"], f["frame"],
                 f["pass_team"], s["r2"], s["mad"]))

    shaw_all = np.concatenate([f["shaw"] for f in frames])
    ours_all = np.concatenate([f["ours"] for f in frames])
    stats = metrics(shaw_all, ours_all)

    # csv: every cell
    pointwise = pd.DataFrame({
        "event_id": np.repeat([f["event_id"] for f in frames], len(TARGETS)),
        "frame": np.repeat([f["frame"] for f in frames], len(TARGETS)),
        "pass_team": np.repeat([f["pass_team"] for f in frames], len(TARGETS)),
        "x": np.tile(TARGETS[:, 0], len(frames)),
        "y": np.tile(TARGETS[:, 1], len(frames)),
        "ppcf_shaw": shaw_all.round(6),
        "ppcf_ours": ours_all.round(6),
    })
    pointwise["difference"] = (pointwise.ppcf_ours - pointwise.ppcf_shaw).round(6)
    pointwise.to_csv(os.path.join(OUT_DIR, "ppcf_pointwise.csv"), index=False)

    # csv: per frame
    pd.DataFrame([
        dict(event_id=f["event_id"], frame=f["frame"], pass_team=f["pass_team"],
             n_players=len(f["players"]), **metrics(f["shaw"], f["ours"]))
        for f in frames
    ]).to_csv(os.path.join(OUT_DIR, "ppcf_per_frame.csv"), index=False)

    # csv: overall
    pd.DataFrame([dict(n_frames=len(frames), **stats)]).to_csv(
        os.path.join(OUT_DIR, "ppcf_summary.csv"), index=False)

    fig_scatter(shaw_all, ours_all, stats,
                os.path.join(OUT_DIR, "ppcf_scatter.png"))
    fig_surfaces(frames, os.path.join(OUT_DIR, "ppcf_surfaces.png"))

    print("\n" + "=" * 60)
    print("%d frames, %s cells" % (len(frames), format(stats["n_cells"], ",")))
    print("  R2                 %.4f" % stats["r2"])
    print("  mean abs deviation %.4f" % stats["mad"])
    print("  RMSE               %.4f" % stats["rmse"])
    print("  bias (ours-Shaw)   %+.4f" % stats["bias"])
    print("  p95 |deviation|    %.4f" % stats["p95_abs_dev"])
    print("  max |deviation|    %.4f" % stats["max_abs_dev"])
    print("  within 0.05        %.1f%% of cells" % (100 * stats["within_0p05"]))
    print("  best fit           ours = %.3f * shaw %+.3f"
          % (stats["slope"], stats["intercept"]))
    print("=" * 60)
    print("\ncsv + png written to %s" % OUT_DIR)


if __name__ == "__main__":
    main()
