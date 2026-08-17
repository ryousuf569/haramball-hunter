import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CAL_DIR))
VALIDATION_DIR = os.path.join(REPO_ROOT, "physics", "validation")
for _p in (REPO_ROOT, VALIDATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import Metrica_IO as mio          # noqa: E402
# Metrica_Velocities is not imported: it needs scipy for its Savitzky-Golay
# smoother and scipy is not a project dependency. add_velocities below does a
# centred difference over the same window instead, which is enough here -- the
# only thing velocity is used for is the SIGN of the closing speed.

DATA_DIR = os.path.join(VALIDATION_DIR, "data")
GAME_ID = 1
FIELD = (106.0, 68.0)   # Metrica's pitch. Rescaled to the sim's 105 on the x axis.
PITCH_X = 105.0
PITCH_Y = 68.0
FPS = 25

FRAMES_CSV = os.path.join(CAL_DIR, "low_block_frames.csv")
PLAYERS_CSV = os.path.join(CAL_DIR, "low_block_defenders.csv")

# --- what counts as a low block ------------------------------------------------
# The StatsBomb set is already filtered (README 1). This is the depth band those
# frames occupy, measured in build_statsbomb_table and used to filter Metrica down
# to the same phase of play. Overwritten at runtime from the data.
BLOCK_DEPTH_MIN_SIM = 59.2        # StatsBomb q05 of back_line_x, sim frame
MIN_POSSESSION_S = 5.0            # drop the chaotic seconds after a turnover
MIN_DEFENDERS = 8                 # enough of the block tracked to read its shape
MIN_ATTACKERS = 3

# --- measurement parameters ----------------------------------------------------
ENGAGE_R = 3.0     # a defender this close to the ball is engaging it
NEAR_R = 5.0       # "committed to the ball" radius, for counting how many step out
BLOCK_R = 10.0     # loose ring: shape support rather than commitment
X_BIN = 5.0        # bin width for every profile below
X_LO, X_HI = 45.0, 100.0   # absolute ball-x range, used only for the artifact panel

# Profiles are conditioned on where the ball is RELATIVE TO THE BLOCK, not on
# absolute ball x. Absolute x is a trap here: the frames with the ball deepest
# are overwhelmingly balls already played THROUGH the block (median 19m goal-side
# of the back line at ball_x > 90), so nobody is near the ball and the naive
# profile reads as "blocks press less near their own box", which is backwards.
# rel_back / rel_mid are signed metres goal-side of that line: negative means the
# ball is still in front of it.
REL_LO, REL_HI, REL_BIN = -30.0, 25.0, 5.0

DEF_LINE_SPLIT = 5  # deepest N outfielders are read as the back line

VEL_WINDOW = 5      # frames (0.2s) for the centred difference. Metrica smooths over 7.
MAX_SPEED = 12.0    # m/s; anything faster is a tracking glitch, not a player

HYPOTHESIS_X = 70.0   # the "only press when the ball is past x=70" proposal


# ===========================================================================
# 1. StatsBomb low-block freeze frames -- the geometry
# ===========================================================================
def build_statsbomb_table():
    """One row per low-block freeze frame, in the sim frame.

    low_block_defenders.csv holds one row per visible player with the line it was
    assigned ('back' / 'mid' for the block, 'att' for the attacking team). The
    keeper is already excluded from both defensive lines by
    calibration_graphing.py, so 'nearest defender' here never means the keeper --
    which matters, because the keeper is nearest to everything once the ball
    reaches the six-yard box.
    """
    players = pd.read_csv(PLAYERS_CSV)
    frames = pd.read_csv(FRAMES_CSV)

    d = players[players["line"].isin(["back", "mid"])].copy()
    d["dist"] = np.hypot(d["x"] - d["ball_x"], d["y"] - d["ball_y"])

    g = d.groupby("event_id")
    per = pd.DataFrame({
        "d1": g["dist"].min(),
        "n_near": g["dist"].apply(lambda s: int((s <= NEAR_R).sum())),
        "n_block": g["dist"].apply(lambda s: int((s <= BLOCK_R).sum())),
        "n_def": g.size(),
    })
    # Line the nearest defender belongs to, and how far it is from its own line's x
    nearest = d.loc[g["dist"].idxmin(), ["event_id", "line", "x", "dist"]]
    per["presser_line"] = nearest.set_index("event_id")["line"]
    per["presser_x"] = nearest.set_index("event_id")["x"]

    per = per.join(frames.set_index("event_id")[
        ["ball_x", "ball_y", "back_line_x", "mid_line_x", "back_gap_max",
         "back_width", "highest_att_x", "att_line_x"]], how="inner")

    # calibration frame (x=0 at the defended goal) -> sim frame
    for col in ("ball_x", "back_line_x", "mid_line_x", "highest_att_x",
                "att_line_x", "presser_x"):
        per[col] = PITCH_X - per[col]

    per["rel_back"] = per["ball_x"] - per["back_line_x"]
    per["rel_mid"] = per["ball_x"] - per["mid_line_x"]
    per["excursion"] = per["presser_x"] - per["back_line_x"]
    per["committed"] = per["d1"] <= NEAR_R
    per["engaged"] = per["d1"] <= ENGAGE_R
    return per.reset_index()


# ===========================================================================
# 2. Metrica tracking -- durations and path lengths only
# ===========================================================================
def add_velocities(team, name):
    """vx/vy per player by centred difference, in place.

    Deliberately not Metrica_Velocities.calc_player_velocities: that needs scipy.
    A centred difference over VEL_WINDOW frames is a coarser estimate, but the
    only use here is whether the presser is moving toward the ball or away, and
    that sign is robust to the smoother.
    """
    dt = VEL_WINDOW / FPS
    for c in [c for c in team.columns if c.endswith("_x") and c.startswith(f"{name}_")]:
        base = c[:-2]
        for axis in ("x", "y"):
            s = team[f"{base}_{axis}"]
            team[f"{base}_v{axis}"] = (s.shift(-VEL_WINDOW // 2)
                                       - s.shift(VEL_WINDOW // 2)) / dt
        speed = np.hypot(team[f"{base}_vx"], team[f"{base}_vy"])
        team.loc[speed > MAX_SPEED, [f"{base}_vx", f"{base}_vy"]] = np.nan
    return team


def flip_second_half(team):
    """Rotate the second half 180 degrees so a team attacks one way all match.

    Written out rather than calling mio.to_single_playing_direction, which does
    `team.Period.idxmax(2)` -- an old pandas signature that raises on 2.x. The
    vendored Metrica files stay exactly as they are; this is the one line that
    needed it.
    """
    team = team.copy()
    second = team.index[team["Period"] == 2]
    if len(second):
        cols = [c for c in team.columns if c[-1].lower() in ("x", "y")]
        team.loc[second, cols] *= -1
    return team


def load_match():
    """Tracking for both teams plus events, metric coords, one playing direction.

    Only Team and Start Frame are read off events, so events are left in raw
    coordinates -- nothing downstream uses their x/y.
    """
    home, away, events = mio.read_match_data(DATA_DIR, GAME_ID)
    home = mio.to_metric_coordinates(home, field_dimen=FIELD)
    away = mio.to_metric_coordinates(away, field_dimen=FIELD)
    home = add_velocities(flip_second_half(home), "Home")
    away = add_velocities(flip_second_half(away), "Away")
    return home, away, events


def possession_by_frame(events, index):
    """Team in possession at each tracking frame, forward-filled from events.

    Only on-ball actions mark possession: PASS / RECOVERY / SHOT / SET PIECE are
    all performed BY the team that has it. CHALLENGE and BALL LOST are excluded
    because the team they are credited to is the team WITHOUT the ball after the
    event, so filling from them would invert possession over exactly the turnover
    moments MIN_POSSESSION_S is there to drop.
    """
    on_ball = events[events["Type"].isin(["PASS", "RECOVERY", "SHOT", "SET PIECE"])]
    marks = pd.Series(on_ball["Team"].values,
                      index=on_ball["Start Frame"].values).sort_index()
    marks = marks[~marks.index.duplicated(keep="last")]
    return marks.reindex(index.union(marks.index)).ffill().reindex(index)


def possession_age(poss):
    """Seconds the current possession has been running, per frame."""
    changed = poss.ne(poss.shift()).fillna(True)
    return poss.groupby(changed.cumsum()).cumcount() / FPS


def playing_direction(team, name):
    """+1 if this team attacks +x in the first-half metric frame, else -1.

    Same rule as mio.find_playing_direction -- the keeper is the player standing
    furthest from the centre spot at kick-off, and a team attacks away from its
    own keeper -- reimplemented for the same pandas 2.x reason as flip_second_half.
    """
    first = team.iloc[0]
    xs = {c: first[c] for c in team.columns
          if c.startswith(f"{name}_") and c.endswith("_x") and np.isfinite(first[c])}
    return -np.sign(xs[max(xs, key=lambda c: abs(xs[c]))])


def player_columns(tracking, team):
    """[(x_col, y_col, vx_col, vy_col, label)] for every tracked slot."""
    out = []
    for c in tracking.columns:
        if c.startswith(f"{team}_") and c.endswith("_x"):
            base = c[:-2]
            if f"{base}_y" in tracking.columns and f"{base}_vx" in tracking.columns:
                out.append((c, f"{base}_y", f"{base}_vx", f"{base}_vy", base))
    return out


def to_sim_frame(x, y, attack_dir):
    """Metrica centre-origin metres -> sim frame, defenders defending x=105.

    Rotating 180 degrees (both axes) rather than mirroring x alone keeps left and
    right the same way round, which matters for the width and gap statistics.
    """
    return ((attack_dir * x + FIELD[0] / 2.0) * (PITCH_X / FIELD[0]),
            attack_dir * y + FIELD[1] / 2.0)


def build_metrica_table(home, away, events, depth_min):
    """Per-frame press measurements over Metrica's low-block phases only."""
    poss = possession_by_frame(events, home.index)
    age = possession_age(poss)
    dir_home = playing_direction(home, "Home")
    tracking = {"Home": home, "Away": away}
    cols = {t: player_columns(tracking[t], t) for t in ("Home", "Away")}
    ball_x_m, ball_y_m = home["ball_x"].values, home["ball_y"].values

    rows = []
    for att_team, def_team in (("Home", "Away"), ("Away", "Home")):
        attack_dir = dir_home if att_team == "Home" else -dir_home
        sel = ((poss.values == att_team) & (age.values >= MIN_POSSESSION_S)
               & np.isfinite(ball_x_m) & np.isfinite(ball_y_m))
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        bx, by = to_sim_frame(ball_x_m[idx], ball_y_m[idx], attack_dir)

        def stack(team, kind):
            c = cols[team]
            if kind == "pos":
                xs = np.column_stack([tracking[team][a].values[idx] for a, _, _, _, _ in c])
                ys = np.column_stack([tracking[team][b].values[idx] for _, b, _, _, _ in c])
                return to_sim_frame(xs, ys, attack_dir)
            vx = np.column_stack([tracking[team][a].values[idx] for _, _, a, _, _ in c])
            vy = np.column_stack([tracking[team][b].values[idx] for _, _, _, b, _ in c])
            # velocities rotate with the frame but take no origin shift
            return attack_dir * vx * (PITCH_X / FIELD[0]), attack_dir * vy

        dx, dy = stack(def_team, "pos")
        dvx, dvy = stack(def_team, "vel")
        ax_, ay_ = stack(att_team, "pos")
        labels = np.array([lab for _, _, _, _, lab in cols[def_team]])
        rows.append(measure_metrica(idx, bx, by, dx, dy, dvx, dvy, ax_, ay_,
                                    labels, def_team, att_team, depth_min))
    return pd.concat(rows, ignore_index=True)


def measure_metrica(idx, bx, by, dx, dy, dvx, dvy, ax_, ay_, labels,
                    def_team, att_team, depth_min):
    n = len(idx)
    d_ok = np.isfinite(dx) & np.isfinite(dy)
    a_ok = np.isfinite(ax_) & np.isfinite(ay_)

    # The keeper is the defender deepest in x. It never presses, and leaving it in
    # would make "nearest defender" meaningless once the ball reaches the box.
    gk = np.argmax(np.where(d_ok, dx, -np.inf), axis=1)
    outfield = d_ok.copy()
    outfield[np.arange(n), gk] = False

    d_ball = np.where(outfield, np.hypot(dx - bx[:, None], dy - by[:, None]), np.inf)
    presser = np.argmin(d_ball, axis=1)
    d1 = d_ball[np.arange(n), presser]

    # Closing speed: component of the presser's velocity along the line to the
    # ball. Positive means actually going at it, which is what separates a press
    # from a defender the ball happened to be passed near.
    px, py = dx[np.arange(n), presser], dy[np.arange(n), presser]
    ux, uy = bx - px, by - py
    un = np.hypot(ux, uy)
    closing = (dvx[np.arange(n), presser] * ux
               + dvy[np.arange(n), presser] * uy) / np.where(un > 1e-6, un, np.nan)

    dx_of = np.where(outfield, dx, np.nan)
    order = np.argsort(np.where(np.isfinite(dx_of), dx_of, np.inf), axis=1)
    deep_first = order[:, ::-1]
    back_cols = deep_first[:, :DEF_LINE_SPLIT]           # deepest = largest x
    mid_cols = deep_first[:, DEF_LINE_SPLIT:DEF_LINE_SPLIT + 4]
    back_x = np.nanmean(dx_of[np.arange(n)[:, None], back_cols], axis=1)
    mid_x = np.nanmean(dx_of[np.arange(n)[:, None], mid_cols], axis=1)

    df = pd.DataFrame({
        "frame": idx, "def_team": def_team, "att_team": att_team,
        "ball_x": bx, "ball_y": by,
        "d1": d1, "closing": closing,
        "n_near": (d_ball <= NEAR_R).sum(axis=1),
        "n_def": outfield.sum(axis=1), "n_att": a_ok.sum(axis=1),
        "presser": labels[presser], "presser_x": px, "presser_y": py,
        "back_line_x": back_x, "mid_line_x": mid_x,
        "rel_back": bx - back_x, "rel_mid": bx - mid_x,
    })

    keep = ((df["ball_x"] >= X_LO) & (df["back_line_x"] >= depth_min)
            & (df["n_def"] >= MIN_DEFENDERS) & (df["n_att"] >= MIN_ATTACKERS)
            & np.isfinite(df["d1"]))
    return df[keep].reset_index(drop=True)


def press_spells(df):
    """Contiguous runs where one defender is committed to the ball.

    Commitment is `nearest outfielder AND inside NEAR_R`, not the tighter
    engaged-and-closing test: at 3m with a positive closing speed the flag
    chatters frame to frame as a carrier turns, which chops one real press into a
    dozen 0.2s fragments and makes the duration meaningless. 5m held by the same
    player is the quantity the sim's latch actually models.

    A spell breaks on a frame gap, a team change, or a different player taking
    the press on. Spells are kept only if they contain at least one genuinely
    engaged frame, which is what separates a press from a defender the ball
    happened to roll past.
    """
    near = df[df["d1"] <= NEAR_R].sort_values(["def_team", "frame"]).reset_index(drop=True)
    brk = (near["presser"].ne(near["presser"].shift())
           | near["def_team"].ne(near["def_team"].shift())
           | (near["frame"].diff() != 1))
    near = near.assign(spell=brk.cumsum(),
                       engaged=(near["d1"] <= ENGAGE_R) & (near["closing"] > 0))
    g = near.groupby("spell")

    # Ground actually covered by the presser during the spell -- the excursion
    # budget the scripted press has to be held to.
    path = g.apply(lambda s: float(np.nansum(np.hypot(s["presser_x"].diff(),
                                                      s["presser_y"].diff()))),
                   include_groups=False)
    spells = pd.DataFrame({
        "ticks": g.size().values,
        "seconds": g.size().values / FPS,
        "ball_x": g["ball_x"].mean().values,
        "presser": g["presser"].first().values,
        "path_m": path.values,
        "net_m": np.hypot(g["presser_x"].last().values - g["presser_x"].first().values,
                          g["presser_y"].last().values - g["presser_y"].first().values),
        "any_engaged": g["engaged"].any().values,
    })
    return spells[spells["any_engaged"]].reset_index(drop=True)


# ===========================================================================
# 3. Profiles and figures
# ===========================================================================
def binned(df, value, agg="median", min_n=30, x="rel_back", bins=None):
    b = bins if bins is not None else np.arange(REL_LO, REL_HI + REL_BIN, REL_BIN)
    g = df.groupby(pd.cut(df[x], b), observed=True)[value]
    centres = np.array([iv.mid for iv in g.groups.keys()])
    n = g.size().values
    stat = (g.median() if agg == "median" else g.mean()).values
    lo = g.quantile(0.25).values if agg == "median" else None
    hi = g.quantile(0.75).values if agg == "median" else None
    ok = n >= min_n
    if lo is None:
        return centres[ok], stat[ok], None, None, n[ok]
    return centres[ok], stat[ok], lo[ok], hi[ok], n[ok]


def band_edges(sb, floor=0.5):
    """The rel_back window inside which committing a defender is the norm.

    Read as the outermost bin edges whose commitment rate clears `floor`. This is
    the whole policy in one line: a real block steps out when the ball is inside a
    band around its own lines, and stays home outside it -- both when the ball is
    still too far in front to be worth chasing, and when it has already gone
    through, which is the keeper's problem rather than the block's.
    """
    c, rate, _, _, _ = binned(sb, "committed", agg="mean")
    ok = np.flatnonzero(rate >= floor)
    if not ok.size:
        return np.nan, np.nan, c, rate
    return (float(c[ok[0]] - REL_BIN / 2.0), float(c[ok[-1]] + REL_BIN / 2.0), c, rate)


def apply_band(df, front, behind):
    """The candidate rule, as defenders.py will evaluate it."""
    return (df["rel_back"] >= front) & (df["rel_back"] <= behind)


def fig_engagement(sb, mt, out):
    """How close does the ball get before anyone steps out?"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    front, behind, c, rate = band_edges(sb)

    cd, med, lo, hi, _ = binned(sb, "d1")
    axes[0].fill_between(cd, lo, hi, alpha=0.25, color="tab:red", label="IQR")
    axes[0].plot(cd, med, "o-", color="tab:red", label="median")
    axes[0].axhline(ENGAGE_R, ls="--", c="0.4", lw=1, label=f"engaged < {ENGAGE_R:.0f}m")
    axes[0].axhline(NEAR_R, ls=":", c="0.4", lw=1, label=f"committed < {NEAR_R:.0f}m")
    axes[0].axvspan(front, behind, color="tab:green", alpha=0.12)
    axes[0].axvline(0, c="0.6", lw=1)
    axes[0].set_xlabel("ball position relative to the back line (m, + = goal-side)")
    axes[0].set_ylabel("nearest block defender to ball (m)")
    axes[0].set_title("StatsBomb low blocks: engagement distance")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(c, 100 * rate, "o-", color="tab:green", label="vs back line")
    cm, ratem, _, _, _ = binned(sb, "committed", agg="mean", x="rel_mid")
    axes[1].plot(cm, 100 * ratem, "s--", color="tab:orange", label="vs midfield line")
    axes[1].axhline(50, ls="--", c="0.4", lw=1)
    axes[1].axvspan(front, behind, color="tab:green", alpha=0.12)
    axes[1].set_xlabel("ball position relative to that line (m, + = goal-side)")
    axes[1].set_ylabel(f"% of frames with a defender inside {NEAR_R:.0f}m")
    axes[1].set_title(f"The press band: {front:+.0f}m to {behind:+.0f}m off the back line")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # The trap. Same data, absolute x, and the trend inverts.
    abs_bins = np.arange(X_LO, X_HI + X_BIN, X_BIN)
    ca, meda, _, _, _ = binned(sb, "d1", x="ball_x", bins=abs_bins)
    cr, rela, _, _, _ = binned(sb, "rel_back", agg="mean", x="ball_x", bins=abs_bins)
    axes[2].plot(ca, meda, "o-", color="tab:red", label="nearest defender (m)")
    axes[2].plot(cr, rela, "s--", color="tab:blue", label="ball rel. back line (m)")
    axes[2].axhline(0, c="0.6", lw=1)
    axes[2].set_xlabel("absolute ball x (sim frame)")
    axes[2].set_ylabel("metres")
    axes[2].set_title("Why absolute x is the wrong axis:\ndeep balls are balls already played through")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(f"Press engagement ({len(sb):,} StatsBomb low-block frames)", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_commitment(sb, mt, out):
    """How many defenders leave the block at once?"""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    front, behind, _, _ = band_edges(sb)

    c, near, _, _, _ = binned(sb, "n_near", agg="mean")
    c2, block, _, _, _ = binned(sb, "n_block", agg="mean")
    axes[0].plot(c, near, "o-", color="tab:red", label=f"within {NEAR_R:.0f}m of ball")
    axes[0].plot(c2, block, "s-", color="tab:orange", label=f"within {BLOCK_R:.0f}m of ball")
    axes[0].axhline(1.0, ls="--", c="0.4", lw=1)
    axes[0].axvspan(front, behind, color="tab:green", alpha=0.12)
    axes[0].set_xlabel("ball position relative to the back line (m)")
    axes[0].set_ylabel("mean block defenders")
    axes[0].set_title("Exactly one defender commits, inside the band")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    inb = sb[apply_band(sb, front, behind)]
    counts = inb["n_near"].value_counts(normalize=True).sort_index()
    counts = counts[counts.index <= 4]
    axes[1].bar(counts.index, 100 * counts.values, color="tab:purple", alpha=0.85)
    axes[1].set_xlabel(f"block defenders within {NEAR_R:.0f}m of the ball")
    axes[1].set_ylabel("% of in-band frames")
    axes[1].set_title("One at a time, inside the band")
    axes[1].grid(alpha=0.25, axis="y")
    for i, v in zip(counts.index, counts.values):
        axes[1].annotate(f"{100*v:.1f}%", (i, 100 * v), ha="center", va="bottom", fontsize=8)

    # Cross-check: does the Metrica low-block subset describe the same phase?
    cs, meds, _, _, _ = binned(sb, "d1")
    cmm, medm, _, _, _ = binned(mt, "d1")
    axes[2].plot(cs, meds, "o-", color="tab:red", label=f"StatsBomb (n={len(sb):,})")
    axes[2].plot(cmm, medm, "s--", color="tab:blue", label=f"Metrica low block (n={len(mt):,})")
    axes[2].axvspan(front, behind, color="tab:green", alpha=0.12)
    axes[2].set_xlabel("ball position relative to the back line (m)")
    axes[2].set_ylabel("median nearest defender to ball (m)")
    axes[2].set_title("Cross-check: same phase, so durations transfer")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle("How much of the block commits to the ball", y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_shape(sb, out):
    """Which line supplies the presser, and what the block pays for it."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    front, behind, _, _ = band_edges(sb)

    bins = np.arange(REL_LO, REL_HI + REL_BIN, REL_BIN)
    cut = pd.cut(sb["rel_back"], bins)
    share = sb.groupby([cut, "presser_line"], observed=True).size().unstack(fill_value=0)
    share = share[share.sum(axis=1) >= 30]
    frac = share.div(share.sum(axis=1), axis=0)
    centres = np.array([iv.mid for iv in frac.index])
    bottom = np.zeros(len(centres))
    for col, colour in (("back", "tab:blue"), ("mid", "tab:orange")):
        if col not in frac:
            continue
        axes[0].bar(centres, 100 * frac[col].values, width=REL_BIN * 0.85,
                    bottom=100 * bottom, label=f"{col} line", color=colour, alpha=0.9)
        bottom += frac[col].values
    axes[0].set_xlabel("ball position relative to the back line (m)")
    axes[0].set_ylabel("% of frames")
    axes[0].set_title("The press comes from the line the ball is in")
    axes[0].legend(fontsize=8, loc="lower right")

    c, med, lo, hi, _ = binned(sb, "back_gap_max")
    axes[1].fill_between(c, lo, hi, alpha=0.25, color="tab:red")
    axes[1].plot(c, med, "o-", color="tab:red")
    axes[1].axvspan(front, behind, color="tab:green", alpha=0.12)
    axes[1].set_xlabel("ball position relative to the back line (m)")
    axes[1].set_ylabel("largest y-gap in the back line (m)")
    axes[1].set_title("The back line does not open up in the press band")
    axes[1].grid(alpha=0.25)

    inb = apply_band(sb, front, behind)
    parts = [sb.loc[inb & ~sb["committed"], "back_gap_max"].dropna(),
             sb.loc[inb & sb["committed"], "back_gap_max"].dropna()]
    axes[2].boxplot(parts, tick_labels=["holding shape", "committed"], showfliers=False)
    axes[2].set_ylabel("largest y-gap in the back line (m)")
    axes[2].set_title("Committing costs the block no shape")
    axes[2].grid(alpha=0.25, axis="y")

    fig.suptitle("A real press is a short step out of the nearest line", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_duration(spells, out):
    """How long one commitment lasts and how far it runs (Metrica low block)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].hist(spells["seconds"], bins=np.arange(0, 5.0, 0.2),
                 color="tab:blue", alpha=0.85)
    for qv, lab, style in ((0.5, "median", "-"), (0.9, "p90", "--")):
        v = spells["seconds"].quantile(qv)
        axes[0].axvline(v, ls=style, c="k", lw=1.2, label=f"{lab} {v:.2f}s")
    axes[0].set_xlabel("length of one defender's commitment (s)")
    axes[0].set_ylabel("spells")
    axes[0].set_title(f"Commitments are short ({len(spells):,} spells)")
    axes[0].legend(fontsize=8)

    axes[1].hist(spells["path_m"].dropna(), bins=np.arange(0, 25, 0.75),
                 color="tab:green", alpha=0.85)
    for qv, lab, style in ((0.5, "median", "-"), (0.9, "p90", "--"), (0.95, "p95", ":")):
        v = spells["path_m"].quantile(qv)
        axes[1].axvline(v, ls=style, c="k", lw=1.2, label=f"{lab} {v:.1f}m")
    axes[1].set_xlabel("ground covered by the presser during one commitment (m)")
    axes[1].set_ylabel("spells")
    axes[1].set_title("A press is a few metres, not a chase")
    axes[1].legend(fontsize=8)

    fig.suptitle("Commitment length and distance -- Metrica low-block phases "
                 "(freeze frames can measure neither)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 4. Derived parameters
# ===========================================================================
def derive_params(sb, mt, spells, depth_min):
    front, behind, _, _ = band_edges(sb)
    inb = apply_band(sb, front, behind)
    band, outside = sb[inb], sb[~inb]
    line_sep = (sb["back_line_x"] - sb["mid_line_x"]).median()

    # What the proposed absolute gate would have done, on the same frames.
    hyp = sb["ball_x"] >= HYPOTHESIS_X

    def r(x, k=2):
        return round(float(x), k)

    rows = [
        ("statsbomb", "n_frames", len(sb), "low-block freeze frames"),
        ("statsbomb", "block_depth_q05_sim", r(depth_min, 1), "back line x, q05 -- the low-block depth band"),
        ("statsbomb", "block_depth_median_sim", r(sb["back_line_x"].median(), 1), "back line x, median"),
        ("statsbomb", "line_sep_median_m", r(line_sep), "back line to midfield line"),

        ("statsbomb", "press_band_front_m", r(front, 1), "band starts this many m in FRONT of the back line (negative)"),
        ("statsbomb", "press_band_behind_m", r(behind, 1), "band ends this many m GOAL-SIDE of the back line"),
        ("statsbomb", "press_band_front_vs_mid_m", r(front + line_sep, 1), "same front edge, measured off the midfield line"),
        ("statsbomb", "commit_rate_in_band", r(band["committed"].mean(), 4), "fraction with a defender inside NEAR_R, ball in band"),
        ("statsbomb", "commit_rate_out_of_band", r(outside["committed"].mean(), 4), "same, ball outside the band"),
        ("statsbomb", "frac_frames_in_band", r(inb.mean(), 4), "how much of a low-block possession the band covers"),

        ("statsbomb", "commit_rate_absolute_x_above_70", r(sb.loc[hyp, "committed"].mean(), 4), "what a plain absolute x>70 gate would select"),
        ("statsbomb", "commit_rate_absolute_x_below_70", r(sb.loc[~hyp, "committed"].mean(), 4), "and what it would reject -- note it is the wrong way round"),

        ("statsbomb", "d1_median_in_band", r(band["d1"].median()), "nearest block defender to ball, in band (m)"),
        ("statsbomb", "d1_median_out_of_band", r(outside["d1"].median()), "same, out of band (m)"),
        ("statsbomb", "n_near_mean_in_band", r(band["n_near"].mean(), 3), "mean defenders within NEAR_R, in band"),
        ("statsbomb", "frac_zero_near_in_band", r((band["n_near"] == 0).mean(), 4), "in-band frames with NOBODY inside NEAR_R"),
        ("statsbomb", "frac_one_near_in_band", r((band["n_near"] == 1).mean(), 4), "in-band frames with exactly one"),
        ("statsbomb", "frac_two_or_more_near_in_band", r((band["n_near"] >= 2).mean(), 4), "in-band frames with 2+"),

        ("statsbomb", "presser_share_back_in_band", r((band["presser_line"] == "back").mean(), 4), "nearest defender is a back-line player, in band"),
        ("statsbomb", "presser_share_mid_in_band", r((band["presser_line"] == "mid").mean(), 4), "nearest defender is a midfielder, in band"),
        ("statsbomb", "back_gap_max_holding_m", r(sb.loc[inb & ~sb["committed"], "back_gap_max"].median()), "largest back-line y-gap, in band, nobody committed"),
        ("statsbomb", "back_gap_max_committed_m", r(sb.loc[inb & sb["committed"], "back_gap_max"].median()), "largest back-line y-gap, in band, someone committed"),

        ("metrica", "n_frames", len(mt), "low-block-depth tracking frames"),
        ("metrica", "d1_median_in_band", r(mt.loc[apply_band(mt, front, behind), "d1"].median()), "cross-check against the StatsBomb figure above (m)"),
        ("metrica", "n_spells", len(spells), "one-defender commitments containing a real engagement"),
        ("metrica", "spell_median_s", r(spells["seconds"].median()), "length of one defender's commitment"),
        ("metrica", "spell_p90_s", r(spells["seconds"].quantile(0.90)), "p90 of the same"),
        ("metrica", "spell_p95_s", r(spells["seconds"].quantile(0.95)), "p95 of the same"),
        ("metrica", "press_path_median_m", r(spells["path_m"].median()), "ground covered by the presser during one commitment"),
        ("metrica", "press_path_p90_m", r(spells["path_m"].quantile(0.90)), "p90 of the same"),
        ("metrica", "press_path_p95_m", r(spells["path_m"].quantile(0.95)), "p95 of the same"),
        ("metrica", "press_net_median_m", r(spells["net_m"].median()), "straight-line displacement over one commitment"),
    ]
    return pd.DataFrame(rows, columns=["source", "parameter", "value", "meaning"])


def main():
    print("StatsBomb low-block freeze frames ...")
    sb = build_statsbomb_table()
    depth_min = float(sb["back_line_x"].quantile(0.05))
    print(f"  {len(sb):,} frames; back line x q05 = {depth_min:.1f} (sim frame)")

    print("Metrica tracking, filtered to that depth band ...")
    home, away, events = load_match()
    mt = build_metrica_table(home, away, events, depth_min)
    spells = press_spells(mt)
    print(f"  {len(mt):,} low-block frames, {len(spells):,} press spells")

    fig_engagement(sb, mt, os.path.join(CAL_DIR, "press_engagement.png"))
    fig_commitment(sb, mt, os.path.join(CAL_DIR, "press_commitment.png"))
    fig_shape(sb, os.path.join(CAL_DIR, "press_shape.png"))
    fig_duration(spells, os.path.join(CAL_DIR, "press_duration.png"))

    sb.to_csv(os.path.join(CAL_DIR, "press_frames.csv"), index=False)
    spells.to_csv(os.path.join(CAL_DIR, "press_spells.csv"), index=False)
    params = derive_params(sb, mt, spells, depth_min)
    params.to_csv(os.path.join(CAL_DIR, "press_policy_params.csv"), index=False)

    print("\n" + params.to_string(index=False))
    print("\nwrote press_engagement.png, press_commitment.png, press_shape.png,")
    print("      press_duration.png, press_frames.csv, press_spells.csv,")
    print("      press_policy_params.csv")


if __name__ == "__main__":
    main()
