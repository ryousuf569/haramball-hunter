"""
render.py -- Matplotlib rendering for the low-block engine.
Coordinate system matches engine.py: global pitch frame, origin bottom-left,
105m x 68m, attacking direction is +x, goal defended by the low block is at
x = 105.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Circle, Rectangle

# Engine physics -- render.py is the only file allowed to change, so we import
# the real integrators/mechanics rather than reimplementing them here.
from engine import (
    DT,
    ball_action,
    ball_mechanics,
    kinematics_integrator,
)
from schema import player_dt

# Defenders now run their learned/scripted low-block behaviour. This module
# returns target velocities for the whole defender line directly (bypassing the
# discrete direction/speed action decoding used for attackers), so we splice
# its output into the roster-wide target-velocity array before integrating.
from defenders import make_defender_state, compute_defender_targets

# Attackers run a throwaway scripted baseline (a fixed formation + fixed-cadence
# passing) purely so the sim looks coherent while the defender script is being
# tested. Like the defenders, it returns continuous target velocities directly
# (bypassing the discrete action_decoding path) plus the ball_idx array that
# engine.ball_action expects. Replace wholesale when a real attacker exists.
from baseline_attacker import (
    compute_attacker_targets,
    backline_offset as att_backline_offset,
    midline_offset as att_midline_offset,
    forward_offset as att_forward_offset,
)

# --- team labels -----------------------------------------------------------
# players['team'] is a U8 string field per schema.py. Set these to match
# whatever string values you actually write into that field when you
# instantiate players (e.g. if you used 'att'/'def' instead, change here).
ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

# --- style -------------------------------------------------------------
PITCH_COLOR = "#1e5631"
LINE_COLOR = "#e8e8e8"
ATTACKER_COLOR = "#1f77b4"
DEFENDER_COLOR = "#d62728"
BALL_COLOR = "#f2f2f2"
HOLDER_RING_COLOR = "#ffd400"
RECEIVER_RING_COLOR = "#cccccc"
FLIGHT_LINE_COLOR = "#ffffff"

PLAYER_RADIUS = 1.1          # m, marker footprint on pitch
VELOCITY_LOOKAHEAD_S = 1.0   # arrow length = displacement over this many seconds


def _draw_pitch(ax, pitch_length=105.0, pitch_width=68.0):
    """Draw standard soccer pitch markings in global engine coordinates."""
    ax.set_facecolor(PITCH_COLOR)

    # Outer boundary
    ax.add_patch(Rectangle((0, 0), pitch_length, pitch_width,
                            fill=False, edgecolor=LINE_COLOR, linewidth=1.5))

    # Halfway line + center circle/spot
    ax.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width],
            color=LINE_COLOR, linewidth=1.2)
    ax.add_patch(Circle((pitch_length / 2, pitch_width / 2), 9.15,
                         fill=False, edgecolor=LINE_COLOR, linewidth=1.2))
    ax.plot(pitch_length / 2, pitch_width / 2, marker="o",
            color=LINE_COLOR, markersize=2)

    # Penalty areas, 6-yard boxes, penalty spots, arcs -- both ends.
    # x=0 end (defended by the attacking team's own goal) and x=105 end
    # (the goal the attacking team, and the low block, are set up around).
    box_w, box_d = 40.32, 16.5
    six_w, six_d = 18.32, 5.5
    spot_d = 11.0
    cy = pitch_width / 2

    for x0, direction in [(0.0, 1), (pitch_length, -1)]:
        # penalty area
        ax.add_patch(Rectangle(
            (x0 if direction == 1 else x0 - box_d, cy - box_w / 2),
            box_d, box_w, fill=False, edgecolor=LINE_COLOR, linewidth=1.2))
        # 6-yard box
        ax.add_patch(Rectangle(
            (x0 if direction == 1 else x0 - six_d, cy - six_w / 2),
            six_d, six_w, fill=False, edgecolor=LINE_COLOR, linewidth=1.2))
        # penalty spot
        spot_x = x0 + direction * spot_d
        ax.plot(spot_x, cy, marker="o", color=LINE_COLOR, markersize=2)
        # penalty arc (only the part outside the box)
        theta = 0.0 if direction == 1 else 180.0
        ax.add_patch(Arc((spot_x, cy), 2 * 9.15, 2 * 9.15,
                          angle=theta, theta1=-53, theta2=53,
                          edgecolor=LINE_COLOR, linewidth=1.2))
        # goal mouth (small rectangle poking out of the goal line)
        goal_w, goal_d = 7.32, 2.0
        ax.add_patch(Rectangle(
            (x0 - goal_d if direction == 1 else x0, cy - goal_w / 2),
            goal_d, goal_w, fill=False, edgecolor=LINE_COLOR, linewidth=1.5))

    ax.set_xlim(-3, pitch_length + 3)
    ax.set_ylim(-3, pitch_width + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_reward_zones(ax, pitch_width=68.0):
    """Optional debug overlay: final-third line and half-space corridors
    (Sections 4.2 / 4.3 of the spec). Off by default -- this is a reward
    debugging aid, not part of the base engine render."""
    ax.axvline(70.0, color="white", linestyle="--", linewidth=1.0, alpha=0.5)
    for y_lo, y_hi in [(14.0, 27.0), (41.0, 54.0)]:
        ax.add_patch(Rectangle((70.0, y_lo), 105.0 - 70.0, y_hi - y_lo,
                                facecolor="white", alpha=0.08, edgecolor=None))


def _draw_team(ax, positions, velocities, ids, color, show_ids, show_velocity):
    if len(positions) == 0:
        return
    ax.scatter(positions[:, 0], positions[:, 1], s=(PLAYER_RADIUS * 55) ** 1.0,
               color=color, edgecolor="black", linewidth=0.6, zorder=3)

    if show_velocity:
        disp = velocities * VELOCITY_LOOKAHEAD_S
        ax.quiver(positions[:, 0], positions[:, 1], disp[:, 0], disp[:, 1],
                  angles="xy", scale_units="xy", scale=1.0,
                  color=color, width=0.0035, alpha=0.85, zorder=2)

    if show_ids:
        for pid, pos in zip(ids, positions):
            ax.annotate(str(int(pid)), xy=(pos[0], pos[1]),
                        xytext=(pos[0] + 0.9, pos[1] + 0.9),
                        fontsize=7, color="white", zorder=4)


def render_frame(players, ball, ax=None, show_ids=True, show_velocity=True,
                  show_zones=False, pitch_length=105.0, pitch_width=68.0,
                  clear=True, title=None):
    """
    Draw one frame of engine state.

    Parameters
    ----------
    players : structured ndarray (player_dt)
        Combined attacker + defender roster, as produced by the engine --
        same array shape ball_mechanics/kinematics_integrator already
        operate on. Split internally by players['team'].
    ball : dict
        The engine's ball state dict (state, holder_id, position, target_id,
        flight_start, flight_target).
    ax : matplotlib.axes.Axes or None
        Axes to draw into. If None, a new figure/axes is created. Pass the
        same ax across calls (e.g. inside a FuncAnimation update function)
        to reuse the figure.
    show_ids : bool
        Draw each player's id next to their marker. Default True -- useful
        for confirming kinematics/pass-target correctness frame by frame.
    show_velocity : bool
        Draw a velocity vector for each player (displacement over
        VELOCITY_LOOKAHEAD_S seconds).
    show_zones : bool
        Overlay the final-third line and half-space corridors from the
        reward spec (Section 4.2/4.3). Off by default -- reward debugging,
        not base rendering.
    clear : bool
        Clear the axes before drawing. Set False if you're managing the
        clear/redraw cycle yourself in an animation loop.
    title : str or None
        Optional title (e.g. tick number) drawn above the pitch.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.5, 6.8))
    else:
        fig = ax.figure

    if clear:
        ax.clear()

    _draw_pitch(ax, pitch_length, pitch_width)
    if show_zones:
        _draw_reward_zones(ax, pitch_width)

    att_mask = players["team"] == ATTACKER_LABEL
    def_mask = players["team"] == DEFENDER_LABEL

    _draw_team(ax, players["position"][att_mask], players["velocity"][att_mask],
               players["id"][att_mask], ATTACKER_COLOR, show_ids, show_velocity)
    _draw_team(ax, players["position"][def_mask], players["velocity"][def_mask],
               players["id"][def_mask], DEFENDER_COLOR, show_ids, show_velocity)

    # Ball-holder highlight
    if ball.get("holder_id") is not None:
        holder_mask = players["id"] == ball["holder_id"]
        if np.any(holder_mask):
            hx, hy = players["position"][holder_mask][0]
            ax.add_patch(Circle((hx, hy), PLAYER_RADIUS + 0.6, fill=False,
                                 edgecolor=HOLDER_RING_COLOR, linewidth=2.0,
                                 zorder=5))

    # In-flight ball: flight path + intended receiver highlight
    if ball["state"] == "in_flight":
        fs, ft = ball["flight_start"], ball["flight_target"]
        ax.plot([fs[0], ft[0]], [fs[1], ft[1]], linestyle="--",
                color=FLIGHT_LINE_COLOR, linewidth=1.0, alpha=0.6, zorder=2)

        if ball.get("target_id") is not None:
            recv_mask = players["id"] == ball["target_id"]
            if np.any(recv_mask):
                rx, ry = players["position"][recv_mask][0]
                ax.add_patch(Circle((rx, ry), PLAYER_RADIUS + 0.6, fill=False,
                                     edgecolor=RECEIVER_RING_COLOR,
                                     linestyle="--", linewidth=1.5, zorder=5))

    # Ball marker itself
    bx, by = ball["position"]
    ax.scatter([bx], [by], s=40, color=BALL_COLOR, edgecolor="black",
               linewidth=0.8, zorder=6)

    if title:
        ax.set_title(title, color="white", fontsize=10)

    return fig, ax


# ---------------------------------------------------------------------------
# Live simulation driver
# ---------------------------------------------------------------------------
# Everything below turns the pure engine (engine.py) into a running, animated
# match. The engine exposes stateless steppers -- action_decoding /
# ball_action / ball_mechanics / kinematics_integrator -- so the driver's job
# is to (1) hold the world state, (2) pick per-tick actions (a simple policy,
# since there's no trained agent wired in yet), and (3) advance one DT tick and
# hand the new state to render_frame.


def _defender_formation_5_4_1(n_def=10, pitch_width=68.0):
    """Starting defender positions in a resting low-block 5-4-1.

    Row order matches defenders.py's line indexing so the scripted policy picks
    up a coherent shape from tick 0: rows 0-4 are the backline, rows 5-8 the
    midfield line, row 9 the lone forward. y-offsets mirror the offset arrays in
    defenders.py (backline spans +/-20m, midfield +/-15m about the centerline).

    Deep near the x=105 goal: backline ~x=100, midfield line ~x=82, forward
    pushed up to ~x=60 (the block's max height per the spec).
    """
    cy = pitch_width / 2.0  # 34.0

    back_x, mid_x, fwd_x = 100.0, 82.0, 60.0
    back_dy = [-20, -10, 0, 10, 20]   # 5 defenders
    mid_dy = [-15, -5, 5, 15]         # 4 midfielders

    positions = [[back_x, cy + dy] for dy in back_dy]
    positions += [[mid_x, cy + dy] for dy in mid_dy]
    positions += [[fwd_x, cy]]        # lone forward, central

    positions = np.array(positions, dtype="f4")

    # Defensive: if the roster asks for a different defender count than the 10
    # this 5-4-1 lays out, fall back to as many of these slots as fit.
    if n_def != len(positions):
        positions = positions[:n_def]
    return positions


def _attacker_formation_2_5_3(n_att=10, ref_x=64.0, pitch_width=68.0):
    """Starting attacker positions in a 2-5-3, already in shape and pushed up
    the field.

    Uses baseline_attacker.py's own line offset arrays so the start shape is
    identical to what the baseline steers toward -- players begin at rest in
    formation instead of drifting in from random spots. The only difference
    from the baseline's resting FORMATION_REF is a further-up-the-pitch x
    reference (ref_x, default 64 vs the baseline's 52), so they kick off
    advanced. Row order matches baseline_attacker.py: rows 0-1 backline, rows
    2-6 midfield, rows 7-9 the front three.
    """
    ref = np.array([ref_x, pitch_width / 2.0])  # centered in y, advanced in x

    positions = np.concatenate([
        ref + att_backline_offset,   # 2 backline
        ref + att_midline_offset,    # 5 midfield
        ref + att_forward_offset,    # 3 forward
    ]).astype("f4")

    # Defensive: if the roster asks for a different attacker count than the 10
    # this 2-5-3 lays out, fall back to as many of these slots as fit.
    if n_att != len(positions):
        positions = positions[:n_att]
    return positions


def make_initial_world(n_att=10, n_def=10, seed=0, start_holder=0):
    """Build a starting roster + ball matching schema.player_dt and the
    engine's ball dict shape. Attacker ids come first so `ball_action`'s
    attacker-id array is a clean slice.

    n_att defaults to 10 to match baseline_attacker.py's 2-5-3 formation
    (2 backline + 5 midfield + 3 forward = 10). n_def stays 10 for the
    defenders' 5-4-1 low block.

    start_holder picks which attacker begins with the ball -- change it to see
    how the low block reacts to different starting situations. It's the ROW
    INDEX into the 2-5-3 attacker layout (baseline_attacker.py's order):
        0-1  backline (deep build-up), e.g. 0 = deep-left
        2-6  midfield line, e.g. 4 = central midfielder
        7-9  front three, e.g. 8 = central striker
    (A negative or out-of-range value is clipped into the attacker rows.)"""
    rng = np.random.default_rng(seed)

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(1, n_att + n_def + 1)  # ids are 1-based, HOLD==0
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL

    # Attackers (10: a 2-5-3, matching baseline_attacker.py) start already in
    # formation shape and pushed up the field (x ref ~64). Defenders sit in a
    # resting low-block 5-4-1 near the x=105 goal they defend. Row layout
    # matches defenders.py: rows 0-4 backline, rows 5-8 midfield, row 9 forward.
    players["position"][:n_att] = _attacker_formation_2_5_3(n_att)
    players["position"][n_att:] = _defender_formation_5_4_1(n_def)
    players["velocity"][:] = 0.0

    attacker_ids = players["id"][:n_att]
    # Pick the starting holder by attacker row index (clipped into range).
    holder_row = int(np.clip(start_holder, 0, n_att - 1))
    holder_id = int(attacker_ids[holder_row])

    ball = {
        "state": "held",
        "holder_id": holder_id,
        "position": players["position"][players["id"] == holder_id][0].copy(),
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }

    # Persistent state for the scripted defenders (holds a short ball_x history
    # deque used to lag the block's depth reference). Created once and threaded
    # through every step().
    defender_state = make_defender_state()
    return players, ball, attacker_ids, rng, defender_state


def step(players, ball, attacker_ids, defender_state, tick_count):
    """Advance the world one DT tick using only engine.py physics.

    Both teams now supply continuous target velocities directly (no discrete
    action_decoding): attackers from the scripted baseline in
    baseline_attacker.py, defenders from the scripted low-block policy in
    defenders.py. Both are spliced into one roster-wide target array before a
    single kinematics integration.
    """
    att_mask = players["team"] == ATTACKER_LABEL
    def_mask = players["team"] == DEFENDER_LABEL

    # Roster-wide target-velocity array, filled per team below. Rows are laid
    # out attackers-first (rows :n_att) then defenders (rows n_att:), matching
    # both policies' row ordering.
    target_velocities = np.zeros((len(players), 2), dtype="f4")

    # 1) attackers run the throwaway baseline -> target velocities + ball_idx.
    #    tick_count drives the baseline's fixed passing cadence.
    attacker_velocities, ball_idx = compute_attacker_targets(
        players, ball, tick_count)
    target_velocities[att_mask] = attacker_velocities

    # 2) defenders run their learned script -> target velocities for the block.
    #    compute_defender_targets returns velocities in defender-row order,
    #    which matches how the roster is laid out (defenders occupy rows n_att:).
    defender_velocities = compute_defender_targets(players, ball, defender_state)
    target_velocities[def_mask] = defender_velocities

    # 3) integrate kinematics for everyone with the combined targets.
    players = kinematics_integrator(players, target_velocities)

    # 4) resolve the ball: pass decision, then flight/hold mechanics
    pass_decision = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)
    ball = ball_mechanics(ball, players, pass_decision)

    return players, ball


def run_simulation(n_att=10, n_def=10, seed=0, n_ticks=2000, interval_ms=None,
                   show_zones=False, start_holder=0):
    """Open a matplotlib window and animate the engine in real time.

    interval_ms defaults to DT * 1000 so wall-clock ~= sim-clock (real time).

    start_holder chooses which attacker kicks off with the ball (attacker row
    index; see make_initial_world). Change it to watch the low block react to
    different starting situations -- e.g. start_holder=0 (deep build-up) vs
    8 (ball already at the central striker).
    """
    if interval_ms is None:
        interval_ms = int(DT * 1000)

    # rng is only used inside make_initial_world (for starting positions); the
    # step loop is now fully scripted, so we don't thread it through.
    players, ball, attacker_ids, _rng, defender_state = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(PITCH_COLOR)

    # mutable state carried across FuncAnimation frames
    world = {"players": players, "ball": ball, "tick": 0}

    def update(_frame):
        world["tick"] += 1
        world["players"], world["ball"] = step(
            world["players"], world["ball"], attacker_ids, defender_state,
            world["tick"])
        t = world["tick"] * DT
        state = world["ball"]["state"]
        render_frame(world["players"], world["ball"], ax=ax,
                     show_zones=show_zones,
                     title=f"tick {world['tick']}  |  t = {t:5.1f}s  |  ball: {state}")
        return []

    # cache_frame_data=False -> don't buffer every frame (state is live/mutating)
    anim = FuncAnimation(fig, update, frames=n_ticks, interval=interval_ms,
                         blit=False, cache_frame_data=False, repeat=False)
    # keep a reference so the animation isn't garbage-collected
    fig._anim = anim

    plt.show()
    return anim


if __name__ == "__main__":
    run_simulation(start_holder=4)