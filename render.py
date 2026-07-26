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
    action_decoding,
    ball_action,
    ball_mechanics,
    kinematics_integrator,
    speed_lookup,
    direction_lookup,
)
from schema import player_dt

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


def make_initial_world(n_att=8, n_def=10, seed=0):
    """Build a starting roster + ball matching schema.player_dt and the
    engine's ball dict shape. Attacker ids come first so `ball_action`'s
    attacker-id array is a clean slice."""
    rng = np.random.default_rng(seed)

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(1, n_att + n_def + 1)  # ids are 1-based, HOLD==0
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL

    # Attackers spread across the middle third, defenders in a low block near
    # the x=105 goal they defend.
    players["position"][:n_att] = rng.uniform([20, 8], [65, 60], size=(n_att, 2))
    players["position"][n_att:] = rng.uniform([72, 8], [100, 60], size=(n_def, 2))
    players["velocity"][:] = 0.0

    attacker_ids = players["id"][:n_att]
    holder_id = int(attacker_ids[0])

    ball = {
        "state": "held",
        "holder_id": holder_id,
        "position": players["position"][players["id"] == holder_id][0].copy(),
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }
    return players, ball, attacker_ids, rng


def choose_actions(players, ball, attacker_ids, rng, pass_prob=0.03):
    """A stand-in policy (no trained agent yet):

    * Every player gets a per-tick (direction_idx, speed_idx) into the engine's
      lookup tables. Attackers drift goalward (+x), defenders converge on the
      ball; both get a little noise so the sim looks alive.
    * The current holder occasionally decides to pass to a random teammate,
      encoded exactly the way `ball_action` expects (a per-attacker choice
      array indexed by sorted teammate order).

    Returns (direction_idx, speed_idx, ball_idx) -- all int arrays sized to the
    full roster / attacker roster respectively.
    """
    n = len(players)
    n_speeds = len(speed_lookup)

    att_mask = players["team"] == ATTACKER_LABEL
    def_mask = ~att_mask

    direction_idx = np.zeros(n, dtype=int)
    speed_idx = np.ones(n, dtype=int)  # default: medium speed

    # Attackers push toward +x (goal), with occasional diagonal variety.
    # direction_lookup: 0=+x, 4/5 are the +x diagonals.
    direction_idx[att_mask] = rng.choice([0, 4, 5], size=att_mask.sum())

    # Defenders steer toward the ball: pick the lookup direction whose unit
    # vector best matches (ball - defender).
    ball_pos = np.asarray(ball["position"], dtype="f4")
    to_ball = ball_pos - players["position"][def_mask]
    # cosine-ish match against each lookup direction (skip the last zero row)
    dirs = direction_lookup[:-1]
    norms = np.linalg.norm(to_ball, axis=1, keepdims=True)
    unit_to_ball = np.divide(to_ball, norms, out=np.zeros_like(to_ball),
                             where=norms > 1e-6)
    scores = unit_to_ball @ dirs.T
    direction_idx[def_mask] = np.argmax(scores, axis=1)

    # A bit of speed variety across everyone.
    speed_idx[:] = rng.integers(0, n_speeds, size=n)

    # Ball action: default HOLD (0) for every attacker. If the ball is held by
    # an attacker, that holder may choose to pass to a random teammate.
    ball_idx = np.zeros(len(attacker_ids), dtype=int)
    holder_id = ball.get("holder_id")
    if (ball["state"] == "held" and holder_id is not None
            and np.any(attacker_ids == holder_id) and rng.random() < pass_prob):
        n_teammates = len(attacker_ids) - 1
        if n_teammates >= 1:
            # holder_choice in [1, n_teammates] indexes sorted teammate ids
            choice = int(rng.integers(1, n_teammates + 1))
            ball_idx[attacker_ids == holder_id] = choice

    return direction_idx, speed_idx, ball_idx


def step(players, ball, attacker_ids, rng):
    """Advance the world one DT tick using only engine.py physics."""
    direction_idx, speed_idx, ball_idx = choose_actions(
        players, ball, attacker_ids, rng)

    # 1) decode discrete actions -> target velocities, integrate kinematics
    target_velocities = action_decoding(direction_idx, speed_idx)
    players = kinematics_integrator(players, target_velocities)

    # 2) resolve the ball: pass decision, then flight/hold mechanics
    pass_decision = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)
    ball = ball_mechanics(ball, players, pass_decision)

    return players, ball


def run_simulation(n_att=8, n_def=10, seed=0, n_ticks=2000, interval_ms=None,
                   show_zones=False):
    """Open a matplotlib window and animate the engine in real time.

    interval_ms defaults to DT * 1000 so wall-clock ~= sim-clock (real time).
    """
    if interval_ms is None:
        interval_ms = int(DT * 1000)

    players, ball, attacker_ids, rng = make_initial_world(n_att, n_def, seed)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(PITCH_COLOR)

    # mutable state carried across FuncAnimation frames
    world = {"players": players, "ball": ball, "tick": 0}

    def update(_frame):
        world["players"], world["ball"] = step(
            world["players"], world["ball"], attacker_ids, rng)
        world["tick"] += 1
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
    run_simulation()