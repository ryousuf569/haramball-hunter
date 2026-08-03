import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Circle, Rectangle

# DT is the only engine constant the driver needs: it sets the animation
# interval and the per-tick real-time budget.
from physics.engine import DT

from environment.lowblock_env import (
    ATTACKER_LABEL,
    DEFENDER_LABEL,
    PC_EXTENT,
    make_initial_world,
    make_ppcf_grid,
    step,
)

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
                  clear=True, title=None, pc_att=None):
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
    pc_att : ndarray (PC_NX, PC_NY) or None
        Attacker pitch-control field from compute_attacker_ppcf, drawn as a
        heatmap under the players over the grid's global-frame extent. None
        skips the overlay.

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

    # Attacker pitch control, under everything else. .T because the field is
    # indexed (x, y) while imshow reads (row, col) = (y, x).
    if pc_att is not None:
        ax.imshow(pc_att.T, origin="lower", extent=PC_EXTENT, cmap="Reds",
                  vmin=0.0, vmax=1.0, alpha=0.55, zorder=1,
                  interpolation="bilinear")

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
# Everything below drives the env in environment/lowblock_env.py and animates
# it. The env exposes make_initial_world/step; the driver's job is to (1) hold
# the world state it hands back, (2) advance one DT tick, and (3) pass the new
# state to render_frame.


def run_simulation(n_att=10, n_def=11, seed=23235454, n_ticks=2500, interval_ms=None,
                   show_zones=False, start_holder=1, show_ppcf=True):
    """Open a matplotlib window and animate the engine in real time.

    interval_ms defaults to DT * 1000 so wall-clock ~= sim-clock (real time).

    start_holder chooses which attacker kicks off with the ball (attacker row
    index; see make_initial_world). Change it to watch the low block react to
    different starting situations -- e.g. start_holder=0 (deep build-up) vs
    8 (ball already at the central striker).

    show_ppcf controls whether the pitch-control heatmap is DRAWN. The field is
    computed every tick either way, because the same call caches each player's
    TTI to the ball in players['i_p'] and intercept_pass needs it -- skipping it
    would silently disable pass interceptions rather than just hiding an overlay.

    Timing note: measured here, step+render is ~130ms/tick with the PPCF overlay
    on vs ~68ms off, against DT=100ms -- so the animation runs at roughly 0.75x
    real time. PPCF is ~54ms of that, nearly all of it the 200 integration steps
    (integration_horizon / integration_timestep) in ppcf.PPCF_grid. Since the
    field is now always computed, show_ppcf=False only saves the draw, not that
    54ms. If real-time playback matters, coarsen the integration instead.
    """
    if interval_ms is None:
        interval_ms = int(DT * 1000)

    # rng is only used inside make_initial_world (for starting positions); the
    # step loop is now fully scripted, so we don't thread it through.
    players, ball, attacker_ids, _rng, defender_state = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(PITCH_COLOR)

    # Cell centres are fixed for the whole run, so build the grid once and reuse
    # the same (n_cells, 2) array on every tick. Always built: the PPCF call is
    # what caches players['i_p'] for intercept_pass, so it is no longer optional.
    ppcf_grid = make_ppcf_grid()

    # mutable state carried across FuncAnimation frames. step_ms/render_ms
    # accumulate the per-tick cost split so the running means below aren't
    # dominated by whichever tick happened to be slow.
    world = {"players": players, "ball": ball, "tick": 0,
             "step_ms": 0.0, "render_ms": 0.0}

    budget_ms = DT * 1000.0

    def update(_frame):
        world["tick"] += 1

        t_step0 = time.perf_counter()
        world["players"], world["ball"], pc_att = step(
            world["players"], world["ball"], attacker_ids, defender_state,
            world["tick"], ppcf_grid=ppcf_grid)
        step_ms = (time.perf_counter() - t_step0) * 1000.0

        t = world["tick"] * DT
        state = world["ball"]["state"]

        t_render0 = time.perf_counter()
        # pc_att is computed every tick (intercept_pass needs the cache it builds),
        # so show_ppcf gates only whether it reaches the heatmap.
        render_frame(world["players"], world["ball"], ax=ax,
                     show_zones=show_zones,
                     pc_att=pc_att if show_ppcf else None,
                     title=f"tick {world['tick']}  |  t = {t:5.1f}s  |  ball: {state}")
        render_ms = (time.perf_counter() - t_render0) * 1000.0

        # Note: render_ms covers building the artists, not the canvas blit that
        # matplotlib does after update() returns, so total_ms understates true
        # wall-clock per frame somewhat.
        total_ms = step_ms + render_ms
        world["step_ms"] += step_ms
        world["render_ms"] += render_ms
        n = world["tick"]
        mean_ms = (world["step_ms"] + world["render_ms"]) / n

        # Flag ticks that blow the real-time budget -- with PPCF on this is the
        # common case, so it's a marker rather than a warning.
        over = "  OVER" if total_ms > budget_ms else ""
        print(f"tick {n:5d}  step {step_ms:6.1f}ms  render {render_ms:6.1f}ms  "
              f"total {total_ms:6.1f}ms  (mean {mean_ms:6.1f}ms / "
              f"budget {budget_ms:.0f}ms){over}")
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
