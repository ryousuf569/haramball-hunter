import os
import time
from collections import Counter, deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arc, Circle, Rectangle

# DT is the only engine constant the driver needs: it sets the animation
# interval and the per-tick real-time budget.
from physics.engine import DT

from attackers.ppo_policy import make_ppo_policy
from attackers.random_policy import random_actions
from attackers.scripted_policy import make_policy
from environment.grid import PC_EXTENT
from environment.lowblock_env import (
    ATTACKER_LABEL,
    DEFENDER_LABEL,
    MAX_TICKS,
    LowBlockEnv,
    make_vector_env,
)
from environment.termination import ZONE_PC_MIN, make_zone, success_gate

POLICIES = ("random", "scripted")  # plus: a path to a trained .pt checkpoint

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

# Ticks averaged for the live steps/s readout. Short enough to react when a
# backend hits a slow patch, long enough that one outlier tick doesn't make the
# number jump around unreadably.
STEP_RATE_WINDOW = 30


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


def _draw_target_zone(ax, zone):
    """Overlay the success zone from environment/termination.py -- a radius from
    goal. Arriving inside it is only half the condition; the team also has to
    control the cells within it, which is what the heatmap underneath shows."""
    ax.add_patch(Circle(zone.centre, zone.radius, fill=False, edgecolor="white",
                         linestyle="--", linewidth=1.2, alpha=0.65, zorder=2))


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
                  zone=None, pitch_length=105.0, pitch_width=68.0,
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
    zone : termination.Zone or None
        Overlay the success radius from environment/termination.py. None
        skips it.
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
    if zone is not None:
        _draw_target_zone(ax, zone)

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
# Everything below drives one of the fixed policies in attackers/ and animates
# it. The world lives in environment/lowblock_env.py's LowBlockEnv -- the same
# gym env model/ppo.py trains against, not a parallel copy of the loop -- so the
# window shows the reward, the shaping and the observation the agent actually
# sees. The driver's job is to (1) pick actions, (2) call env.step, (3) draw the
# env's state, and (4) start a fresh episode whenever one ends, so a single
# window shows many possessions and a running outcome tally.


def _stack_actions(actions):
    """Policy triple (direction, speed, ball) -> the env's (n_att, 3) action."""
    return np.stack(actions, axis=1).astype(np.int64)


def _agent_actions(agent, obs, deterministic):
    """Batched observation (n_envs, n_att, obs_dim) -> (n_envs, n_att, 3).

    Mirrors PPOPolicy.__call__ but takes the observation the env already built
    rather than rebuilding it, which also means no second PPCF pass per tick.
    """
    obs_t = torch.as_tensor(np.asarray(obs, dtype="f4"))
    with torch.no_grad():
        if deterministic:
            n_env, n_att, dim = obs_t.shape
            dist, _ = agent._dist_and_value(obs_t.reshape(n_env * n_att, dim))
            act = dist.mode().reshape(n_env, n_att, 3)
        else:
            act, _logp, _val = agent.act(obs_t)
    return act.numpy()


class _EnvDriver:
    """One LowBlockEnv, or n of them under a vector env, behind one interface.

    n_envs=1 is the path to read: reset/step map straight onto the gym API and
    the driver owns the episode boundary. n_envs>1 goes through make_vector_env
    so the window can show a training-shaped batch stepping in lockstep -- every
    env advances, one of them is drawn, and the outcome tally fills n times
    faster.

    The vector env is synchronous on purpose: AsyncVectorEnv keeps its sub-envs
    in subprocesses, where players/ball live in another address space and there
    is nothing here to draw. Autoreset is left at gymnasium's NEXT_STEP default,
    which returns the terminal observation and only resets on the following
    step -- that is what lets a finished possession stay on screen while the
    driver holds off calling step.
    """

    def __init__(self, n_envs, **env_kwargs):
        self.n_envs = n_envs
        if n_envs == 1:
            self.venv = None
            self.envs = [LowBlockEnv(**env_kwargs)]
        else:
            self.venv = make_vector_env(n_envs=n_envs, asynchronous=False,
                                        **env_kwargs)
            self.envs = list(self.venv.envs)
        self.obs = None

    def reset(self, seed=None):
        if self.venv is None:
            # A bare LowBlockEnv observes (n_att, obs_dim); everything
            # downstream wants the vector env's leading env axis, so add it.
            obs, _info = self.envs[0].reset(seed=seed)
            self.obs = obs[None]
        else:
            self.obs, _info = self.venv.reset(seed=seed)
        return self.obs

    def reset_one(self, i):
        """Restart sub-env i. Single-env only -- the vector env autoresets."""
        assert self.venv is None
        obs, _info = self.envs[i].reset()
        self.obs = obs[None]
        return self.obs

    def step(self, actions):
        """actions is (n_envs, n_att, 3). Returns (rewards, outcomes)."""
        if self.venv is None:
            obs, reward, _term, _trunc, info = self.envs[0].step(actions[0])
            self.obs = obs[None]
            return np.array([reward], dtype=float), [info.get("outcome")]

        self.obs, reward, _term, _trunc, info = self.venv.step(actions)
        return np.asarray(reward, dtype=float), self._outcomes(info)

    def _outcomes(self, info):
        # The vector env stacks each sub-env's info key into one object array
        # (None where that env did not terminate). Read it defensively: the key
        # is absent entirely on a step where every env autoreset.
        raw = info.get("outcome")
        outcomes = [None] * self.n_envs
        if raw is None:
            return outcomes
        for i in range(self.n_envs):
            value = raw[i]
            if isinstance(value, str):
                outcomes[i] = value
        return outcomes

    def close(self):
        if self.venv is not None:
            self.venv.close()


def run_simulation(n_att=10, n_def=11, seed=245365, n_ticks=2500, interval_ms=None,
                   show_zone=True, start_holder=1, show_ppcf=True,
                   max_ticks=MAX_TICKS, pass_prob=None, hold_ticks=8,
                   zone=None, policy="runs/vanilla_10m_cuda_rung2", hold_shape=True,
                   deterministic=False, pc_min=None, n_envs=1, render_env=0):
    """Open a matplotlib window and animate the attackers against the
    calibrated low block, restarting on every terminal outcome.

    The world is a LowBlockEnv from environment/lowblock_env.py -- the env
    model/ppo.py trains on -- so what the window shows is what the agent gets:
    the title carries the per-episode return and the last tick's shaping term
    alongside the gate readout.

    n_envs > 1 puts that env behind make_vector_env (synchronous; see
    _EnvDriver) and steps a whole batch per frame, drawing sub-env render_env
    and tallying outcomes from all of them. The tally then fills n_envs times
    faster per wall-clock second, and the steps/s readout becomes the batch's
    throughput -- which is the number to compare across PPCF backends. Set
    PPCF_BACKEND=cuda in the environment to run the kernel from physics/ppcf.cu
    instead of physics/ppcf.py; it is read at import time, so set it before
    launching rather than from inside the process.

    policy picks who drives them: "random" for the uniform control, "scripted"
    for attackers/scripted_policy.py, or a path to a .pt checkpoint written by
    model/ppo.py (e.g. "runs/ppo_500k_s0/best.pt") to watch a trained agent
    play. Measured over 25 episodes, random scores ~16% and scripted ~84%, so
    expect the scripted window to look like football and resolve in about a
    third of the ticks; a checkpoint lands wherever its training got to, which
    the title bar names so windows are not confusable.

    deterministic applies to checkpoints only. False (default) samples the
    policy, which is how it behaved during training; True takes the argmax of
    every action head, which reads its intent more cleanly but is off the
    distribution it was trained on.

    hold_shape is passed to the scripted policy and ignored by random. True
    keeps each attacker on its kickoff offset from the centroid so the unit
    advances in shape; False sends all ten at the disc, which is the degenerate
    control for whether the success gate can be farmed by crowding.

    interval_ms defaults to DT * 1000 so wall-clock ~= sim-clock (real time).

    pass_prob goes to random_policy.random_actions and is ignored by the
    scripted policy and by checkpoints. None means a uniform draw over the whole
    ball head, which is the honest control for the probe and releases the ball
    on roughly (n_att - 1) / n_att of carrying ticks. Pass a float to see the
    same movement with a calmer ball.

    start_holder chooses which attacker kicks off with the ball (attacker row
    index; see make_initial_world). Change it to watch the low block react to
    different starting situations -- e.g. start_holder=0 (deep build-up) vs
    8 (ball already at the central striker). Negative draws a fresh holder per
    episode, which is what LowBlockEnv's start_holder=None does.

    hold_ticks is how many frames the terminal state stays on screen before the
    next episode starts, so an outcome is readable rather than a flicker.

    zone is a termination.Zone; None builds the default -- except when policy
    is a checkpoint, where None takes the gate the checkpoint was trained on
    (zone and pc_min both), so a curriculum run is not judged against a gate it
    never saw. pc_min=None follows the same rule. Pass either to override.

    zone is a termination.Zone; None builds the default. Pass
    make_zone(x, y, radius) to move or resize it. The title prints the live
    gate -- whether the ball is inside the disc, and the team's mean control
    over it -- so you can see which half of the condition is binding while a
    possession runs, rather than only learning that it failed.

    show_ppcf controls whether the pitch-control heatmap is DRAWN. The field is
    computed every tick either way, because the same call caches each player's
    TTI to the ball in players['i_p'] and intercept_pass needs it -- skipping it
    would silently disable pass interceptions rather than just hiding an overlay.
    """
    ppo = None
    if policy not in POLICIES:
        if not (str(policy).endswith(".pt") or os.path.isdir(policy)):
            raise ValueError(f"policy must be one of {POLICIES}, a .pt "
                             f"checkpoint, or a run directory, got {policy!r}")
        if not os.path.exists(policy):
            raise FileNotFoundError(f"no checkpoint at {policy!r}")
    if interval_ms is None:
        interval_ms = int(DT * 1000)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor(PITCH_COLOR)

    # Loaded once and reused across episodes. The checkpoint carries the gate it
    # was trained on, so zone/pc_min are resolved from it rather than the other
    # way round -- a curriculum run judged against a gate it never saw reads as
    # a far worse agent than it is. Only the agent and that metadata are used
    # from here: actions come off the env's own observation, so PPOPolicy's
    # observation shell and clock go unused (see _agent_actions).
    if policy not in POLICIES:
        ppo = make_ppo_policy(policy, zone=zone, n_att=n_att, n_def=n_def,
                              max_ticks=max_ticks, deterministic=deterministic,
                              seed=seed)
        zone = ppo.zone
        if pc_min is None:
            pc_min = ppo.pc_min
    if zone is None:
        zone = make_zone()
    if pc_min is None:
        pc_min = ZONE_PC_MIN
    if not 0 <= render_env < n_envs:
        raise ValueError(f"render_env {render_env} is out of range for "
                         f"{n_envs} env(s)")

    # LowBlockEnv rebuilds the zone from its centre/radius, so hand it those
    # numbers rather than the object -- make_zone is deterministic, so the gate
    # the env checks is the one resolved above. A negative start_holder means
    # "draw one per episode", matching the convention model/ppo.py's cfg uses.
    driver = _EnvDriver(
        n_envs, n_att=n_att, n_def=n_def, max_tick=max_ticks,
        start_holder=(None if start_holder is not None and start_holder < 0
                      else start_holder),
        zone_x=float(zone.centre[0]), zone_y=float(zone.centre[1]),
        zone_radius=float(zone.radius), pc_min=float(pc_min))
    # One seed for the whole run: the env draws a fresh episode seed from its
    # own generator on every reset, so possessions vary without the driver
    # having to feed it seed + episode by hand.
    driver.reset(seed=seed)
    shown = driver.envs[render_env]

    # The random policy needs its own stream. Each env's rng drives the engine's
    # stochasticity -- duels, interceptions, pass noise -- not the action draw.
    rng = np.random.default_rng(seed)

    # Mutable state carried across FuncAnimation frames. step_ms/render_ms
    # accumulate the per-frame cost split so the running means aren't dominated
    # by whichever frame happened to be slow. recent_step_ms is the sliding
    # window behind the live steps/s readout -- engine cost only, so the number
    # compares backends rather than matplotlib. tally is the whole point of the
    # auto-reset: it is a live preview of the random-vs-calibrated probe, and
    # ret/returns put the env's own reward next to it.
    world = {"ep": np.ones(n_envs, dtype=int), "step_ms": 0.0, "render_ms": 0.0,
             "ticks": 0, "outcome": None, "freeze": 0, "tally": Counter(),
             "recent_step_ms": deque(maxlen=STEP_RATE_WINDOW), "frames": 0,
             "ret": np.zeros(n_envs), "returns": []}

    # One scripted policy per env, rebuilt at every episode boundary: it
    # captures its slots from the formation draw on the first tick, so an
    # instance outliving its episode would play the new one against the
    # previous episode's shape. None means the random control.
    policies = [None] * n_envs

    def new_policy(i):
        policies[i] = (make_policy(zone, hold_shape)
                       if policy == "scripted" else None)

    for i in range(n_envs):
        new_policy(i)

    def actions_for():
        """(n_envs, n_att, 3) actions for the observation the envs just gave."""
        if ppo is not None:
            return _agent_actions(ppo.agent, driver.obs, deterministic)
        out = np.empty((n_envs, n_att, 3), dtype=np.int64)
        for i, env in enumerate(driver.envs):
            pol = policies[i]
            out[i] = _stack_actions(
                random_actions(n_att, rng, pass_prob) if pol is None
                else pol(env.players, env.ball, env.attacker_ids))
        return out

    budget_ms = DT * 1000.0
    if ppo is not None:
        label = ppo.label()
        print(f"loaded {ppo.ckpt_path} | step {ppo.step} | "
              f"training success {100 * ppo.stats.get('success', float('nan')):.1f}%")
        print(f"gate {ppo.gate()}")
    elif policy == "scripted":
        label = f"scripted{'' if hold_shape else ' (crowding)'}"
    else:
        label = policy

    def update(_frame):
        # Terminal state lingers for hold_ticks frames so it can be read. In
        # vector mode the whole batch waits with it, and the shown env's
        # autoreset fires on the first step after the freeze.
        if world["outcome"] is not None:
            world["freeze"] += 1
            if world["freeze"] >= hold_ticks:
                if driver.venv is None:
                    driver.reset_one(render_env)
                world["ep"][render_env] += 1
                world["outcome"] = None
                world["freeze"] = 0
            return []

        # ticks counts env-steps (n_envs per frame); frames counts the timed
        # calls, so the two costs below stay in their own units.
        world["ticks"] += n_envs
        world["frames"] += 1

        t_step0 = time.perf_counter()
        rewards, outcomes = driver.step(actions_for())
        step_ms = (time.perf_counter() - t_step0) * 1000.0
        world["recent_step_ms"].append(step_ms)
        world["ret"] += rewards

        outcome = outcomes[render_env]
        # Read before the loop below zeroes it, so a terminal frame shows the
        # return the episode actually finished on rather than the next one's 0.
        ep_return = float(world["ret"][render_env])
        for i, ep_outcome in enumerate(outcomes):
            if ep_outcome is None:
                continue
            world["tally"][ep_outcome] += 1
            world["returns"].append(float(world["ret"][i]))
            n_done = sum(world["tally"].values())
            rates = "  ".join(
                f"{k} {world['tally'][k] / n_done:.0%}"
                for k in ("success", "failure", "timeout"))
            env_tag = "" if n_envs == 1 else f"env {i}  "
            print(f"{env_tag}episode {world['ep'][i]:4d}  "
                  f"{ep_outcome.upper():8s} at tick {driver.envs[i].tick:4d}  "
                  f"|  return {world['ret'][i]:+6.2f}  "
                  f"|  over {n_done} eps: {rates}")
            world["ret"][i] = 0.0
            # The scripted policy is stale from here: whichever reset comes
            # next -- ours after the freeze, or the vector env's on the next
            # step -- draws a new formation for it to key off.
            new_policy(i)
            if i != render_env:
                world["ep"][i] += 1
        if outcome is not None:
            world["outcome"] = outcome  # ep bumped when the freeze expires

        t = shown.tick * DT
        state = shown.ball["state"]
        ended = f"  |  {outcome.upper()}" if outcome is not None else ""

        # Engine throughput over the last STEP_RATE_WINDOW frames: what the
        # backend could sustain, not what the window is showing -- FuncAnimation
        # holds playback at 1/DT frames/s regardless of how fast step() returns.
        # In vector mode a frame is n_envs env-steps, so this is the batch rate,
        # comparable to what benchmarks/vs_ppcf.py reports.
        window = world["recent_step_ms"]
        mean_step_ms = sum(window) / len(window)
        per_env = "" if n_envs == 1 else f", {n_envs} envs"
        rate = (f"  |  {n_envs * 1000.0 / mean_step_ms:6.1f} steps/s "
                f"({mean_step_ms:.1f}ms/frame{per_env})" if mean_step_ms > 0
                else "")

        # Live gate readout: which half of the success condition is binding.
        gate = ""
        if shown.pc_att is not None:
            in_zone, control = success_gate(shown.players, shown.ball,
                                            shown.pc_att, zone)
            if in_zone is not None:
                gate = (f"  |  in zone: {'Y' if in_zone else 'n'}  "
                        f"zone PC: {control:.2f}")

        t_render0 = time.perf_counter()
        # The env keeps its last pc_att (an offside turnover returns before the
        # PPCF call), and computes one every tick regardless because
        # intercept_pass needs the TTI cache that call builds -- so show_ppcf
        # gates only whether the field reaches the heatmap.
        env_tag = "" if n_envs == 1 else f"env {render_env}/{n_envs}  |  "
        render_frame(shown.players, shown.ball, ax=ax,
                     zone=zone if show_zone else None,
                     pc_att=shown.pc_att if show_ppcf else None,
                     title=f"{label}  |  {env_tag}ep {world['ep'][render_env]}"
                           f"  |  tick {shown.tick}  |  t = {t:5.1f}s  |  "
                           f"ball: {state}  |  R {ep_return:+5.2f}"
                           f"{rate}{gate}{ended}")
        render_ms = (time.perf_counter() - t_render0) * 1000.0

        # Note: render_ms covers building the artists, not the canvas blit that
        # matplotlib does after update() returns, so total_ms understates true
        # wall-clock per frame somewhat.
        world["step_ms"] += step_ms
        world["render_ms"] += render_ms
        return []

    # cache_frame_data=False -> don't buffer every frame (state is live/mutating)
    anim = FuncAnimation(fig, update, frames=n_ticks, interval=interval_ms,
                         blit=False, cache_frame_data=False, repeat=False)
    # keep a reference so the animation isn't garbage-collected
    fig._anim = anim

    plt.show()
    driver.close()

    n_done = sum(world["tally"].values())
    if n_done:
        print(f"\n{label}: {n_done} episodes over {n_envs} env(s), "
              f"mean {world['ticks'] / n_done:.0f} ticks")
        for k in ("success", "failure", "timeout"):
            print(f"  {k:8s} {world['tally'][k]:4d}  "
                  f"{world['tally'][k] / n_done:6.1%}")
        if world["returns"]:
            print(f"  mean return {np.mean(world['returns']):+.3f}")
        frames = max(world["frames"], 1)
        mean_ms = (world["step_ms"] + world["render_ms"]) / frames
        print(f"  mean {mean_ms:.1f}ms/frame against a {budget_ms:.0f}ms budget")
        # Engine-only rate over the whole run -- the number to quote when
        # comparing backends. Render cost sits alongside it so a slow window
        # isn't mistaken for a slow backend.
        step_s = world["step_ms"] / 1000.0
        if step_s > 0:
            print(f"  step {world['step_ms'] / frames:.2f}ms/frame -> "
                  f"{world['ticks'] / step_s:.1f} steps/s  "
                  f"(render {world['render_ms'] / frames:.1f}ms/frame, "
                  f"excluded)")
    return anim


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="scripted",
                    help='"random", "scripted", or a path to a .pt checkpoint')
    ap.add_argument("--start-holder", type=int, default=6,
                    help="attacker row that kicks off with the ball; "
                         "negative draws one per episode")
    ap.add_argument("--seed", type=int, default=245365)
    ap.add_argument("--deterministic", action="store_true",
                    help="checkpoints only: argmax every head instead of sampling")
    ap.add_argument("--no-ppcf", action="store_true",
                    help="hide the pitch-control heatmap (still computed)")
    ap.add_argument("--n-envs", type=int, default=1,
                    help="LowBlockEnvs to step per frame, under a sync vector "
                         "env; >1 fills the tally faster and makes the steps/s "
                         "readout the batch rate")
    ap.add_argument("--render-env", type=int, default=0,
                    help="which sub-env to draw when --n-envs > 1")
    args = ap.parse_args()

    run_simulation(policy=args.policy, start_holder=args.start_holder,
                   seed=args.seed, deterministic=args.deterministic,
                   show_ppcf=not args.no_ppcf, n_envs=args.n_envs,
                   render_env=args.render_env)
