import sys
from functools import partial

import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from physics.engine import (
    DT,
    DRIBBLE_V_MAX,
    V_MAX,
    action_decoding,
    ball_action,
    ball_mechanics,
    cap_speed,
    direction_lookup,
    kinematics_integrator,
    local_to_global,
    speed_lookup,
)
from schema import player_dt
from physics.ppcf import PPCF_grid
from defenders.defenders import (
    make_defender_state,
    compute_defender_targets,
    gk_positioning,
)
from defenders.turnover import (
    ground_duel,
    intercept_pass,
    check_offside,
    nearest_defender_to,
    offside_line,
    apply_turnover,
)
from environment.grid import PC_NX, PC_NY, PC_CELL_SIZE
from environment.termination import check_shot_opening, pcf_in_area
from environment.reward import (
    RewardConfig,
    agent_shaping,
    build_zone_masks,
    make_pcf_state,
    reset_agent_potential,
    reset_potential_from_pc_att,
    step_reward_from_pc_att,
    zone_value,
)
from attackers.calibrate_attacker_formation import sample_attacker_formation
from defenders.calibrate_defender_formation import sample_defender_formation


# Deferred so deleting the throwaway baseline can't make the env unimportable.
def compute_attacker_targets(players, ball, tick_count):
    from attackers.baseline_attacker import (
        compute_attacker_targets as _baseline)
    return _baseline(players, ball, tick_count)


ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

# Episode outcomes, step()'s 4th return. None means the episode is still live.
SUCCESS = "success"   # attackers worked a shot opening
FAILURE = "failure"   # defenders forced a turnover
TIMEOUT = "timeout"   # step() has no horizon; LowBlockEnv raises this at max_ticks

PITCH_CENTER_Y = 34.0
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_EXTENT_XY = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype="f4")

# Per-agent action: (direction, speed, ball). The first two index engine's
# lookup tables; the third is 0 == HOLD or the k-th sorted teammate id, so it
# has n_att entries rather than n_att - 1.
N_DIRECTIONS = len(direction_lookup)   # 9: eight headings plus stand
N_SPEEDS = len(speed_lookup)           # 3: 0.3 / 0.6 / 1.0 of V_MAX
ACTION_HEADS = 3


def ball_actions(n_att):
    return n_att


# Widths as functions of the roster, so a different n_att still lines up.
def obs_dim(n_players, n_att):
    # 4 ego + 4*n_players roster + 2 ball + 1 in-flight + 1 clock
    # + 3 nearest-defender + 1 local pitch control + 2 zone pitch control
    # + 2 offside (line, own margin) + n_att agent one-hot
    return 4 + 4 * n_players + 4 + 8 + n_att


# Distance at which a defender stops mattering, for the nearest-defender
# feature. Raw metres over the pitch diagonal would squash every distance that
# matters (0-15 m) into the bottom tenth of the range; clipping here spends the
# whole [0, 1] on the band where pressure actually changes behaviour.
PRESSURE_RADIUS = 25.0


def obs_clock_idx(n_players):
    """Column of the remaining-time feature in build_obs' layout.

    It used to be the last column; the pitch-control tail now sits after it, so
    anything reading obs[:, -1] for the clock is reading pc_hs instead.
    """
    return 4 + 4 * n_players + 3


def state_dim(n_players, n_att):
    # 4*n_players roster + 2 ball + n_att holder one-hot + 2 PC + 1 clock
    return 4 * n_players + 2 + n_att + 3


# obs() and global_state() both go through these. If they ever diverge, actor
# and critic see different units and it presents as a value loss that plateaus.
def norm_pos(pos):
    """Pitch coordinates -> [0, 1]^2 (the integrator already clips to pitch)."""
    return (np.asarray(pos, dtype="f4") / PITCH_EXTENT_XY).astype("f4")


def norm_vel(vel):
    """Velocities -> [-1, 1]^2 (the integrator already caps speed at V_MAX)."""
    return (np.asarray(vel, dtype="f4") / np.float32(V_MAX)).astype("f4")


def remaining_time(tick, max_ticks):
    """Fraction of the horizon left, in [0, 1]. Not redundant: timeout is a
    terminal here, not a truncation, so the same position is worth very
    different amounts at tick 10 and tick 299. Without the clock the two are one
    input and the env stops being Markov in the observation -- Pardo et al.,
    Time Limits in Reinforcement Learning (2018). Do not delete it."""
    return float(np.clip(1.0 - tick / max_ticks, 0.0, 1.0))


# Row 0 keeper, 1-5 backline, 6-9 midfield, 10 forward -- the layout
# compute_defender_targets indexes. The sampled frames never labelled a keeper,
# so row 0 starts on its resting target for a centred ball.
def _defender_formation_gk_5_4_1(rng, n_def=11):
    outfield = sample_defender_formation(rng, n_def=n_def - 1)
    gk = gk_positioning(PITCH_CENTER_Y, PITCH_CENTER_Y, 0.0001)
    return np.vstack([gk, outfield]).astype("f4")


# Rows deepest-first (2 backline, 5 midfield, 3 forward), sampled the same way.
def _attacker_formation_2_5_3(rng, n_att=10):
    return sample_attacker_formation(rng, n_att=n_att).astype("f4")


# PC_NX/PC_NY/PC_CELL_SIZE live in environment.grid so termination.py can bin a
# position without importing this module back; re-exported here for callers.
def make_ppcf_grid():
    i = np.arange(PC_NX)
    j = np.arange(PC_NY)
    ii, jj = np.meshgrid((i + 0.5) * PC_CELL_SIZE, (j + 0.5) * PC_CELL_SIZE,
                         indexing="ij")
    grid_local = np.stack([ii, jj], axis=-1).reshape(-1, 2)  # (1054, 2)
    return local_to_global(grid_local)


# Global-frame extent of the grid, for imshow/pcolormesh overlays.
_pc_corners = local_to_global(np.array([[0.0, 0.0],
                                        [PC_NX * PC_CELL_SIZE,
                                         PC_NY * PC_CELL_SIZE]]))
PC_EXTENT = (_pc_corners[0, 0], _pc_corners[1, 0],
             _pc_corners[0, 1], _pc_corners[1, 1])


def nearest_defender_feats(players, n_att):
    """(n_att, 3) float32: [unit vector toward nearest defender, scaled distance].

    Rows n_att: are the defenders -- the same fixed roster order obs() uses.

    A unit vector plus a separate magnitude rather than a raw relative vector:
    the direction stays fully informative at any range, and the magnitude gets
    its own clipped scale (see PRESSURE_RADIUS). The raw coordinates are already
    in the observation, so this adds no information the network could not in
    principle derive -- it adds it in the form the decision actually uses,
    which is the difference between learnable and learnable in this lifetime.
    """
    att = np.asarray(players["position"][:n_att], dtype=float)
    dfn = np.asarray(players["position"][n_att:], dtype=float)

    delta = dfn[None, :, :] - att[:, None, :]        # (n_att, n_def, 2)
    dist = np.linalg.norm(delta, axis=-1)            # (n_att, n_def)

    rows = np.arange(len(att))
    j = dist.argmin(axis=1)
    v, d = delta[rows, j], dist[rows, j]

    # d == 0 means a defender is standing exactly on the attacker; the zero
    # vector is the honest answer for "which way", same guard as TTI_vec.
    unit = v / np.where(d > 0, d, 1.0)[:, None]
    scaled = np.minimum(d, PRESSURE_RADIUS) / PRESSURE_RADIUS
    return np.concatenate([unit, scaled[:, None]], axis=1).astype("f4")


# Metres either side of the offside line that the margin feature resolves.
# Beyond this the exact number stops mattering: you are either comfortably
# onside or comfortably off.
OFFSIDE_MARGIN_SCALE = 20.0


def holder_row_of(ball, attacker_ids):
    """Attacker row holding the ball, or None. Row index, not player id."""
    holder_id = ball.get("holder_id")
    if holder_id is None:
        return None
    rows = np.flatnonzero(np.asarray(attacker_ids) == holder_id)
    return int(rows[0]) if rows.size else None


def ball_action_mask(players, ball, attacker_ids, n_att):
    """Legal ball actions, bool (n_att, ball_actions), True == legal.

    Index 0 is HOLD and is legal on every row; index k is the k-th sorted
    teammate id, matching engine.ball_action. Non-holders get HOLD only -- one
    legal action rather than none, because an all-False row sends a masked
    softmax to NaN while a one-hot row gives log-prob 0 and entropy 0 for free.

    The holder additionally loses any target the pass would be flagged offside
    for. That is not a convenience: 43% of all passing losses were offside
    turnovers, a flat tax the policy cannot avoid because the line is the
    second-largest defender x and nothing in the observation used to say so.
    Masking them turns passing from uniformly worse than carrying into the
    better option under pressure (crossover at ~4m of defender distance). A real
    player does not attempt an obviously offside pass either.
    """
    mask = np.zeros((n_att, ball_actions(n_att)), dtype=bool)
    mask[:, 0] = True

    row = holder_row_of(ball, attacker_ids)
    if row is None:
        return mask

    mask[row, :] = True
    holder_id = int(np.asarray(attacker_ids)[row])
    mates = np.sort(np.asarray(attacker_ids)[np.asarray(attacker_ids) != holder_id])
    for k, target_id in enumerate(mates):
        if check_offside(players, holder_id, int(target_id)):
            mask[row, k + 1] = False
    return mask


def offside_feats(players, n_att):
    """(n_att, 2) float32: the offside line, and each attacker's margin to it.

    Column 0 is the line in normalised pitch x, shared by every row -- an
    attacker cannot make a run to stay onside without knowing where the line is.
    Column 1 is its own signed margin, positive when onside, scaled the same way
    as the nearest-defender distance and for the same reason: the raw
    coordinates are already in the observation, but not in the form the decision
    uses.
    """
    line = offside_line(players)
    own_x = np.asarray(players["position"][:n_att, 0], dtype=float)
    margin = np.clip((line - own_x) / OFFSIDE_MARGIN_SCALE, -1.0, 1.0)
    col = np.full(n_att, line / PITCH_LENGTH, dtype="f4")
    return np.stack([col, margin.astype("f4")], axis=1)


def build_obs(players, ball, n_att, tick, max_ticks, pc_att, f3_mask, hs_mask,
              normalization="mean"):
    """Per-agent observations, float32 (n_att, obs_dim). THE one implementation.

    Row i is attacker i:
        [0:4]     own position, own velocity
        [4:88]    all 21 players, norm_pos block then norm_vel block, fixed
                  roster order (attackers :n_att, then defenders)
        [88:90]   ball position
        [90]      ball in-flight flag
        [91]      remaining time
        [92:95]   nearest defender: unit vector (2), scaled distance (1)
        [95]      local attacker pitch control, mean over the cells within
                  termination.AREA_RADIUS of this attacker
        [96:98]   pc_f3, pc_hs -- the two zone values Phi is built from
        [98:100]  offside line (shared), own signed margin to it
        [100:110] agent one-hot -- which of the n_att rows this is

    Everything from index 4 to 92 is shared across rows; the ego prefix and the
    last six are per-agent (bar the two zone values). The ego prefix is what
    lets one set of weights act as ten agents, and duplicates the self slot
    because the network cannot otherwise find it.

    The one-hot is what makes a formation representable at all. Without it the
    shared weights are a pure function of where you are standing, so two
    attackers at the same point are guaranteed the same action distribution and
    nothing can assign ten players to ten distinct roles.

    pc_att is the (PC_NX, PC_NY) attacker-control surface for the state being
    observed -- the same one step() already computes for the shot test, so the
    new features cost no extra pitch-control call. Callers must pass the
    surface matching THIS state; a stale one silently teaches the wrong thing.

    LowBlockEnv.obs and learned_env.obs both route here. They used to be
    hand-kept clones, which is a standing invitation for the inference path to
    feed a policy a layout it was not trained on.
    """
    pos = norm_pos(players["position"])
    vel = norm_vel(players["velocity"])

    flat_pc = np.asarray(pc_att, dtype=float).reshape(-1)
    shared = np.concatenate([
        pos.reshape(-1),
        vel.reshape(-1),
        norm_pos(ball["position"]).reshape(-1),
        np.array([ball["state"] == "in_flight",
                  remaining_time(tick, max_ticks)], dtype="f4"),
    ])
    zones = np.array([zone_value(flat_pc, f3_mask, normalization),
                      zone_value(flat_pc, hs_mask, normalization)], dtype="f4")

    ego = np.concatenate([pos[:n_att], vel[:n_att]], axis=1)
    near = nearest_defender_feats(players, n_att)
    local_pc = pcf_in_area(
        np.asarray(players["position"][:n_att], dtype=float),
        np.asarray(pc_att, dtype=float).reshape(PC_NX, PC_NY),
    ).astype("f4")[:, None]

    tail = np.broadcast_to(zones, (n_att, zones.size))
    agent_id = np.eye(n_att, dtype="f4")
    return np.concatenate(
        [ego, np.broadcast_to(shared, (n_att, shared.size)),
         near, local_pc, tail, offside_feats(players, n_att), agent_id],
        axis=1).astype("f4")


def compute_attacker_ppcf(players, ppcf_grid, ball_pos):
    result = PPCF_grid(ppcf_grid, players, ball_pos)  # (n_cells, n_players)
    is_att = players["team"] == ATTACKER_LABEL
    return result[:, is_att].sum(axis=1).reshape(PC_NX, PC_NY)


def make_initial_world(n_att=10, n_def=11, seed=11, start_holder=0):
    rng = np.random.default_rng(seed)

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(1, n_att + n_def + 1)  # ids are 1-based, HOLD==0
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL

    # Both shapes are sampled from real low-block freeze frames, so `seed`
    # actually changes the initial state. Defenders are a keeper on its line
    # plus a resting 5-4-1 near the x=105 goal they defend.
    players["position"][:n_att] = _attacker_formation_2_5_3(rng, n_att)
    players["position"][n_att:] = _defender_formation_gk_5_4_1(rng, n_def)
    players["velocity"][:] = 0.0

    attacker_ids = players["id"][:n_att]
    # Starting holder by attacker row index, clipped into range.
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

    # Scripted-defender state (a short ball_x history used to lag the block's
    # depth reference), created once and threaded through every step(). The seed
    # also flows into the turnover RNG, so seeds give distinct duel rolls.
    defender_state = make_defender_state(seed=seed)
    return players, ball, attacker_ids, rng, defender_state


def step(players, ball, attacker_ids, defender_state, tick_count,
         ppcf_grid=None, exit_on_turnover=False, attacker_velocities=None,
         attacker_ball_idx=None, verbose=True):
    """Advance the world one tick.

    attacker_velocities (n_att, 2) and attacker_ball_idx (n_att,) are the seam
    the learned policy drives: supply both and baseline_attacker.py is not
    consulted at all. Pass neither to keep the baseline. The defenders are
    scripted by design and have no such seam.

    verbose: print turnover/shot events. LowBlockEnv turns this off.
    """
    att_mask = players["team"] == ATTACKER_LABEL
    def_mask = players["team"] == DEFENDER_LABEL

    # Roster-wide targets, attackers-first then defenders, matching both
    # policies' row ordering.
    target_velocities = np.zeros((len(players), 2), dtype="f4")

    # 1) attackers: the policy when supplied, else the baseline. Both or
    #    neither -- one alone would silently fall back and discard it. The
    #    integrator clamps to A_MAX/V_MAX, so shape is the only requirement.
    if (attacker_velocities is None) != (attacker_ball_idx is None):
        raise ValueError("pass attacker_velocities and attacker_ball_idx "
                         "together, or neither")
    if attacker_velocities is None:
        attacker_velocities, attacker_ball_idx = compute_attacker_targets(
            players, ball, tick_count)
    ball_idx = attacker_ball_idx
    target_velocities[att_mask] = attacker_velocities

    # 1b) whoever is carrying the ball is capped below V_MAX. Applied to the
    #     target, not the integrator, so acceleration and everything else is
    #     unchanged and the cap lifts the tick the ball leaves his feet.
    carrier_id = ball.get("holder_id") if ball["state"] == "held" else None
    if carrier_id is not None:
        carrier = np.flatnonzero(players["id"] == carrier_id)
        if carrier.size:
            target_velocities[carrier[0]] = cap_speed(
                target_velocities[carrier[0]], DRIBBLE_V_MAX)

    # 2) defenders run their script, in defender-row order (rows n_att:).
    defender_velocities = compute_defender_targets(players, ball, defender_state)
    target_velocities[def_mask] = defender_velocities

    # 3) integrate everyone with the combined targets.
    players = kinematics_integrator(players, target_velocities)

    # 4) resolve the ball. Keep where it started so intercept_pass can test the
    #    segment it swept, not just where it ended up.
    prev_ball_pos = np.asarray(ball["position"], dtype=float).copy()
    pass_decision = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)

    # 4a) offside, checked once at release (as RoboCup does), before
    #     ball_mechanics overwrites ball['position']. An offside pass never
    #     leaves the holder's feet: the nearest defender to the intended
    #     receiver gets it and the flight is skipped.
    holder_id, is_pass, target_id = pass_decision
    if is_pass and check_offside(players, holder_id, target_id):
        target_pos = players["position"][players["id"] == target_id][0]
        winner = nearest_defender_to(players, target_pos)
        ball = apply_turnover(ball, winner)
        if verbose:
            print(f"    TURNOVER (offside) tick {tick_count}: pass to {target_id} flagged, "
                  f"defender {winner} gets it")
        if exit_on_turnover:
            sys.exit()
        pc_att = None
        if ppcf_grid is not None:
            pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])
        return players, ball, pc_att, FAILURE

    # The rng gives the pass its placement error; a scattered ball can land
    # nearer a defender than the intended receiver, which is what ball_mechanics
    # resolves. Same generator as the duels, so one seed still fixes the episode.
    ball = ball_mechanics(ball, players, pass_decision, defender_state["rng"])

    # 5) attacker pitch control on the post-integration state. Runs before the
    #    turnover checks because it also caches each player's TTI to the ball in
    #    players['i_p'], which intercept_pass reads.
    pc_att = None
    if ppcf_grid is not None:
        pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])

    # 6) turnovers, on the state the tick actually ended on -- a defender who
    #    closes into range this tick can win it this tick. The two are mutually
    #    exclusive on ball state (held vs in_flight), so at most one fires.
    winner = ground_duel(players, ball, defender_state["rng"], None, dt=DT)
    if winner is not None:
        ball = apply_turnover(ball, winner)
        if verbose:
            print(f"    TURNOVER (duel) tick {tick_count}: defender {winner} won the ball")
        if exit_on_turnover:
            sys.exit()
    else:
        winner = intercept_pass(players, ball, defender_state["rng"],
                                prev_ball_pos, dt=DT)
        if winner is not None:
            ball = apply_turnover(ball, winner)
            if verbose:
                print(f"    TURNOVER (intercept) tick {tick_count}: defender {winner} cut out the pass")
            if exit_on_turnover:
                sys.exit()

    # A pass that arrived this tick can have been collected by a defender rather
    # than cut out mid-flight -- ball_mechanics gives it to whoever is nearest
    # the landing spot. Neither turnover test above sees that, so check it here.
    if winner is None and ball["state"] == "held":
        holder = ball.get("holder_id")
        if holder is not None and players["team"][players["id"] == holder][0] == DEFENDER_LABEL:
            winner = int(holder)
            if verbose:
                print(f"    TURNOVER (loose pass) tick {tick_count}: defender {winner} collected it")
            if exit_on_turnover:
                sys.exit()

    # 7) terminate. The shot test reads the pitch-control surface, so a run
    #    without a ppcf_grid can only ever end on a turnover.
    outcome = None
    if winner is not None:
        outcome = FAILURE
    elif pc_att is not None and check_shot_opening(players, ball, pc_att):
        outcome = SUCCESS
        if verbose:
            print(f"    SHOT OPENING tick {tick_count}: attacker {ball['holder_id']} "
                  f"is clear to shoot")

    return players, ball, pc_att, outcome


# Bound here so LowBlockEnv.step can call it without the method name shadowing it.
world_step = step


class LowBlockEnv(gym.Env):
    """Gymnasium wrapper: one episode = one low-block possession.

    The reward is environment.reward's potential-based shaping, fed the PPCF
    surface step() already builds for the shot test, so it costs no extra
    pitch-control call. The attackers are the learners; the low block is
    scripted and stays that way.

    All ten attackers share one set of weights, so the interface is per-agent
    and batched on a leading n_att axis:
        obs()                 (n_att, 92) float32, actor input
        action                (n_att, 3) MultiDiscrete: direction, speed, ball
        info["state"]         99-dim global vector, critic input
        info["action_mask"]   (n_att, ball_actions) bool, ball head only

    reset() returns state and mask too: a rollout needs V(s_0) and a legal first
    action before it has taken a step.

    baseline_attacker.py is a defender test harness, not a policy. With
    scripted_attackers=False -- how training, the tests and the benchmark all
    run -- it is bypassed entirely and never imported.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_att=10, n_def=11, max_ticks=300, cfg=None,
                 scripted_attackers=True):
        super().__init__()
        self.n_att = n_att
        self.n_def = n_def
        self.max_ticks = max_ticks
        self.cfg = cfg if cfg is not None else RewardConfig()
        self.scripted_attackers = scripted_attackers

        # Built once per env, not per episode: fixed geometry, and masks that
        # are a pure function of it.
        self.ppcf_grid = make_ppcf_grid()
        self.f3_mask, self.hs_mask = build_zone_masks(self.ppcf_grid)
        self.pcf_state = make_pcf_state()

        n_players = n_att + n_def
        self.n_players = n_players
        self.ball_actions = ball_actions(n_att)
        self.obs_dim = obs_dim(n_players, n_att)
        self.state_dim = state_dim(n_players, n_att)

        # Bounds are the unit box, not pitch metres -- see norm_pos/norm_vel.
        ego_lo = [0.0, 0.0, -1.0, -1.0]
        ego_hi = [1.0, 1.0, 1.0, 1.0]
        pos_lo, pos_hi = np.zeros(2 * n_players), np.ones(2 * n_players)
        vel_lo, vel_hi = -np.ones(2 * n_players), np.ones(2 * n_players)
        # Tail, in build_obs order: nearest-defender unit vector (2, in
        # [-1, 1]), its scaled distance (1), local pitch control (1), then
        # pc_f3/pc_hs -- unbounded above for the same reason as the critic's,
        # normalization="area" puts them in square metres.
        near_lo, near_hi = [-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]
        pc_lo, pc_hi = [0.0, 0.0, 0.0], [1.0, np.inf, np.inf]
        # offside line in normalised x, then a signed margin in [-1, 1]
        off_lo, off_hi = [0.0, -1.0], [1.0, 1.0]
        row_lo = np.concatenate([ego_lo, pos_lo, vel_lo, [0.0, 0.0], [0.0],
                                 [0.0], near_lo, pc_lo, off_lo,
                                 np.zeros(n_att)])
        row_hi = np.concatenate([ego_hi, pos_hi, vel_hi, [1.0, 1.0], [1.0],
                                 [1.0], near_hi, pc_hi, off_hi,
                                 np.ones(n_att)])
        self.observation_space = gym.spaces.Box(
            low=np.tile(row_lo, (n_att, 1)).astype("f4"),
            high=np.tile(row_hi, (n_att, 1)).astype("f4"),
            dtype=np.float32)

        # Not a Gymnasium-required space -- the vector wrappers ignore it -- but
        # the critic's input width has to be documented somewhere. The two PC
        # entries are unbounded above because normalization="area" puts them in
        # square metres; under the "mean" default they sit in [0, 1].
        state_lo = np.concatenate([pos_lo, vel_lo, [0.0, 0.0],
                                   np.zeros(n_att), [0.0, 0.0], [0.0]])
        state_hi = np.concatenate([pos_hi, vel_hi, [1.0, 1.0],
                                   np.ones(n_att), [np.inf, np.inf], [1.0]])
        self.state_space = gym.spaces.Box(
            low=state_lo.astype("f4"), high=state_hi.astype("f4"),
            dtype=np.float32)

        # One row per attacker; shared weights, so every row has the same nvec.
        self.action_space = gym.spaces.MultiDiscrete(
            np.tile([N_DIRECTIONS, N_SPEEDS, self.ball_actions], (n_att, 1)))

        self.players = None
        self.ball = None
        self.attacker_ids = None
        self.defender_state = None
        self.tick = 0
        # Cached off the reward's parts dict so global_state() needs no second
        # pitch-control call.
        self.pc_f3 = 0.0
        self.pc_hs = 0.0
        # The full (PC_NX, PC_NY) surface, kept so obs() can read local pitch
        # control off it. reset() fills it before the first obs() call.
        self.pc_att = None

    def remaining_time(self):
        return remaining_time(self.tick, self.max_ticks)

    def obs(self):
        """See build_obs -- this is a thin binding of it to the env's state.

        self.pc_att is the surface from the step that produced the current
        state, so the observation's pitch-control features describe the state
        the policy is about to act in, not the one before it.
        """
        return build_obs(self.players, self.ball, self.n_att, self.tick,
                         self.max_ticks, self.pc_att, self.f3_mask,
                         self.hs_mask, self.cfg.normalization)

    def holder_row(self):
        """Attacker row holding the ball, or None if it is in flight or a
        defender has it. Row index, not player id."""
        holder_id = self.ball.get("holder_id")
        if holder_id is None:
            return None
        rows = np.flatnonzero(self.attacker_ids == holder_id)
        return int(rows[0]) if rows.size else None

    def global_state(self):
        """The critic's input, float32 (state_dim,), not per-agent:

            [0:84]   all 21 players, same blocks and order as obs()
            [84:86]  ball position
            [86:96]  holder one-hot over attackers, all zero if no attacker has it
            [96:98]  pc_f3, pc_hs -- the reward's zone values, already computed
            [98]     remaining time

        Slots are fixed by roster order and never sorted by anything that varies
        within an episode: a slot that changes meaning mid-episode makes the
        critic's input non-stationary.
        """
        pos = norm_pos(self.players["position"])
        vel = norm_vel(self.players["velocity"])
        holder = np.zeros(self.n_att, dtype="f4")
        row = self.holder_row()
        if row is not None:
            holder[row] = 1.0
        return np.concatenate([
            pos.reshape(-1),
            vel.reshape(-1),
            norm_pos(self.ball["position"]).reshape(-1),
            holder,
            np.array([self.pc_f3, self.pc_hs, self.remaining_time()],
                     dtype="f4"),
        ]).astype("f4")

    def action_mask(self):
        """See ball_action_mask -- this is a thin binding of it to the env."""
        return ball_action_mask(self.players, self.ball, self.attacker_ids,
                                self.n_att)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # make_initial_world seeds the formations and the turnover RNG off one
        # integer; drawing it from self.np_random is what makes reset(seed=...)
        # reproducible and successive episodes distinct.
        ep_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        holder = int(self.np_random.integers(0, self.n_att))
        (self.players, self.ball, self.attacker_ids,
         _rng, self.defender_state) = make_initial_world(
            n_att=self.n_att, n_def=self.n_def, seed=ep_seed,
            start_holder=holder)
        self.tick = 0

        # Phi(s0) comes from the initial state, which no step() has touched --
        # hence the one extra PPCF call per episode. Skipping it would break the
        # telescoping identity environment/reward_readme.md rests on.
        pc_att = compute_attacker_ppcf(self.players, self.ppcf_grid,
                                       self.ball["position"])
        phi_0, parts = reset_potential_from_pc_att(
            pc_att, self.f3_mask, self.hs_mask, self.pcf_state, self.cfg)
        self.pc_f3 = parts["pc_f3"]
        self.pc_hs = parts["pc_hs"]
        self.pc_att = pc_att
        reset_agent_potential(self.pcf_state, pc_att,
                              self.players["position"][:self.n_att])

        return self.obs(), {"phi_0": phi_0, "pc_f3": parts["pc_f3"],
                            "pc_hs": parts["pc_hs"], "tick": self.tick,
                            "state": self.global_state(),
                            "action_mask": self.action_mask(),
                            "agent_reward": np.zeros(self.n_att, dtype="f4")}

    def decode_action(self, action):
        """(n_att, 3) of (direction, speed, ball) -> the two arrays step() wants.

        Direction and speed go through engine.action_decoding, so a policy
        target velocity is expressible in the same set the physics was
        calibrated on.
        """
        a = np.asarray(action, dtype=np.int64).reshape(self.n_att, ACTION_HEADS)
        vel = action_decoding(a[:, 0], a[:, 1]).astype("f4")

        # ball_action re-masks non-holder slots anyway; zeroing them here keeps
        # the output readable and matches what action_mask() advertises.
        ball_idx = np.zeros(self.n_att, dtype=int)
        row = self.holder_row()
        if row is not None:
            ball_idx[row] = int(a[row, 2])
        return vel, ball_idx

    def step(self, action):
        avel, ball_idx = (None, None)
        if not self.scripted_attackers:
            avel, ball_idx = self.decode_action(action)

        self.players, self.ball, pc_att, outcome = world_step(
            self.players, self.ball, self.attacker_ids, self.defender_state,
            self.tick, ppcf_grid=self.ppcf_grid, attacker_velocities=avel,
            attacker_ball_idx=ball_idx, verbose=False)
        self.tick += 1

        # step() has no horizon of its own; the timeout is imposed here.
        if outcome is None and self.tick >= self.max_ticks:
            outcome = TIMEOUT

        reward, parts = step_reward_from_pc_att(
            pc_att, self.f3_mask, self.hs_mask, self.pcf_state, self.cfg,
            outcome=outcome)

        # All three outcomes terminate, timeout included: reward.py already
        # zeroes Phi(terminal) for it, so reporting it as truncated would invite
        # a learner to bootstrap V(s_T) on top and break the telescoping
        # identity. truncated stays False.
        terminated = outcome is not None
        truncated = False

        self.pc_f3 = parts["pc_f3"]
        self.pc_hs = parts["pc_hs"]
        # world_step only returns a surface when it was given a grid; it always
        # is here, but keep the last one rather than write None into obs().
        if pc_att is not None:
            self.pc_att = pc_att
        parts["tick"] = self.tick
        parts["state"] = self.global_state()
        parts["action_mask"] = self.action_mask()
        # Per-attacker shaping, on the same surface. It rides alongside the
        # scalar reward rather than inside it: the team reward stays exactly
        # what reward_readme.md describes, and the learner adds this per agent.
        parts["agent_reward"] = agent_shaping(
            self.pcf_state, self.pc_att,
            self.players["position"][:self.n_att], self.cfg,
            terminal=terminated).astype("f4")
        return self.obs(), float(reward), terminated, truncated, parts


def make_vector_env(n_envs=6, asynchronous=False, seed=None,
                    autoreset_mode=None, **env_kwargs):
    """SyncVectorEnv (serial, one process) or AsyncVectorEnv (one process each).

    partial rather than a closure so the factories pickle for the Async spawn.

    autoreset_mode: Gymnasium defaults to AutoresetMode.NEXT_STEP, where the
        call after a terminal one is a reset that ignores its action and pays
        reward 0 -- a junk transition a learner has to mask out. SAME_STEP
        resets on the terminal call instead, with the real terminal reward and
        the final state in info["final_obs"]. Training should prefer SAME_STEP.
    """
    fns = [partial(LowBlockEnv, **env_kwargs) for _ in range(n_envs)]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    kwargs = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    venv = cls(fns, **kwargs)
    if seed is not None:
        venv.reset(seed=seed)
    return venv
