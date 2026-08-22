import numpy as np
from functools import partial
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from physics.engine import (
    V_MAX,
    DRIBBLE_V_MAX, 
    PITCH_X, 
    PITCH_Y,
    action_decoding,
    ball_action,
    ball_mechanics,
    cap_speed,
    kinematics_integrator,
    direction_lookup,
    speed_lookup
)
from physics.ppcf import PPCF_grid
from schema import player_dt
from attackers.calibrate_attacker_formation import sample_attacker_formation
from attackers.random_policy import random_actions
from defenders.calibrate_defender_formation import sample_defender_formation
from defenders.defenders import (
    compute_defender_targets,
    gk_positioning,
    make_defender_state,
)
from defenders.turnover import (
    apply_turnover,
    check_offside,
    ground_duel,
    intercept_pass,
    nearest_defender_to,
)
from environment.grid import PC_NX, PC_NY, make_ppcf_grid
from environment.termination import (ZONE_PC_MIN, check_success, make_zone,
                                     zone_control)
from defenders.turnover import HALFWAY_X, offside_line

ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

SUCCESS = "success"
FAILURE = "failure"
TIMEOUT = "timeout"

MAX_TICKS = 250

N_CONSTRAINTS = 4 # reserved slots, zeros until the Lagrangian layer exists
REL_SCALE = 30.0 # metres, for relative positions
DIST_SCALE = 50.0 # metres, for scalar distances
BALL_SPEED_CAP = 15.0  # reception resync can spike ball speed to ~30
OFFSIDE_MARGIN_SCALE = 20.0
POS_SCALE = np.array([PITCH_X, PITCH_Y], dtype="f4")


def obs_dim(n_att, n_def):
    return (2 + 2 # own position, own velocity
            + 5 * (n_att - 1) # teammates: rel pos, rel vel, is-carrier
            + 4 * n_def # defenders: rel pos, rel vel, fixed roster order
            + 7 # ball block
            + 2 # zone: distance, zone control
            + 3 # offside block
            + 1 # clock
            + N_CONSTRAINTS # reserved constraint rates
            + n_att) # agent one-hot


def make_initial_world(n_att=10, n_def=11, seed=0, start_holder=0):
    rng = np.random.default_rng(seed)

    att_pos = sample_attacker_formation(rng, n_att=n_att)
    # defenders.py indexes the roster GK-first; the sampler only fits outfielders
    outfield = sample_defender_formation(rng, n_def=n_def - 1)
    gk = gk_positioning(float(np.mean(att_pos[:, 1])), 34, 0.0001)
    def_pos = np.vstack([gk, outfield])

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(n_att + n_def, dtype="int16")
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL
    players["position"][:n_att] = att_pos
    players["position"][n_att:] = def_pos

    attacker_ids = players["id"][:n_att].copy()
    ball = {
        "state": "held",
        "holder_id": int(attacker_ids[start_holder]),
        "position": players["position"][start_holder].copy(),
        "velocity": np.zeros(2, dtype="f4"),
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }
    return players, ball, attacker_ids, rng, make_defender_state(seed)


def step(players, ball, attacker_ids, defender_state, tick, rng,
         ppcf_grid=None, zone=None, max_ticks=MAX_TICKS, pass_prob=None,
         pc_min=ZONE_PC_MIN, actions=None):
    n_att = len(attacker_ids)
    if ppcf_grid is None:
        ppcf_grid = make_ppcf_grid()
    if zone is None:
        zone = make_zone()

    # actions is (direction_idx, speed_idx, ball_idx); None draws them randomly.
    if actions is None:
        actions = random_actions(n_att, rng, pass_prob)
    direction_idx, speed_idx, ball_idx = actions
    att_targets = action_decoding(direction_idx, speed_idx)

    holder_id = ball.get("holder_id")
    carrying = holder_id is not None and bool(np.any(attacker_ids == holder_id))
    if carrying:
        holder_row = int(np.flatnonzero(players["id"] == holder_id)[0])
        att_targets[holder_row] = cap_speed(att_targets[holder_row], DRIBBLE_V_MAX)

    pass_decision = ball_action(ball_idx, holder_id, attacker_ids)
    _, is_pass, target_id = pass_decision

    # Offside is judged on the kick, so it goes between ball_action and
    # ball_mechanics -- the pass never actually leaves.
    if is_pass and check_offside(players, holder_id, target_id):
        stealer = nearest_defender_to(players, ball["position"])
        return players, apply_turnover(ball, stealer), None, FAILURE

    def_targets = compute_defender_targets(players, ball, defender_state)
    target_velocities = np.vstack([att_targets, def_targets]).astype("f4")

    prev_ball_pos = np.asarray(ball["position"], dtype=float).copy()
    players = kinematics_integrator(players, target_velocities)
    ball = ball_mechanics(ball, players, pass_decision, rng)

    # Populates players['i_p'] (TTI to the ball), which intercept_pass reads.
    per_player = PPCF_grid(ppcf_grid, players,
                           np.asarray(ball["position"], dtype=float))
    att_cols = players["team"] == ATTACKER_LABEL
    pc_att = per_player[:, att_cols].sum(axis=1).reshape(PC_NX, PC_NY)

    stealer = intercept_pass(players, ball, rng, prev_ball_pos)
    if stealer is None:
        stealer = ground_duel(players, ball, rng, None)
    if stealer is not None:
        return players, apply_turnover(ball, stealer), pc_att, FAILURE

    # A completed pass can land on a defender via receiver_at.
    holder_id = ball.get("holder_id")
    if (ball["state"] == "held" and holder_id is not None
            and not np.any(attacker_ids == holder_id)):
        return players, ball, pc_att, FAILURE

    if check_success(players, ball, pc_att, zone, pc_min=pc_min):
        return players, ball, pc_att, SUCCESS

    if tick >= max_ticks:
        return players, ball, pc_att, TIMEOUT

    return players, ball, pc_att, None


def run_episode(n_att=10, n_def=11, seed=0, start_holder=0,
                max_ticks=MAX_TICKS, pass_prob=None, ppcf_grid=None, zone=None,
                pc_min=ZONE_PC_MIN, policy=None):
    # policy is a callable (players, ball, attacker_ids) -> actions, or None for
    # random. Scripted policies carry per-episode state, so build a fresh one
    # per call -- see scripted_policy.make_policy.
    players, ball, attacker_ids, rng, defender_state = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)
    if ppcf_grid is None:
        ppcf_grid = make_ppcf_grid()
    if zone is None:
        zone = make_zone()

    for tick in range(1, max_ticks + 1):
        actions = None if policy is None else policy(players, ball, attacker_ids)
        players, ball, _pc, outcome = step(
            players, ball, attacker_ids, defender_state, tick, rng,
            ppcf_grid=ppcf_grid, zone=zone, max_ticks=max_ticks,
            pass_prob=pass_prob, pc_min=pc_min, actions=actions)
        if outcome is not None:
            return outcome, tick
    return TIMEOUT, max_ticks

world_step = step

class LowBlockEnv(gym.Env):

    def __init__(self, n_att=10, n_def=11, max_tick=200):
        super().__init__()
        self.n_att, self.n_def = n_att, n_def
        self.max_ticks = max_tick
        self.obs_dim = obs_dim(n_att, n_def)

        nvec = np.tile([len(direction_lookup), len(speed_lookup), n_att], (n_att, 1))
        self.action_space = gym.spaces.MultiDiscrete(nvec)
        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(n_att, self.obs_dim), dtype=np.float32)

        self.ppcf_grid = make_ppcf_grid()
        self.zone = make_zone()
        self.zone_centre = np.asarray(self.zone.centre, dtype="f4")
        self.eye = np.eye(n_att, dtype="f4")

        self.players = None
        self.ball = None
        self.attacker_ids = None
        self.defender_state = None
        self.rng = None
        self.pc_att = None
        self.tick = 0

    def obs(self):
        n = self.n_att
        pos = np.asarray(self.players["position"], dtype="f4")
        vel = np.asarray(self.players["velocity"], dtype="f4")
        att_p, att_v = pos[:n], vel[:n]
        def_p, def_v = pos[n:], vel[n:]

        holder_id = self.ball.get("holder_id")
        is_carrier = (np.zeros(n, dtype="f4") if holder_id is None else
                      (self.players["id"][:n] == holder_id).astype("f4"))
        keep = ~np.eye(n, dtype=bool)
        rel_p = (att_p[None, :, :] - att_p[:, None, :])[keep].reshape(n, n - 1, 2)
        rel_v = (att_v[None, :, :] - att_v[:, None, :])[keep].reshape(n, n - 1, 2)
        mate_carrier = np.broadcast_to(is_carrier[None, :], (n, n))[keep].reshape(n, n - 1, 1)
        mates = np.concatenate(
            [rel_p / REL_SCALE, rel_v / V_MAX, mate_carrier], axis=2)

        dp = def_p[None, :, :] - att_p[:, None, :]
        dv = def_v[None, :, :] - att_v[:, None, :]
        defs = np.concatenate([dp / REL_SCALE, dv / V_MAX], axis=2)

        b_p = np.asarray(self.ball["position"], dtype="f4")
        b_v = cap_speed(np.asarray(self.ball["velocity"], dtype="f4"), BALL_SPEED_CAP)
        rel_b = b_p[None, :] - att_p
        ball_blk = np.concatenate([
            rel_b / REL_SCALE,
            np.linalg.norm(rel_b, axis=1, keepdims=True) / DIST_SCALE,
            np.broadcast_to(b_v / BALL_SPEED_CAP, (n, 2)),
            np.full((n, 1), float(self.ball["state"] == "in_flight"), dtype="f4"),
            is_carrier[:, None],
        ], axis=1)

        # Distance only, no direction: the zone centre never moves, so its
        # bearing is recoverable from own position, which is already in here.
        rel_z = self.zone_centre[None, :] - att_p
        zone_blk = np.concatenate([
            np.linalg.norm(rel_z, axis=1, keepdims=True) / DIST_SCALE,
            np.full((n, 1), zone_control(self.pc_att, self.zone), dtype="f4"),
        ], axis=1)

        line = offside_line(self.players)
        margin = att_p[:, 0] - line # positive = beyond the line
        off_blk = np.stack([
            np.full(n, line / PITCH_X, dtype="f4"),
            np.clip(margin / OFFSIDE_MARGIN_SCALE, -1.0, 1.0),
            ((margin > 0.0) & (att_p[:, 0] > HALFWAY_X)).astype("f4"),
        ], axis=1)

        out = np.concatenate([
            att_p / POS_SCALE,
            att_v / V_MAX,
            mates.reshape(n, -1),
            defs.reshape(n, -1),
            ball_blk,
            zone_blk,
            off_blk,
            np.full((n, 1), self.tick / self.max_ticks, dtype="f4"),
            np.zeros((n, N_CONSTRAINTS), dtype="f4"), # reserved
            self.eye,
        ], axis=1).astype("f4")

        assert out.shape == (n, self.obs_dim), (out.shape, self.obs_dim)
        return out

def make_vector_env(n_envs=6, asynchronous=False, seed=None,
                    autoreset_mode=None, **env_kwargs):
    fns = [partial(LowBlockEnv, **env_kwargs) for _ in range(n_envs)]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    kwargs = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    venv = cls(fns, **kwargs)
    if seed is not None:
        venv.reset(seed=seed)
    return venv