import numpy as np
from functools import partial
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from physics.engine import (
    DRIBBLE_V_MAX,
    action_decoding,
    ball_action,
    ball_mechanics,
    cap_speed,
    kinematics_integrator,
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
from environment.termination import ZONE_PC_MIN, check_success, make_zone

ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

SUCCESS = "success"
FAILURE = "failure"
TIMEOUT = "timeout"

MAX_TICKS = 250


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
    pass

def make_vector_env(n_envs=6, asynchronous=False, seed=None,
                    autoreset_mode=None, **env_kwargs):
    fns = [partial(LowBlockEnv, **env_kwargs) for _ in range(n_envs)]
    cls = AsyncVectorEnv if asynchronous else SyncVectorEnv
    kwargs = {} if autoreset_mode is None else {"autoreset_mode": autoreset_mode}
    venv = cls(fns, **kwargs)
    if seed is not None:
        venv.reset(seed=seed)
    return venv