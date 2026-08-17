import numpy as np

from physics.engine import (
    DRIBBLE_V_MAX,
    action_decoding,
    ball_action,
    ball_mechanics,
    cap_speed,
    direction_lookup,
    kinematics_integrator,
    speed_lookup,
)
from physics.ppcf import PPCF_grid
from schema import player_dt
from attackers.calibrate_attacker_formation import sample_attacker_formation
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

N_DIRECTIONS = len(direction_lookup)
N_SPEEDS = len(speed_lookup)

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
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }
    return players, ball, attacker_ids, rng, make_defender_state(seed)


def random_actions(n_att, rng, pass_prob=None):
    direction_idx = rng.integers(0, N_DIRECTIONS, size=n_att)
    speed_idx = rng.integers(0, N_SPEEDS, size=n_att)
    if pass_prob is None:
        # Uniform over the whole ball head. This is the honest control for the
        # probe, and it releases on ~(n_att-1)/n_att of carrying ticks.
        ball_idx = rng.integers(0, n_att, size=n_att)
    else:
        ball_idx = np.where(rng.random(n_att) < pass_prob,
                            rng.integers(1, n_att, size=n_att), 0)
    return direction_idx, speed_idx, ball_idx


def step(players, ball, attacker_ids, defender_state, tick, rng,
         ppcf_grid=None, zone=None, max_ticks=MAX_TICKS, pass_prob=None,
         pc_min=ZONE_PC_MIN):
    n_att = len(attacker_ids)
    if ppcf_grid is None:
        ppcf_grid = make_ppcf_grid()
    if zone is None:
        zone = make_zone()

    direction_idx, speed_idx, ball_idx = random_actions(n_att, rng, pass_prob)
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
                pc_min=ZONE_PC_MIN):
    players, ball, attacker_ids, rng, defender_state = make_initial_world(
        n_att, n_def, seed, start_holder=start_holder)
    if ppcf_grid is None:
        ppcf_grid = make_ppcf_grid()
    if zone is None:
        zone = make_zone()

    for tick in range(1, max_ticks + 1):
        players, ball, _pc, outcome = step(
            players, ball, attacker_ids, defender_state, tick, rng,
            ppcf_grid=ppcf_grid, zone=zone, max_ticks=max_ticks,
            pass_prob=pass_prob, pc_min=pc_min)
        if outcome is not None:
            return outcome, tick
    return TIMEOUT, max_ticks
