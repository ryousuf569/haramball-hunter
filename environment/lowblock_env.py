import sys
import numpy as np
from physics.engine import (
    DT,
    ball_action,
    ball_mechanics,
    kinematics_integrator,
    local_to_global,
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
    apply_turnover,
)
from environment.grid import PC_NX, PC_NY, PC_CELL_SIZE
from environment.termination import check_shot_opening
from attackers.baseline_attacker import compute_attacker_targets
from attackers.calibrate_attacker_formation import sample_attacker_formation
from defenders.calibrate_defender_formation import sample_defender_formation
ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

# Episode outcomes, step()'s 4th return. None means the episode is still live:
# the defenders have neither conceded the shot nor won the ball back yet.
SUCCESS = "success"   # attackers worked a shot opening
FAILURE = "failure"   # defenders forced a turnover

PITCH_CENTER_Y = 34.0


# Row 0 keeper, 1-5 backline, 6-9 midfield, 10 forward: the row layout
# compute_defender_targets indexes with GK_INDEX/BACKLINE_INDICES/etc. The 10
# outfielders are sampled from real low-block frames; those frames never
# labelled a keeper, so row 0 just starts on the resting target the policy
# would hold it at for a centred ball, rather than on a made-up sampled slot.
def _defender_formation_gk_5_4_1(rng, n_def=11):
    outfield = sample_defender_formation(rng, n_def=n_def - 1)
    gk = gk_positioning(PITCH_CENTER_Y, PITCH_CENTER_Y, 0.0001)
    return np.vstack([gk, outfield]).astype("f4")


# Rows deepest-first (2 backline, 5 midfield, 3 forward), sampled the same way
def _attacker_formation_2_5_3(rng, n_att=10):
    return sample_attacker_formation(rng, n_att=n_att).astype("f4")


# pitch-control grid. PC_NX/PC_NY/PC_CELL_SIZE are imported from environment.grid
# so termination.py can bin a position onto these cells without importing this
# module back; they stay re-exported here for the callers that already read them.
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

    # Attackers (10: a 2-5-3) start in a shape sampled from real low-block
    # freeze frames, so `seed` now actually changes the initial state instead of
    # replaying one hand-placed formation. Defenders are 11: a keeper on its
    # line plus a resting low-block 5-4-1 near the x=105 goal they defend. Row
    # layout matches defenders.py: row 0 keeper, rows 1-5 backline, rows 6-9
    # midfield, row 10 forward.
    players["position"][:n_att] = _attacker_formation_2_5_3(rng, n_att)
    players["position"][n_att:] = _defender_formation_gk_5_4_1(rng, n_def)
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
    # through every step(). The seed flows into the turnover RNG, so distinct
    # seeds give distinct duel/interception rolls rather than identical replays.
    defender_state = make_defender_state(seed=seed)
    return players, ball, attacker_ids, rng, defender_state


def step(players, ball, attacker_ids, defender_state, tick_count,
         ppcf_grid=None, exit_on_turnover=False):
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

    # 4) resolve the ball: pass decision, then flight/hold mechanics. Remember
    #    where the ball started the tick so intercept_pass can test the segment it
    #    swept rather than just where it ended up.
    prev_ball_pos = np.asarray(ball["position"], dtype=float).copy()
    pass_decision = ball_action(ball_idx, ball.get("holder_id"), attacker_ids)

    # 4a) offside, checked once at release (as RoboCup does) on the positions the
    #     pass was played from. Must come before ball_mechanics, which overwrites
    #     ball['position'] and flips the state to in_flight. An offside pass never
    #     leaves the holder's feet: the nearest defender to the intended receiver
    #     gets it, and the flight is skipped entirely. The holder is passed in
    #     because the ball's release point is their feet, and ball['position'] is
    #     still a tick stale until ball_mechanics resyncs it below.
    holder_id, is_pass, target_id = pass_decision
    if is_pass and check_offside(players, holder_id, target_id):
        target_pos = players["position"][players["id"] == target_id][0]
        winner = nearest_defender_to(players, target_pos)
        ball = apply_turnover(ball, winner)
        print(f"    TURNOVER (offside) tick {tick_count}: pass to {target_id} flagged, "
              f"defender {winner} gets it")
        if exit_on_turnover:
            sys.exit()
        pc_att = None
        if ppcf_grid is not None:
            pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])
        return players, ball, pc_att, FAILURE

    ball = ball_mechanics(ball, players, pass_decision)

    # 5) attacker pitch control on the post-integration positions/velocities.
    #    Runs before the turnover checks because it also caches each player's TTI
    #    to the ball in players['i_p'], which intercept_pass reads below.
    pc_att = None
    if ppcf_grid is not None:
        pc_att = compute_attacker_ppcf(players, ppcf_grid, ball["position"])

    # 6) turnovers. Both run after the integration and the ball resolution so they
    #    see the positions and ball state the tick actually ended on -- a defender
    #    who closes into range this tick can win it this tick. The two are mutually
    #    exclusive on ball state (held vs in_flight), so at most one can fire.
    winner = ground_duel(players, ball, defender_state["rng"], None, dt=DT)
    if winner is not None:
        ball = apply_turnover(ball, winner)
        print(f"    TURNOVER (duel) tick {tick_count}: defender {winner} won the ball")
        if exit_on_turnover:
            sys.exit()
    else:
        winner = intercept_pass(players, ball, defender_state["rng"],
                                prev_ball_pos, dt=DT)
        if winner is not None:
            ball = apply_turnover(ball, winner)
            print(f"    TURNOVER (intercept) tick {tick_count}: defender {winner} cut out the pass")
            if exit_on_turnover:
                sys.exit()

    # 7) terminate. A turnover -- from either check above, or from the offside
    #    branch that already returned -- ends the episode in failure. Otherwise
    #    an attacker on the ball, in space, with a real chance ends it in
    #    success. The shot test reads the pitch-control surface, so a run
    #    without a ppcf_grid can only ever end on a turnover.
    outcome = None
    if winner is not None:
        outcome = FAILURE
    elif pc_att is not None and check_shot_opening(players, ball, pc_att):
        outcome = SUCCESS
        print(f"    SHOT OPENING tick {tick_count}: attacker {ball['holder_id']} "
              f"is clear to shoot")

    return players, ball, pc_att, outcome
