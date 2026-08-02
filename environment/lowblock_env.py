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
from defenders.defenders import make_defender_state, compute_defender_targets
from defenders.turnover import (
    ground_duel,
    intercept_pass,
    check_offside,
    nearest_defender_to,
    apply_turnover,
)
from attackers.baseline_attacker import compute_attacker_targets
from attackers.calibrate_attacker_formation import sample_attacker_formation
ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"


def _defender_formation_5_4_1(n_def=10, pitch_width=68.0):
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


def _attacker_formation_2_5_3(rng, n_att=10):
    """Draw a 2-5-3 attacker shape from the StatsBomb-calibrated model.

    This used to be a fixed offset table around a fixed reference point, so
    every episode opened on the identical picture and an agent could memorise
    it rather than learn anything. Now the centroid depth, the lateral position,
    how stretched the shape is and each player's slot offset are all sampled
    from real low-block freeze frames -- see
    attackers/calibrate_attacker_formation.py for the fit.

    Rows come back deepest-first (2 backline, 5 midfield, 3 forward), the same
    back-to-front ordering the old table used, so baseline_attacker's line
    slices still line up with the roster rows.
    """
    return sample_attacker_formation(rng, n_att=n_att).astype("f4")


# pitch-control grid
PC_NX, PC_NY = 31, 34
PC_CELL_SIZE = 2.0


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


def make_initial_world(n_att=10, n_def=10, seed=11, start_holder=0):
    rng = np.random.default_rng(seed)

    players = np.zeros(n_att + n_def, dtype=player_dt)
    players["id"] = np.arange(1, n_att + n_def + 1)  # ids are 1-based, HOLD==0
    players["team"][:n_att] = ATTACKER_LABEL
    players["team"][n_att:] = DEFENDER_LABEL

    # Attackers (10: a 2-5-3) start in a shape sampled from real low-block
    # freeze frames, so `seed` now actually changes the initial state instead of
    # replaying one hand-placed formation. Defenders sit in a resting low-block
    # 5-4-1 near the x=105 goal they defend. Row layout matches defenders.py:
    # rows 0-4 backline, rows 5-8 midfield, row 9 forward.
    players["position"][:n_att] = _attacker_formation_2_5_3(rng, n_att)
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
    #     gets it, and the flight is skipped entirely.
    _holder_id, is_pass, target_id = pass_decision
    if is_pass and check_offside(players, target_id):
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
        return players, ball, pc_att

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

    return players, ball, pc_att
