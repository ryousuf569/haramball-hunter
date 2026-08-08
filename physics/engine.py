import numpy as np
from schema import player_dt

V_MAX = 5.0
A_MAX = 7.0
DT = 0.1
HOLD = 0

# Pass speed scales with pass length. Fitted to 793 Metrica passes in
# physics/validation/pass_speed_calibration.py; see that script for the fit, the
# held-out check and the caveat about per-pass scatter.
PASS_SPEED_A = 4.5292
PASS_SPEED_B = 0.3537
PASS_SPEED_MAX = 14.93   # median speed of real passes over 25m; the fit is capped here
BALL_SPEED = PASS_SPEED_MAX   # scalar upper bound, for callers that need one

# Placement error, as a fraction of pass length, and how far the intended
# receiver may be from where the ball lands and still collect it. Without these
# a pass is a teleport: the ball snapped onto the target's release-time position
# and changed hands however far the target had since run, which made a long ball
# strictly safer than keeping it.
PASS_SIGMA = 0.03
RECEPTION_RADIUS = 3.0

PITCH_X = 105.0
PITCH_Y = 68.0


def pass_speed(length):
    length = np.maximum(np.asarray(length, dtype=float), 1e-6)
    return np.minimum(PASS_SPEED_A * length ** PASS_SPEED_B, PASS_SPEED_MAX)


def pass_scatter(length, rng):
    if rng is None:
        return np.zeros(2)
    return rng.normal(0.0, PASS_SIGMA * float(length), size=2)


def receiver_at(players, position, target_id):
    # The intended target gets it if it is still near the landing spot, else
    # whoever is closest does -- which may be a defender.
    target_pos = get_position_by_id(players, target_id)
    if np.linalg.norm(np.asarray(target_pos, dtype=float) - position) <= RECEPTION_RADIUS:
        return target_id
    delta = np.asarray(players['position'], dtype=float) - position
    return int(players['id'][np.argmin(np.einsum('ij,ij->i', delta, delta))])

ball = {
    'state': str,
    'holder_id': int | None,
    'position': list[int],
    'target_id': int | None,
    'flight_start': list[int],
    'flight_target': list[int],
}

offset = np.array([43.0, 0.0])
root_2 = np.sqrt(2)
direction_lookup = np.array([
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
    [root_2 / 2, root_2 / 2],
    [root_2 / 2, -root_2 / 2],
    [-root_2 / 2, root_2 / 2],
    [-root_2 / 2, -root_2 / 2],
    [0, 0],
])

speed_lookup = np.array([0.3 * V_MAX, 0.6 * V_MAX, 1.0 * V_MAX])

def get_position_by_id(players, player_id):
    return players['position'][players['id'] == player_id][0]

def global_to_local(pos):
    return pos - offset

def local_to_global(pos):
    return pos + offset

def action_decoding(direction_idx, speed_idx):

    target_directions = direction_lookup[direction_idx]
    target_speeds = speed_lookup[speed_idx]
    target_velocities = target_directions * target_speeds[:, np.newaxis]

    return target_velocities

def ball_action(ball_idx, holder_id, player_ids):

    holder_is_attacker = holder_id is not None and np.any(player_ids == holder_id)

    if holder_is_attacker:
        ball_idx = np.where(player_ids == holder_id, ball_idx, HOLD)
        holder_choice = ball_idx[player_ids == holder_id][0]
        is_pass = holder_choice != HOLD

        if is_pass:
            teammate_ids = np.sort(player_ids[player_ids != holder_id])
            target_id = teammate_ids[holder_choice - 1]
            return holder_id, True, target_id
        else:
            return holder_id, False, None
    else:
        return holder_id, False, None

def ball_mechanics(ball, players, pass_decision, rng=None):
    holder_id, is_pass, target_id = pass_decision
    ball = dict(ball)  # copy, don't mutate caller's dict

    if ball['state'] == 'held':
        if is_pass:
            # holder just released it -- freeze start/target, switch to flight
            start_pos = get_position_by_id(players, holder_id)
            target_pos = get_position_by_id(players, target_id)
            length = np.linalg.norm(np.asarray(target_pos, dtype=float)
                                    - np.asarray(start_pos, dtype=float))
            target_pos = (np.asarray(target_pos, dtype=float)
                          + pass_scatter(length, rng)).astype('f4')

            ball['state'] = 'in_flight'
            ball['holder_id'] = None
            ball['target_id'] = target_id
            ball['flight_start'] = start_pos
            ball['flight_target'] = target_pos
            ball['position'] = start_pos
        else:
            # still held -- resync position to holder's current (post-kinematics) position
            ball['position'] = get_position_by_id(players, holder_id)

    elif ball['state'] == 'in_flight':
        remaining_vector = ball['flight_target'] - ball['position']
        remaining_distance = np.linalg.norm(remaining_vector)
        # Speed is a property of the pass, not of the tick, so it comes off the
        # frozen start/target rather than off how far the ball has left to run --
        # otherwise the ball would decelerate as it approached the receiver.
        pass_length = np.linalg.norm(ball['flight_target'] - ball['flight_start'])
        step_distance = float(pass_speed(pass_length)) * DT

        if remaining_distance <= step_distance:
            # arrives this tick -- snap, don't overshoot
            ball['position'] = ball['flight_target']
            ball['state'] = 'held'
            ball['holder_id'] = receiver_at(
                players, np.asarray(ball['flight_target'], dtype=float),
                ball['target_id'])
            ball['target_id'] = None
        else:
            direction = remaining_vector / remaining_distance
            ball['position'] = ball['position'] + direction * step_distance

    return ball

def kinematics_integrator(players, target_velocities):

    gap = target_velocities - players['velocity']
    gap_norm = np.linalg.norm(gap, axis=1)

    max_step = A_MAX * DT
    needs_scaling = gap_norm > max_step

    new_velocity = np.copy(target_velocities)
    new_velocity[needs_scaling] = players['velocity'][needs_scaling] + (gap[needs_scaling] / gap_norm[needs_scaling, None]) * max_step

    speed = np.linalg.norm(new_velocity, axis=1)
    over_max = speed > V_MAX
    new_velocity[over_max] = (new_velocity[over_max] / speed[over_max, None]) * V_MAX

    acceleration_realized = (new_velocity - players['velocity']) / DT
    new_position = players['position'] + players['velocity'] * DT + 0.5 * acceleration_realized * DT**2

    # Clip position AND kill the velocity component that ran into the line.
    # Clipping alone left a player pinned at x=105 advertising 5 m/s it was not
    # travelling, and made the boundary a free place to park.
    for axis, hi in ((0, PITCH_X), (1, PITCH_Y)):
        outside = (new_position[:, axis] < 0.0) | (new_position[:, axis] > hi)
        new_velocity[outside, axis] = 0.0
        new_position[:, axis] = np.clip(new_position[:, axis], 0.0, hi)

    players = players.copy()
    players['velocity'] = new_velocity
    players['position'] = new_position

    return players