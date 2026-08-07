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


def pass_speed(length):
    length = np.maximum(np.asarray(length, dtype=float), 1e-6)
    return np.minimum(PASS_SPEED_A * length ** PASS_SPEED_B, PASS_SPEED_MAX)

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

def ball_mechanics(ball, players, pass_decision):
    holder_id, is_pass, target_id = pass_decision
    ball = dict(ball)  # copy, don't mutate caller's dict

    if ball['state'] == 'held':
        if is_pass:
            # holder just released it -- freeze start/target, switch to flight
            start_pos = get_position_by_id(players, holder_id)
            target_pos = get_position_by_id(players, target_id)

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
            ball['holder_id'] = ball['target_id']
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

    new_position[:, 0] = np.clip(new_position[:, 0], 0.0, 105.0)
    new_position[:, 1] = np.clip(new_position[:, 1], 0.0, 68.0)

    players = players.copy()
    players['velocity'] = new_velocity
    players['position'] = new_position

    return players