import numpy as np
from schema import player_dt

V_MAX = 5.0
A_MAX = 7.0
DT = 0.1

offset = np.array([43.0, 0.0])

def global_to_local(pos):
    return pos - offset

def local_to_global(pos):
    return pos + offset

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