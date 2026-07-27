import numpy as np
from collections import deque

BACKLINE_INDICES = slice(0, 5)
MIDFIELD_INDICES = slice(5, 9)
FORWARD_INDEX = slice(9, 10)
V_MAX = 5.0

backline_offset = np.array([[0, -20], [0, -10], [0, 0], [0, 10], [0, 20]])
midline_offset = np.array([[0, -15], [0, -5], [0, 5], [0, 15]])

def make_defender_state():
    return {
        "ball_x_history": deque(maxlen=5)
    }

def y_centroid(ball_y, pitch_center, gain):
    return pitch_center + gain * (ball_y - pitch_center)

def calculate_depth_ref(defender_state, ball_x):

    defender_state["ball_x_history"].append(ball_x)
    oldest_ball_x = defender_state["ball_x_history"][0]

    mid_line_x = max(oldest_ball_x, 70)
    back_line_x = mid_line_x + 10

    return mid_line_x, back_line_x

def attacker_positioning(ball_y, pitch_center, gain):

    y = y_centroid(ball_y, pitch_center, gain=0.4)
    pos = np.array([[52, y]])

    return pos

def apply_compactness_snapback(def_positions, target_velocities, v_max, lo=8.0, hi=12.0):
    lines = [BACKLINE_INDICES, MIDFIELD_INDICES] 
    for line_slice in lines:
        line_pos = def_positions[line_slice]
        order = np.argsort(line_pos[:, 1])  # sort by y within the line
        sorted_idx = np.arange(len(line_pos))[order]
        for k in range(len(sorted_idx) - 1):
            i, j = sorted_idx[k], sorted_idx[k + 1]
            gap = abs(line_pos[j, 1] - line_pos[i, 1])
            if gap < lo or gap > hi:
                line_centroid = line_pos.mean(axis=0)
                for idx in (i, j):
                    to_centroid = line_centroid - line_pos[idx]
                    d = np.linalg.norm(to_centroid)
                    if d > 1e-6:
                        global_idx = line_slice.start + idx
                        target_velocities[global_idx] = (to_centroid / d) * (0.5 * v_max)
    return target_velocities

def apply_pressure_trigger(players, ball, def_positions, target_velocities, v_max,
                            box_x_min=88.5, box_y=(13.84, 54.16), trigger_dist=15.0, support_dist=5.0):
    holder_id = ball.get("holder_id")
    att_mask = players["team"] == "attacker"
    if holder_id is None or not np.any(players["id"][att_mask] == holder_id):
        return target_velocities  # no attacker holds the ball right now

    holder_pos = players["position"][players["id"] == holder_id][0]
    dist_to_box = max(0.0, box_x_min - holder_pos[0])  # 0 if already inside/past the box edge in x
    # (spec says "within 15m of the penalty area" -- treating this as x-distance to the box's near edge,
    #  clipped at 0 once inside; revisit if y-out-of-box-width cases matter for your scenarios)

    if dist_to_box > trigger_dist:
        return target_velocities

    teammate_positions = players["position"][att_mask & (players["id"] != holder_id)]
    if len(teammate_positions) == 0:
        return target_velocities
    support_present = np.any(np.linalg.norm(teammate_positions - holder_pos, axis=1) < support_dist)
    if not support_present:
        return target_velocities

    dists_to_holder = np.linalg.norm(def_positions - holder_pos, axis=1)
    nearest_two = np.argsort(dists_to_holder)[:2]
    for idx in nearest_two:
        to_holder = holder_pos - def_positions[idx]
        d = np.linalg.norm(to_holder)
        if d > 1e-6:
            target_velocities[idx] = (to_holder / d) * v_max
    return target_velocities

def compute_defender_targets(players, ball, defender_state):

    defender_mask = players["team"] == "defender"
    defender_positions = players["position"][defender_mask]

    mid_line_x, back_line_x = calculate_depth_ref(defender_state, ball["position"][0])
    dy = y_centroid(ball["position"][1], pitch_center=34, gain=0.2)

    back_centroid = np.array([back_line_x, dy])
    mid_centroid = np.array([mid_line_x, dy])

    targets = np.zeros_like(defender_positions)
    targets[BACKLINE_INDICES] = back_centroid + backline_offset
    targets[MIDFIELD_INDICES] = mid_centroid + midline_offset
    targets[FORWARD_INDEX] = attacker_positioning(ball["position"][1], 34, 0.7)

    to_target = targets - defender_positions
    dist = np.linalg.norm(to_target, axis=1, keepdims=True)
    unit = np.divide(to_target, dist, out=np.zeros_like(to_target), where=dist > 1e-6)
    target_velocities = unit * V_MAX

    target_velocities = apply_compactness_snapback(defender_positions, target_velocities, V_MAX)
    target_velocities = apply_pressure_trigger(players, ball, defender_positions, target_velocities, V_MAX)

    return target_velocities
