import numpy as np
import math as m
from collections import deque

BACKLINE_INDICES = slice(0, 5)
MIDFIELD_INDICES = slice(5, 9)
FORWARD_INDEX = slice(9, 10)
V_MAX = 5.0
A_MAX = 7.0
HARAM_DEPTH_OFFSET = 15.0

PITCH_X = 105.0
LINE_SEP = 10

# Pulled from the regression set for clipping 5% and 95% quantile
# These are in the calibration frame (defending x=0), so mirror inputs before clipping
ATT_LINE_MIN = 16.09
ATT_LINE_MAX = 42.25
BALL_X_MIN= 13.26
BALL_X_MAX = 57.48

backline_offset = np.array([[-1.77, -20], [0.93, -10], [1.42, 0], [0.80, 10], [-1.38, 20]])
midline_offset = np.array([[-0.47, -15], [0.18, -5], [0.44, 5], [-0.20, 15]])

def make_defender_state():
    return {
        "ball_x_history": deque(maxlen=5),
        "attacker_line_history": deque(maxlen=5),
    }

def y_centroid(ball_y, pitch_center, gain):
    return pitch_center + gain * (ball_y - pitch_center)

def calculate_depth_ref(defender_state, ball_x, attacker_line_x):

    defender_state["ball_x_history"].append(ball_x)
    defender_state["attacker_line_history"].append(attacker_line_x)
    oldest_ball_x = defender_state["ball_x_history"][0]
    oldest_attacker_line = defender_state["attacker_line_history"][0]

    back_line_x = PITCH_X - (16.070 + (0.8144 * oldest_attacker_line) - (0.1067 * oldest_ball_x)) + HARAM_DEPTH_OFFSET
    mid_line_x = back_line_x - LINE_SEP

    return mid_line_x, back_line_x

def attacker_positioning(ball_y, pitch_center, gain):

    y = y_centroid(ball_y, pitch_center, gain=0.4)
    pos = np.array([[52, y]])

    return pos

def apply_compactness_snapback(def_positions, target_velocities, v_max, lo=2.5, hi=16.0):
    lines = [BACKLINE_INDICES, MIDFIELD_INDICES]
    speed = 0.5 * v_max
    for line_slice in lines:
        line_pos = def_positions[line_slice]
        order = np.argsort(line_pos[:, 1])  # sort by y within the line
        sorted_idx = np.arange(len(line_pos))[order]
        line_centroid = line_pos.mean(axis=0)

        # A player can sit in two violating pairs at once (too close to one
        # neighbour, too far from the other). Accumulate every correction and
        # clip once at the end, so which pair "wins" isn't decided by loop order.
        corrections = np.zeros_like(line_pos)

        for k in range(len(sorted_idx) - 1):
            i, j = sorted_idx[k], sorted_idx[k + 1]
            gap = abs(line_pos[j, 1] - line_pos[i, 1])

            if gap > hi:
                # too spread out: pull both back toward the line centroid
                for idx in (i, j):
                    to_centroid = line_centroid - line_pos[idx]
                    d = np.linalg.norm(to_centroid)
                    if d > 1e-6:
                        corrections[idx] += (to_centroid / d) * speed

            elif gap < lo:
                i_to_j = line_pos[j] - line_pos[i]
                d = np.linalg.norm(i_to_j)
                if d > 1e-6:
                    axis = i_to_j / d
                else:
                    # players are stacked: no axis to separate along, so fall
                    # back to +y by sorted order to break the tie deterministically
                    axis = np.array([0.0, 1.0])
                corrections[i] -= axis * speed
                corrections[j] += axis * speed

        # clip to the same 0.5 * v_max magnitude, then override only the
        # players this pass actually had something to say about
        mag = np.linalg.norm(corrections, axis=1, keepdims=True)
        scale = np.divide(speed, mag, out=np.zeros_like(mag), where=mag > speed)
        corrections = np.where(mag > speed, corrections * scale, corrections)

        # Blend the spacing correction into the existing target-seeking velocity
        # instead of replacing it, so a player in a spacing violation still makes
        # progress toward their slot instead of freezing to fix the gap.
        blended = target_velocities[line_slice] + corrections
        bmag = np.linalg.norm(blended, axis=1, keepdims=True)
        bscale = np.divide(v_max, bmag, out=np.ones_like(bmag), where=bmag > v_max)
        target_velocities[line_slice] = blended * np.minimum(bscale, 1.0)
    return target_velocities

def apply_pressure_trigger(players, ball, def_positions, target_velocities, v_max,
                            box_x_min=88.5, box_y=(13.84, 54.16), trigger_dist=15.0, support_dist=5.0,
                            cover_y_sep=8.0):
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

    # Presser comes from midfield (they sit in front of the block), coverer from the
    # backline. Picking the global nearest two always drew both from the backline --
    # it sits ~10m closer to a deep holder -- which vacated the middle of the block.
    presser = MIDFIELD_INDICES.start + np.argmin(dists_to_holder[MIDFIELD_INDICES])

    back_d = dists_to_holder[BACKLINE_INDICES].copy()
    y_sep = np.abs(def_positions[BACKLINE_INDICES, 1] - def_positions[presser, 1])
    eligible = y_sep >= cover_y_sep
    if np.any(eligible):
        back_d[~eligible] = np.inf
    coverer = BACKLINE_INDICES.start + np.argmin(back_d)

    to_holder = holder_pos - def_positions[presser]
    d = np.linalg.norm(to_holder)
    if d > 1e-6:
        target_velocities[presser] = (to_holder / d) * v_max

    # The coverer shifts to a point between the holder and their own slot rather than
    # charging the ball, so the block keeps its shape behind the press.
    cover_point = 0.5 * (holder_pos + def_positions[coverer])
    to_cover = cover_point - def_positions[coverer]
    d = np.linalg.norm(to_cover)
    if d > 1e-6:
        target_velocities[coverer] = (to_cover / d) * (0.6 * v_max)

    return target_velocities

def compute_defender_targets(players, ball, defender_state):

    defender_mask = players["team"] == "defender"
    defender_positions = players["position"][defender_mask]

    attacker_mask = players["team"] == "attacker"
    attacker_positions = players["position"][attacker_mask]

    attacker_x = attacker_positions[:, 0]
    distances = np.abs(attacker_x - PITCH_X)
    closest_indices = np.argsort(distances)[:4]
    attacker_line = np.mean(attacker_x[closest_indices])

    attacker_line = np.clip(PITCH_X - attacker_line, ATT_LINE_MIN, ATT_LINE_MAX)
    ball_x_clipped = np.clip(PITCH_X - ball["position"][0], BALL_X_MIN, BALL_X_MAX)

    mid_line_x, back_line_x = calculate_depth_ref(defender_state, ball_x_clipped, attacker_line)
    mid_dy = y_centroid(ball["position"][1], pitch_center=34, gain=0.31)
    back_dy = y_centroid(ball["position"][1], pitch_center=34, gain=0.19)

    back_centroid = np.array([back_line_x, back_dy])
    mid_centroid = np.array([mid_line_x, mid_dy])

    targets = np.zeros_like(defender_positions)
    targets[BACKLINE_INDICES] = back_centroid + backline_offset
    targets[MIDFIELD_INDICES] = mid_centroid + midline_offset
    targets[FORWARD_INDEX] = attacker_positioning(ball["position"][1], 34, 0.7)

    to_target = targets - defender_positions
    dist = np.linalg.norm(to_target, axis=1, keepdims=True)
    unit = np.divide(to_target, dist, out=np.zeros_like(to_target), where=dist > 1e-6)

    # speed calculation: constraint that allows the defenders not to overshoot in their targets
    # before defenders would go a constant speed even if the target was 0.1m away, causing an overshoot
    # in position

    speed = np.sqrt(2 * A_MAX * dist)
    speed = np.minimum(speed, V_MAX)
    speed[dist < 0.4] = 0

    target_velocities = unit * speed

    target_velocities = apply_compactness_snapback(defender_positions, target_velocities, V_MAX)
    target_velocities = apply_pressure_trigger(players, ball, defender_positions, target_velocities, V_MAX)

    return target_velocities
