import numpy as np
import math as m
from collections import deque

GK_INDEX = slice(0, 1)
BACKLINE_INDICES = slice(1, 6)
MIDFIELD_INDICES = slice(6, 10)
FORWARD_INDEX = slice(10, 11)
V_MAX = 5.0
A_MAX = 7.0
HARAM_DEPTH_OFFSET = 15.0

PITCH_X = 105.0
LINE_SEP = 10

# Regression-set 5%/95% clips, in the calibration frame (x=0), so mirror inputs first
ATT_LINE_MIN = 16.09
ATT_LINE_MAX = 42.25
BALL_X_MIN= 13.26
BALL_X_MAX = 57.48

SEED = 42

# Press policy. Every number here is measured in calibration/press_calibration.py;
# see calibration/README.md section 10 for the study.
# Band is signed metres vs the back line, positive = goal-side. Inside it a real
# low block has a defender within 5m of the ball 70% of the time; outside, 21%.
PRESS_BAND_FRONT = -20.0 # 4.7m in front of the midfield line
PRESS_BAND_BEHIND = 5.0 # past this the ball is through; it is the keeper's
PRESS_MAX_EXCURSION = 10.0 # leash off own slot; real press path p90 is 9.1m
PRESS_LATCH_TICKS = 25 # 2.5s; real commitment p90 is 2.44s
# Seconds of the carrier's motion the presser aims ahead of. Running at where
# the ball is now is pure pursuit, and a pursuer with the same top speed as the
# man he is chasing never closes: the press latched on 95% of ticks against a
# scripted carry and still never got inside the 1.2m ground_duel radius.
PRESS_LEAD_SECONDS = 1.0

backline_offset = np.array([[-1.77, -20], [0.93, -10], [1.42, 0], [0.80, 10], [-1.38, 20]])
midline_offset = np.array([[-0.47, -15], [0.18, -5], [0.44, 5], [-0.20, 15]])

def make_defender_state(seed=SEED):
    return {
        "ball_x_history": deque(maxlen=5),
        "attacker_line_history": deque(maxlen=5),
        "rng": np.random.default_rng(seed),
        # presser/coverer are defender-array row indices; target_key is the
        # attacker id the press is committed to (see press_target)
        "press_latch": {"presser": None, "coverer": None, "ticks_left": 0,
                        "target_key": None},
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

def gk_positioning(ball_y, pitch_center, gain):

    y = y_centroid(ball_y, pitch_center, gain=0.0001)
    pos = np.array([[104, y]])

    return pos

def apply_compactness_snapback(def_positions, target_velocities, v_max, lo=2.5, hi=16.0):
    lines = [BACKLINE_INDICES, MIDFIELD_INDICES]
    speed = 0.5 * v_max
    for line_slice in lines:
        line_pos = def_positions[line_slice]
        order = np.argsort(line_pos[:, 1])  # sort by y within the line
        sorted_idx = np.arange(len(line_pos))[order]
        line_centroid = line_pos.mean(axis=0)

        # Accumulate then clip once, so loop order doesn't decide which violating pair wins
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
                    # stacked players: no axis to separate along, so +y breaks the tie
                    axis = np.array([0.0, 1.0])
                corrections[i] -= axis * speed
                corrections[j] += axis * speed

        # clip to the same 0.5 * v_max magnitude the corrections were built at
        mag = np.linalg.norm(corrections, axis=1, keepdims=True)
        scale = np.divide(speed, mag, out=np.zeros_like(mag), where=mag > speed)
        corrections = np.where(mag > speed, corrections * scale, corrections)

        # Blend rather than replace, so a player fixing a gap still makes progress to their slot
        blended = target_velocities[line_slice] + corrections
        bmag = np.linalg.norm(blended, axis=1, keepdims=True)
        bscale = np.divide(v_max, bmag, out=np.ones_like(bmag), where=bmag > v_max)
        target_velocities[line_slice] = blended * np.minimum(bscale, 1.0)
    return target_velocities

def press_target(players, ball):
    att_ids = players["id"][players["team"] == "attacker"]

    if ball["state"] == "in_flight":
        target_id = ball.get("target_id")
        if target_id is None or not np.any(att_ids == target_id):
            return None, None
        # flight_target is frozen at release and is where ball_mechanics snaps
        # the ball on arrival, so it is the true reception point.
        return np.asarray(ball["flight_target"], dtype=float), target_id

    holder_id = ball.get("holder_id")
    if holder_id is None or not np.any(att_ids == holder_id):
        return None, None
    return players["position"][players["id"] == holder_id][0], holder_id


def press_aim(players, ball, target_key, target_pos, presser_pos, v_max):
    """Where the presser runs, leading a carrier by how long he needs to get there.

    A ball in flight already has a fixed destination, so there is nothing to
    lead; only a carrier moves the target while the presser is closing on it.
    """
    if ball["state"] != "held" or target_key is None:
        return target_pos

    vel = players["velocity"][players["id"] == target_key]
    if vel.size == 0:
        return target_pos

    gap = np.linalg.norm(np.asarray(target_pos, dtype=float) - presser_pos)
    lead = min(gap / max(v_max, 1e-6), PRESS_LEAD_SECONDS)
    return np.asarray(target_pos, dtype=float) + np.asarray(vel[0], dtype=float) * lead


def in_press_band(target_pos, back_line_x):
    rel_back = float(target_pos[0]) - float(back_line_x)
    return PRESS_BAND_FRONT <= rel_back <= PRESS_BAND_BEHIND


def apply_pressure_trigger(players, ball, def_positions, def_slots, target_velocities,
                           v_max, defender_state, back_line_x,
                           support_dist=14.0, cover_y_sep=8.0):
    # support_dist=14: the 2-5-3 never spaces players under ~5.7m, so 5m never fired
    att_mask = players["team"] == "attacker"
    latch = defender_state["press_latch"]

    # Every geometry test below reads the destination, not ball['position'] --
    # mid-flight the ball is somewhere neither team is standing.
    target_pos, target_key = press_target(players, ball)

    # Drop the latch when the ball moves on to a different attacker, or the
    # presser charges a stale position
    if latch["ticks_left"] > 0 and latch["target_key"] != target_key:
        latch["ticks_left"] = 0

    if target_pos is None:
        return target_velocities  # loose ball, or a defender has it

    if not in_press_band(target_pos, back_line_x):
        latch["ticks_left"] = 0     # ball has left the band; whoever was out comes back
        return target_velocities

    if latch["ticks_left"] > 0:
        presser = latch["presser"]
        # Break the commitment early if the ball has dragged the presser off
        if np.linalg.norm(def_positions[presser] - def_slots[presser]) > PRESS_MAX_EXCURSION:
            latch["ticks_left"] = 0

    if latch["ticks_left"] > 0:
        # Committed press: the latch fixes who presses and covers
        latch["ticks_left"] -= 1
        presser = latch["presser"]
        coverer = latch["coverer"]
    else:
        teammate_positions = players["position"][att_mask & (players["id"] != target_key)]
        if len(teammate_positions) == 0:
            return target_velocities
        support_present = np.any(np.linalg.norm(teammate_positions - target_pos, axis=1) < support_dist)
        if not support_present:
            return target_velocities

        # Selection is by SLOT distance, not body distance
        slot_d = np.linalg.norm(def_slots - target_pos, axis=1)

        # The keeper holds its line; it never presses or covers, so drop it out of both argmins
        slot_d[GK_INDEX] = np.inf

        presser = int(np.argmin(slot_d))
        if slot_d[presser] > PRESS_MAX_EXCURSION:
            return target_velocities    # nobody's zone: the block holds shape

        # Supporter is the next closest, kept off the presser's toes by the same spacing rule
        cover_d = slot_d.copy()
        cover_d[presser] = np.inf
        y_sep = np.abs(def_slots[:, 1] - def_slots[presser, 1])
        eligible = y_sep >= cover_y_sep
        eligible[presser] = False
        if np.any(eligible):
            cover_d[~eligible] = np.inf
        coverer = int(np.argmin(cover_d))

        # commit this pair for the next PRESS_LATCH_TICKS ticks
        latch["presser"] = presser
        latch["coverer"] = coverer
        latch["ticks_left"] = PRESS_LATCH_TICKS
        latch["target_key"] = target_key

    aim = press_aim(players, ball, target_key, target_pos,
                    def_positions[presser], v_max)
    to_target = aim - def_positions[presser]
    d = np.linalg.norm(to_target)
    if d > 1e-6:
        target_velocities[presser] = (to_target / d) * v_max

    # Coverer shifts halfway to the ball instead of charging, keeping shape behind the press
    if coverer is not None:
        cover_point = 0.5 * (target_pos + def_positions[coverer])
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
    targets[GK_INDEX] = gk_positioning(ball["position"][1], 34, 0.0001)
    targets[BACKLINE_INDICES] = back_centroid + backline_offset
    targets[MIDFIELD_INDICES] = mid_centroid + midline_offset
    targets[FORWARD_INDEX] = attacker_positioning(ball["position"][1], 34, 0.7)

    to_target = targets - defender_positions
    dist = np.linalg.norm(to_target, axis=1, keepdims=True)
    unit = np.divide(to_target, dist, out=np.zeros_like(to_target), where=dist > 1e-6)

    # Speed tapers with distance so defenders decelerate into their slot instead of overshooting
    speed = np.sqrt(2 * A_MAX * dist)
    speed = np.minimum(speed, V_MAX)
    speed[dist < 0.4] = 0

    target_velocities = unit * speed
    target_velocities = apply_compactness_snapback(defender_positions, target_velocities, V_MAX)
    target_velocities = apply_pressure_trigger(players, ball, defender_positions, targets,
                                               target_velocities, V_MAX, defender_state,
                                               back_line_x)

    return target_velocities
