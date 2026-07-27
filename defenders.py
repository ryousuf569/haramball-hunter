import numpy as np
from collections import deque

BACKLINE_INDICES = slice(0, 5)
MIDFIELD_INDICES = slice(5, 9)
FORWARD_INDEX = slice(9, 10)

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

    mid_line_x = min(oldest_ball_x, 60)
    back_line_x = mid_line_x + 18

    return mid_line_x, back_line_x

def attacker_positioning(ball_y, pitch_center, gain):

    y = y_centroid(ball_y, pitch_center, gain)
    pos = np.array([[52, y]])

    return pos