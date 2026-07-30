import numpy as np

from physics.tti import intercept_probability_vec

# RoboCup 2D turnover models, scaled to haramball-hunter
# RoboCup defaults: tackle_dist=2.0, tackle_width=1.25, tackle_exponent=6,
# tackle_back_dist=0. Its units are a coarser sim; the transferable parts are
# the aspect ratio (2.0/1.25 = 1.6) and the exponent, not the absolute size.

DUEL_A       = 1.20   # semi-axis along defender heading (m)
DUEL_B       = 0.50   # semi-axis across heading (m)  ~= A/1.6
DUEL_A_BACK  = 0.40   # reduced reach behind the defender (RoboCup uses 0; 0.5*A here)
DUEL_EXP     = 6      # RoboCup tackle_exponent. soft-edged box, hard cutoff
R_MAX        = 1.20   # bounding-circle prefilter = max(A, B, A_BACK)
LAM_MAX      = 2.1    # hazard rate (1/s) for a perfect defender: -ln(1-0.65)/0.5s
ADV_FLOOR    = 0.5    # lambda multiplier at worst closing speed; 1.0 at best
V_MAX        = 5.0    # mirrors engine.V_MAX; kept local so defenders/ never imports engine
EPS          = 1e-6

# RoboCup has no interception model to copy
KICKABLE_AREA   = 1.085  # RoboCup player_size + ball_size + kickable_margin = 0.3 + 0.085 + 0.7
INTERCEPT_P_MIN = 0.80   # control probability that counts as a won interception. CALIBRATED
BALL_SPEED      = 15.0   # mirrors engine.BALL_SPEED, kept local for the same reason as V_MAX
LOB_DIST        = 25.0   # passes longer than this would be lofted in real play

def ground_duel(players, ball, rng, holder_idx, dt=0.1):

    if ball["state"] != "held": 
        return None
    
    holder_id = ball.get("holder_id")

    if holder_id is None: 
        return None

    if holder_idx is None:
        holder_idx = np.flatnonzero(players["id"] == holder_id)
        if holder_idx.size == 0: 
            return None
        holder_idx = holder_idx[0]

    if players["team"][holder_idx] != "attacker": 
        return None

    dmask = players["team"] == "defender"
    dpos = players["position"][dmask]
    delta = dpos - ball["position"] 

    sq = np.einsum("ij,ij->i", delta, delta)
    if not (sq <= R_MAX * R_MAX).any():
        return None

    # 1v1, so only the closest defender contests it and everything below is scalar
    cand = np.argmin(sq)

    to_ball = -delta[cand]
    dist = np.sqrt(sq[cand])
    v_def = players["velocity"][dmask][cand]
    speed = np.sqrt(v_def @ v_def)

    # RoboCup uses body frame; we have no facing, so direction of travel stands in
    if speed > EPS:
        u = v_def / speed
        dx = to_ball @ u
        dy = to_ball[0] * -u[1] + to_ball[1] * u[0]
    else:
        # standing still: no heading to rotate into, so use plain radial distance
        dx, dy = dist, 0.0

    # RoboCup's super-ellipse: exponent 6 holds p near 1, then cuts off at the edge
    reach = DUEL_A if dx >= 0.0 else DUEL_A_BACK
    fail_prob = (abs(dx) / reach) ** DUEL_EXP + (abs(dy) / DUEL_B) ** DUEL_EXP
    p_geometry = 1.0 - fail_prob
    if p_geometry <= 0.0:
        return None

    # to_ball points defender -> ball, so a positive projection means the gap is closing
    v_holder = players["velocity"][holder_idx]
    closing = (to_ball @ (v_def - v_holder)) / max(dist, EPS)
    advantage = np.clip(closing / V_MAX, -1.0, 1.0)
    multiplier = ADV_FLOOR + (1.0 - ADV_FLOOR) * 0.5 * (advantage + 1.0)

    # RoboCup rolls once per command; we test every tick, so hazard-rate it for dt independence
    lam = LAM_MAX * p_geometry * multiplier
    p_tick = 1.0 - np.exp(-lam * dt)

    if rng.random() < p_tick:
        return int(players["id"][dmask][cand])
    return None

def intercept_pass(players, ball, rng, prev_pos, dt=0.1):
    if ball["state"] != "in_flight":
        return None

    lob = ball["flight_target"] - ball["flight_start"]
    if np.sqrt(lob @ lob) > LOB_DIST:
        return None

    dmask = players["team"] == "defender"
    dpos = players["position"][dmask]

    # Distance to the segment the ball SWEPT this tick, not just to where it landed
    a = np.asarray(prev_pos, dtype=float)
    ab = np.asarray(ball["position"], dtype=float) - a
    L2 = ab @ ab
    if L2 > EPS:
        # clamp to [0, 1] so this stays on the ground actually covered this tick
        t = np.clip(((dpos - a) @ ab) / L2, 0.0, 1.0)
        closest = a + t[:, None] * ab
    else:
        closest = a  # ball didn't move: segment is a point

    delta = dpos - closest
    sq = np.einsum("ij,ij->i", delta, delta)
    near = sq <= KICKABLE_AREA * KICKABLE_AREA
    if not near.any():
        return None

    # T is how long the pass still has to run
    to_target = ball["flight_target"] - ball["position"]
    T = np.sqrt(to_target @ to_target) / BALL_SPEED

    react_exp = players["i_p"][dmask][near]
    p = intercept_probability_vec(T, react_exp)

    best = np.argmax(p)
    if p[best] > INTERCEPT_P_MIN:
        return int(players["id"][dmask][near][best])
    return None

def check_offside(players, target_id):
    # RoboCup marks offside on the kick event, not continuously,
    # so in render.py this check happens right after ball_action
    dmask = players["team"] == "defender"
    line = players["position"][dmask][:, 0].max()

    target_x = players["position"][players["id"] == target_id][0][0]
    return bool(target_x > line)   # strict, as rcssserver

def nearest_defender_to(players, position):
    dmask = players["team"] == "defender"
    delta = players["position"][dmask] - position
    return int(players["id"][dmask][np.argmin(np.einsum("ij,ij->i", delta, delta))])

def apply_turnover(ball, defender_id):
    ball = dict(ball)
    ball["state"] = "held"
    ball["holder_id"] = defender_id
    ball["target_id"] = None
    ball["position"] = np.asarray(ball["position"], dtype="f4")
    return ball
