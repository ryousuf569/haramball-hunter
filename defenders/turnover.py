import numpy as np

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

def apply_turnover(ball, defender_id):
    """Hand the ball to `defender_id`, clearing any in-flight pass."""
    ball = dict(ball)
    ball["state"] = "held"
    ball["holder_id"] = defender_id
    ball["target_id"] = None
    ball["position"] = np.asarray(ball["position"], dtype="f4")
    return ball
