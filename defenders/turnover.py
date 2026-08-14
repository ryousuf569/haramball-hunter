import numpy as np

from physics.tti import intercept_probability_vec

# Ground-duel gate, fitted to football in
# defenders/calibration/duel_calibration.py against the ground duels Metrica
# Sample_Game_1 labels in its event file (CHALLENGE with a GROUND-* or TACKLE-*
# subtype, plus THEFT), measured over the carry time its tracking says those
# duels happened during. README section 7 has the whole study.
#
# Only DUEL_EXP is still RoboCup's. Its tackle box (tackle_dist=2.0,
# tackle_width=1.25, tackle_back_dist=0) is strongly forward-biased, and that
# does not survive contact with the data: at a real dispossession the p90 across
# the defender's heading is 2.08m against 2.05m along it, so a duel is as wide as
# it is deep. Fitting the axes per p90 and scaling all three together until the
# gate encloses 90% of real dispossessions gives the three below. There IS a
# front/back asymmetry -- p90 1.42m behind against 2.05m in front -- but it is
# 1.45x, not RoboCup's 3x.
#
# This matters more than the rate did. ground_duel is the model's only channel
# for taking the ball off a carrier, so a gate that cannot see a dispossession
# cannot produce it at any LAM_MAX. RoboCup's ellipse covered 14 of the 53 real
# ground-duel losses; rescaling it to the right reach but keeping its ratios only
# reached 28; these axes cover 48.
DUEL_A       = 2.38   # semi-axis along defender heading (m)
DUEL_B       = 2.38   # semi-axis across heading (m). Equal to A: the fit put them
                      # 1.5% apart, which 53 losses cannot resolve into a shape
DUEL_A_BACK  = 1.64   # reduced reach behind the defender (RoboCup uses 0)
DUEL_EXP     = 6      # RoboCup tackle_exponent. soft-edged box, hard cutoff.
                      # Fitted, and NOT IDENTIFIED: scanned 1-10 jointly with the
                      # gate scale, the 95% likelihood-ratio interval is the whole
                      # range and 6 sits 0.29 off the peak. It also barely matters
                      # -- refitting the scale with it, P(lost within 1s) at 0.5m
                      # only moves 0.186 to 0.151 across exponents 2 to 10. Kept
                      # at 6 on that evidence, not by inheritance
R_MAX        = 2.38   # bounding-circle prefilter = max(A, B, A_BACK)
# Fitted: the occupancy MLE, (ground-duel losses inside the gate) / (gated
# exposure, summed over every defender in the gate, matching how the hazards are
# summed below). Metrica game 1 has 1,682s of carrying with an opponent tracked
# and 53 labelled ground-duel losses in it, 48 of which fall inside the gate
# above, over 203s of contact-summed exposure -- 0.236/s, bootstrap 95% CI
# 0.17-0.32 over 2000 whole-spell resamples. Was 2.1, which was a target rather
# than a measurement (-ln(1-0.65)/0.5s) and 8.8x too fast: it gave a carrier at
# 1.0m an 81% chance of losing the ball inside one second, where real carriers
# survive about half the ground duels they are put in at all. The estimate barely
# moves with the gate size -- a wider gate buys proportionally more exposure --
# so it is the gate's SHAPE, not this number, that sets how a 1v1 plays out.
LAM_MAX      = 0.57   # hazard rate (1/s) for a perfect defender
# ENGAGEMENT. A defender who is near the ball is not necessarily contesting it:
# Metrica logs 351 moments with an opponent inside this gate but only 103
# annotated challenges, so two thirds of proximity is somebody running past. The
# thing that separates the two is whether he is actually closing on the ball, and
# that was already in the model as this multiplier -- just far too weak. Measured
# by closing speed, the dispossession hazard runs ~0.15 of its peak below 1 m/s
# and rises steeply past 3 m/s, a spread of about 6x; the old linear 0.5-to-1.0
# ramp allowed only 2x, and the data rejects it (dlogL -4.7 against the fit).
#
# ADV_FLOOR is fitted: 0.14, 95% likelihood-ratio interval 0.06-0.30. ADV_EXP is
# NOT identified by the likelihood alone (2.0-14.6, running to the scan edge like
# DUEL_EXP did), so 3 is taken as the value inside that interval which best
# reproduces the model-free binned hazard-vs-closing curve -- two criteria, as
# with DUEL_A. It is a real shape, not a rescale: the relationship is convex, and
# a linear ramp cannot be flat-and-low up to 2 m/s and steep past 3.
#
# The tautology check that matters: at the instant a tackle lands the defender is
# arriving at the ball by construction. Measuring closing 0.32s BEFORE contact
# keeps the same ~0.20 floor-to-peak ratio, so this is intent, not outcome.
ADV_FLOOR    = 0.14   # lambda multiplier when not closing at all; 1.0 at V_MAX
ADV_EXP      = 3      # convexity of the ramp between them
V_MAX        = 5.0    # mirrors engine.V_MAX; kept local so defenders/ never imports engine
EPS          = 1e-6

# RoboCup has no interception model to copy.
#
# INTERCEPT_REACH is the perpendicular distance from the ball's swept segment
# inside which a defender is deemed able to get a foot to it. It started as
# RoboCup's kickable area (player_size + ball_size + kickable_margin = 0.3 +
# 0.085 + 0.7 = 1.085), which is the radius a stationary player can touch a ball
# from within one 100ms tick. That is the wrong quantity here: it is a reaching
# distance, not an intercepting one. A pro closing a passing lane covers 2-3m in
# the ~1s a short pass is in the air, so a gate at 1.085 only ever caught a
# defender the pass was played straight at. 2.5m is that closing distance, and it
# is the geometric gate only -- intercept_probability_vec still has to clear
# INTERCEPT_P_MIN on the TTI, so being in the corridor is necessary, not
# sufficient.
INTERCEPT_REACH = 2.5
INTERCEPT_P_MIN = 0.80   # control probability that counts as a won interception. CALIBRATED
LOB_DIST        = 25.0   # passes longer than this are lofted
# A lofted ball used to be uninterceptable outright, which made any pass over
# 25m free. It is not free in real play: it clears the defence on the way up and
# comes back down onto a contested landing spot. So a loft is only contestable
# once it is past LOFT_DESCENT of its flight, and with a reduced reach there.
LOFT_DESCENT    = 0.6
LOFT_REACH      = 0.6    # fraction of INTERCEPT_REACH kept on a lofted ball

# Mirrors engine.pass_speed, kept local for the same reason as V_MAX. These MUST
# match physics/engine.py: this is how long the defence thinks the ball will take,
# and ball_mechanics is how long it actually takes, so a mismatch is a defence
# solving the wrong pursuit problem. Fitted in
# physics/validation/pass_speed_calibration.py.
PASS_SPEED_A    = 4.5292
PASS_SPEED_B    = 0.3537
PASS_SPEED_MAX  = 14.93
BALL_SPEED      = PASS_SPEED_MAX   # scalar upper bound, for callers that need one


def pass_speed(length):
    """Ball speed in m/s for a pass of this length, in metres."""
    length = np.maximum(np.asarray(length, dtype=float), 1e-6)
    return np.minimum(PASS_SPEED_A * length ** PASS_SPEED_B, PASS_SPEED_MAX)

# Offside. Attackers attack x=105, so their own half is x < 52.5 and no pass
# received there can be offside. Kept local for the same reason as V_MAX.
HALFWAY_X       = 52.5

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
    near = sq <= R_MAX * R_MAX
    if not near.any():
        return None

    # EVERY defender in range contests, not just the closest. This used to take
    # argmin, which made being surrounded by three defenders exactly as safe as
    # being marked by one -- a carrier could walk through the block as long as he
    # kept beating whoever happened to be nearest. Competing risks: independent
    # hazards on the same carrier add, so the tick's total is the sum and the
    # winner is drawn in proportion to what each contributed. The 1v1 case is
    # unchanged, which is the case duel_calibration.py fitted.
    delta = delta[near]
    dist = np.sqrt(sq[near])
    to_ball = -delta
    v_def = players["velocity"][dmask][near]
    speed = np.sqrt(np.einsum("ij,ij->i", v_def, v_def))

    # RoboCup uses body frame; we have no facing, so direction of travel stands in.
    # Standing still: no heading to rotate into, so use plain radial distance.
    moving = speed > EPS
    safe = np.where(moving, speed, 1.0)
    ux, uy = v_def[:, 0] / safe, v_def[:, 1] / safe
    dx = np.where(moving, to_ball[:, 0] * ux + to_ball[:, 1] * uy, dist)
    dy = np.where(moving, to_ball[:, 0] * -uy + to_ball[:, 1] * ux, 0.0)

    # RoboCup's super-ellipse: exponent 6 holds p near 1, then cuts off at the edge
    reach = np.where(dx >= 0.0, DUEL_A, DUEL_A_BACK)
    fail_prob = (np.abs(dx) / reach) ** DUEL_EXP + (np.abs(dy) / DUEL_B) ** DUEL_EXP
    p_geometry = 1.0 - fail_prob
    live = p_geometry > 0.0
    if not live.any():
        return None

    # to_ball points defender -> ball, so a positive projection means the gap is closing
    v_holder = players["velocity"][holder_idx]
    closing = (np.einsum("ij,j->i", to_ball, -v_holder)
               + np.einsum("ij,ij->i", to_ball, v_def)) / np.maximum(dist, EPS)
    advantage = np.clip(closing / V_MAX, -1.0, 1.0)
    multiplier = ADV_FLOOR + (1.0 - ADV_FLOOR) * (0.5 * (advantage + 1.0)) ** ADV_EXP

    # RoboCup rolls once per command; we test every tick, so hazard-rate it for dt independence
    lam_each = LAM_MAX * np.where(live, p_geometry, 0.0) * multiplier
    lam = lam_each.sum()
    p_tick = 1.0 - np.exp(-lam * dt)

    if rng.random() < p_tick:
        # which of them actually got it. One contestant is the common case and
        # this reduces to picking him, so the 1v1 stream is a single extra draw.
        share = lam_each / lam
        cand = int(np.searchsorted(np.cumsum(share), rng.random()))
        cand = min(cand, share.size - 1)
        return int(players["id"][dmask][near][cand])
    return None

def intercept_pass(players, ball, rng, prev_pos, dt=0.1):
    if ball["state"] != "in_flight":
        return None

    lob = ball["flight_target"] - ball["flight_start"]
    pass_length = np.sqrt(lob @ lob)

    reach = INTERCEPT_REACH
    if pass_length > LOB_DIST:
        flown = ball["position"] - ball["flight_start"]
        if np.sqrt(flown @ flown) < LOFT_DESCENT * pass_length:
            return None
        reach = INTERCEPT_REACH * LOFT_REACH

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
    near = sq <= reach * reach
    if not near.any():
        return None

    # T is how long the pass still has to run
    to_target = ball["flight_target"] - ball["position"]
    T = np.sqrt(to_target @ to_target) / pass_speed(pass_length)

    react_exp = players["i_p"][dmask][near]
    p = intercept_probability_vec(T, react_exp)

    best = np.argmax(p)
    if p[best] > INTERCEPT_P_MIN:
        return int(players["id"][dmask][near][best])
    return None

def offside_line(players):
    """x of the last outfield defender, or the halfway line if there are too few.

    The keeper is normally the deepest, so the second largest x is the outfielder
    who sets the line. Exposed because the observation carries it as a feature
    and the ball mask tests against it; check_offside reads it from here so the
    two cannot drift apart.
    """
    def_x = players["position"][players["team"] == "defender"][:, 0]
    if def_x.size < 2:
        return float(HALFWAY_X)
    return float(np.partition(def_x, -2)[-2])


def check_offside(players, holder_id, target_id):
    # RoboCup marks offside on the kick event, not continuously, so the env runs
    # this right after ball_action and before ball_mechanics
    if target_id is None or holder_id is None or target_id == holder_id:
        return False

    target_x = players["position"][players["id"] == target_id][0][0]

    # attackers attack x=105, so "ahead" always means larger x
    if target_x <= HALFWAY_X:
        return False

    # The pass leaves the holder's feet, which is exactly where ball_mechanics
    # sets flight_start, so read it off the holder rather than ball['position']
    ball_x = players["position"][players["id"] == holder_id][0][0]
    if target_x <= ball_x:
        return False

    if (players["team"] == "defender").sum() < 2:
        return False

    return bool(target_x > offside_line(players))

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
