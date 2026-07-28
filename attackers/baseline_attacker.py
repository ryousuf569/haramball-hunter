"""
baseline_attacker.py -- Throwaway test-harness policy for the attackers.

This exists ONLY to stress-test the scripted low-block defender in defenders.py.
It is NOT the RL baseline and NOT the learned attacker policy. There is no
learning here and no real tactical intelligence -- just a few hand-tuned rules
so the defender faces something less trivial than a metronome. Attackers hold a
2-5-3 formation (with per-player jitter + a mild ball-lean so they don't move
uniformly), and on a fixed cadence the holder passes with a forward bias, drops
a back pass when the front is marked, and mixes in occasional riskier / long
balls. Do not read deep tactical intent into any of this -- if you need a real
attacker, replace this file wholesale.

Matches defenders.py's calling convention: the main entry point returns a
continuous (n_att, 2) target-velocity array directly (bypassing engine.py's
action_decoding/direction_lookup/speed_lookup), plus the ball_idx array that
engine.ball_action expects. No termination, turnover, or scoring logic lives
here -- movement and pass decisions only.
"""

import numpy as np

V_MAX = 5.0
HOLD = 0

# The goal the attackers are moving toward is at x = 105 (see render.py). "Forward"
# means +x; this is used to bias passes and movement goalward.
GOAL_X = 105.0

# How often the current holder makes a pass, in ticks. Kept large enough that
# passes aren't happening every frame (the ball would never settle otherwise).
PASS_INTERVAL = 40

# --- pass-scoring weights (all deliberately hand-tuned constants, not learned) ---
# Openness (distance to that teammate's nearest defender) is the base of the score.
# Small weight on "don't pass to someone miles away" so passes stay plausible.
PASS_DISTANCE_PENALTY = 0.15
# Reward for moving the ball forward (+x). Positive dx (goalward) is rewarded;
# a back pass gets a negative contribution here, so it only wins when forward
# options are all tightly marked -- i.e. back passes happen when necessary.
PASS_FORWARD_WEIGHT = 0.35

# --- risk / variety (deterministic, tick-seeded -- no RNG state to thread) ---
# On some passing ticks the holder skips the safest choice and takes a riskier
# one: either the 2nd-best-scoring teammate, or an occasional long ball to the
# most advanced open teammate regardless of distance.
RISKY_PASS_EVERY = 3      # every 3rd pass, take the 2nd-best option instead of the best
LONG_BALL_EVERY = 5       # every 5th pass, launch a long ball to the most advanced teammate

# --- per-attacker movement individuality ---
# A fixed per-row jitter (m) added to each player's formation slot so they don't
# all sit and steer identically. Deterministic, indexed by attacker row.
PLAYER_JITTER = np.array([
    [ 1.5,  2.0], [-2.0,  1.0], [ 2.5, -1.5], [-1.0,  2.5], [ 1.0, -2.0],
    [-2.5, -1.0], [ 2.0,  1.5], [-1.5, -2.5], [ 1.5,  1.0], [-1.0, -1.5],
])
# How strongly each attacker leans toward the ball (small -- keeps the shape but
# lets players drift a bit toward play, so movement isn't uniform).
BALL_LEAN = 0.12

# Formation lines, laid out the same way as defenders.py's offset arrays: a
# fixed (n, 2) offset per line, symmetric around 0 in y, added to a moving
# reference point. Rows are grouped by line the same way defenders.py slices
# its backline/midfield/forward blocks.
#   backline: 2, midfield: 5, forward: 3  ->  10 attackers total.
N_ATT = 10
BACKLINE_INDICES = slice(0, 2)
MIDFIELD_INDICES = slice(2, 6)
FORWARD_INDICES = slice(6, 10)

# y spans the full 68m width the same way defenders.py's arrays do; x offsets
# stagger the lines back-to-front around the shared reference point.
backline_offset = np.array([[-18, -12], [-18, 12]])
midline_offset = np.array([[0, -24], [0, -12], [0, 0], [0, 12]])
forward_offset = np.array([[20, -20], [25, -8], [25, 8], [20, 20]])

# Shared formation reference point (a plain fixed centroid around the
# mid-to-attacking-third). No ball tracking / basculation lag -- this is
# deliberately much simpler than the defender's depth reference.
FORMATION_REF = np.array([52.0, 34.0])


def formation_targets(ref, ball_pos):
    """Target position for each attacker: shared reference point + line offset,
    then a per-player jitter and a mild individual lean toward the ball.

    Same offset-array pattern as compute_defender_targets: build a (n_att, 2)
    array and fill each line slice with ref + that line's offset block. The
    jitter/ball-lean are what break the previously-uniform movement -- each row
    gets a slightly different slot, so no two attackers steer identically.
    """
    targets = np.zeros((N_ATT, 2))
    targets[BACKLINE_INDICES] = ref + backline_offset
    targets[MIDFIELD_INDICES] = ref + midline_offset
    targets[FORWARD_INDICES] = ref + forward_offset

    # Individuality: fixed per-player jitter + a small pull toward the ball.
    targets = targets + PLAYER_JITTER + BALL_LEAN * (ball_pos - targets)
    return targets


def steer_to_targets(positions, targets):
    """Convert target positions into steering velocities, exactly like
    defenders.py Step 9: direction-to-target, normalized, scaled to V_MAX, with
    the same near-zero-distance guard against divide-by-zero."""
    to_target = targets - positions
    dist = np.linalg.norm(to_target, axis=1, keepdims=True)
    unit = np.divide(to_target, dist, out=np.zeros_like(to_target), where=dist > 1e-6)
    return unit * V_MAX


def choose_pass_target(holder_id, attacker_ids, att_positions, def_positions, pass_count):
    """Pick a teammate to pass to and encode the choice for ball_action.

    Scoring (still deliberately simple -- not a model): for each other attacker,
        score = openness + forward_progress - distance_penalty
    where
        openness          = distance to that teammate's nearest defender,
        forward_progress  = PASS_FORWARD_WEIGHT * (their x - holder x)  [+x is goalward],
        distance_penalty  = PASS_DISTANCE_PENALTY * distance from the holder.

    Forward progress biases the ball goalward, but a well-marked front line lets
    a safe back pass win instead (its openness beats the marked forwards) -- so
    back passes happen when necessary rather than never.

    Variety (deterministic on `pass_count`, so no RNG state to thread):
      * every LONG_BALL_EVERY-th pass -> long ball to the most ADVANCED teammate
        (largest x) that is at least somewhat open, ignoring distance,
      * every RISKY_PASS_EVERY-th pass -> take the 2nd-best score instead of the
        best (a tighter, riskier ball rather than the safest option).

    Returns a holder_choice in [1, n_teammates] indexing the SORTED teammate-id
    array, matching the encoding engine.ball_action decodes (see render.py's
    choose_actions()).
    """
    teammate_ids = np.sort(attacker_ids[attacker_ids != holder_id])
    holder_pos = att_positions[attacker_ids == holder_id][0]

    scores = np.empty(len(teammate_ids))
    tpositions = np.empty((len(teammate_ids), 2))
    for i, tid in enumerate(teammate_ids):
        tpos = att_positions[attacker_ids == tid][0]
        tpositions[i] = tpos
        openness = np.min(np.linalg.norm(def_positions - tpos, axis=1))
        forward_progress = PASS_FORWARD_WEIGHT * (tpos[0] - holder_pos[0])
        from_holder = np.linalg.norm(tpos - holder_pos)
        scores[i] = openness + forward_progress - PASS_DISTANCE_PENALTY * from_holder

    # Occasional long ball: throw it to the most advanced teammate that is at
    # least reasonably open, distance be damned.
    if pass_count % LONG_BALL_EVERY == 0:
        reasonably_open = openness_mask(tpositions, def_positions)
        if np.any(reasonably_open):
            advanced = np.where(reasonably_open, tpositions[:, 0], -np.inf)
            i = int(np.argmax(advanced))
            return i + 1

    order = np.argsort(scores)[::-1]  # best score first
    # Occasional riskier pass: take the runner-up instead of the safest option.
    if pass_count % RISKY_PASS_EVERY == 0 and len(order) > 1:
        i = int(order[1])
    else:
        i = int(order[0])
    return i + 1  # +1: choice 0 is HOLD; teammates are 1-indexed


def openness_mask(tpositions, def_positions, min_open=6.0):
    """Boolean mask of teammates who are at least `min_open` metres from the
    nearest defender -- "reasonably open" for the long-ball target filter."""
    nearest = np.array([
        np.min(np.linalg.norm(def_positions - tpos, axis=1)) for tpos in tpositions
    ])
    return nearest >= min_open


def compute_attacker_targets(players, ball, tick_count):
    """Movement + pass decisions for the attacker half of the roster.

    Parameters mirror compute_defender_targets: the full roster `players`
    structured array, the engine `ball` dict, plus a `tick_count` so passing
    can fire on a fixed cadence.

    Returns
    -------
    target_velocities : (n_att, 2) float array
        Continuous steering velocities in attacker-row order, ready to splice
        into the roster-wide target array (bypassing action_decoding).
    ball_idx : (n_att,) int array
        Per-attacker ball decision for engine.ball_action: HOLD (0) for every
        non-holder every tick; the holder's slot carries the pass choice on
        passing ticks (indexing sorted teammate ids), HOLD otherwise.
    """
    att_mask = players["team"] == "attacker"
    def_mask = players["team"] == "defender"

    attacker_ids = players["id"][att_mask]
    att_positions = players["position"][att_mask]
    def_positions = players["position"][def_mask]

    # Movement: everyone -- including the ball-holder -- holds their formation
    # slot, but each slot now carries a per-player jitter and a mild ball-lean
    # (see formation_targets), so movement is no longer uniform across the line.
    # No special dribbling/carrying behavior for the holder.
    ball_pos = np.asarray(ball["position"], dtype=float)
    targets = formation_targets(FORMATION_REF, ball_pos)
    target_velocities = steer_to_targets(att_positions, targets)

    # Ball decision: default HOLD for every attacker every tick.
    ball_idx = np.zeros(len(attacker_ids), dtype=int)

    # Every PASS_INTERVAL ticks, if an attacker is holding, choose a pass. The
    # chooser biases forward, allows back passes when the front is marked, and
    # mixes in riskier / long-ball variety keyed off the pass count.
    holder_id = ball.get("holder_id")
    holder_is_attacker = holder_id is not None and np.any(attacker_ids == holder_id)
    if (ball["state"] == "held" and holder_is_attacker
            and tick_count > 0 and tick_count % PASS_INTERVAL == 0
            and len(attacker_ids) > 1):
        pass_count = tick_count // PASS_INTERVAL
        choice = choose_pass_target(
            holder_id, attacker_ids, att_positions, def_positions, pass_count)
        ball_idx[attacker_ids == holder_id] = choice

    return target_velocities, ball_idx
