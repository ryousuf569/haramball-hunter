import numpy as np

from physics.engine import direction_lookup, speed_lookup
from defenders.turnover import check_offside

STOP_RADIUS = 1.5   # m; inside this an attacker holds its slot
PRESS_RADIUS = 4.0  # m; the carrier releases when a defender is this close
OPEN_MIN = 5.0      # m; clearance at which a teammate counts as open

_DIRS = np.asarray(direction_lookup[:-1], dtype=float)  # 8 headings; last is [0,0]
STAND = len(direction_lookup) - 1
FULL_SPEED = len(speed_lookup) - 1


def make_slots(players, n_att):
    # Each attacker's offset from the attacking centroid at kickoff. Targets are
    # zone centre + offset, so the unit advances while keeping its shape. Without
    # this every attacker converges on the disc and owns it for free, which would
    # make the scripted ceiling meaningless.
    pos = np.asarray(players["position"][:n_att], dtype=float)
    return pos - pos.mean(axis=0)


def _direction_idx(vectors):
    d = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = np.divide(vectors, d, out=np.zeros_like(vectors), where=d > 1e-6)
    idx = np.argmax(unit @ _DIRS.T, axis=1)
    return np.where(d[:, 0] > STOP_RADIUS, idx, STAND)


def _clearance(players, positions):
    # metres from each position to the nearest defender
    dpos = np.asarray(players["position"][players["team"] == "defender"],
                      dtype=float)
    delta = dpos[None, :, :] - positions[:, None, :]
    return np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).min(axis=1)


def _pass_index(attacker_ids, holder_id, target_id):
    # ball_action reads the holder's entry as 0 = hold, else an index into the
    # sorted teammate ids
    teammates = np.sort(attacker_ids[attacker_ids != holder_id])
    return int(np.flatnonzero(teammates == target_id)[0]) + 1


def scripted_actions(players, ball, attacker_ids, slots, zone, hold_shape=True):
    n_att = len(attacker_ids)
    pos = np.asarray(players["position"][:n_att], dtype=float)
    centre = np.asarray(zone.centre, dtype=float)

    # hold_shape=False is the degenerate control: everyone runs at the disc.
    targets = centre + slots if hold_shape else np.tile(centre, (n_att, 1))
    direction_idx = _direction_idx(targets - pos)
    speed_idx = np.full(n_att, FULL_SPEED, dtype=int)
    ball_idx = np.zeros(n_att, dtype=int)

    holder_id = ball.get("holder_id")
    if (ball.get("state") != "held" or holder_id is None
            or not np.any(attacker_ids == holder_id)):
        return direction_idx, speed_idx, ball_idx

    row = int(np.flatnonzero(attacker_ids == holder_id)[0])
    # The carrier runs at the disc rather than at its slot. step() caps it to
    # DRIBBLE_V_MAX, so this is a dribble.
    direction_idx[row] = _direction_idx((centre - pos[row])[None, :])[0]

    clearance = _clearance(players, pos)
    if clearance[row] >= PRESS_RADIUS:
        return direction_idx, speed_idx, ball_idx

    # Under pressure: release to the most open teammate that is closer to the
    # disc than the carrier and would not be offside.
    goal_d = np.linalg.norm(pos - centre, axis=1)
    best, best_clear = None, OPEN_MIN
    for j in range(n_att):
        if j == row or goal_d[j] >= goal_d[row] or clearance[j] < best_clear:
            continue
        if check_offside(players, holder_id, int(attacker_ids[j])):
            continue
        best, best_clear = j, clearance[j]
    if best is not None:
        ball_idx[row] = _pass_index(attacker_ids, holder_id,
                                    int(attacker_ids[best]))
    return direction_idx, speed_idx, ball_idx


def make_policy(zone, hold_shape=True):
    # One instance per episode: slots are captured from the first tick, which is
    # the formation draw.
    state = {}

    def policy(players, ball, attacker_ids):
        if "slots" not in state:
            state["slots"] = make_slots(players, len(attacker_ids))
        return scripted_actions(players, ball, attacker_ids, state["slots"],
                                zone, hold_shape)

    return policy
