"""The behaviours we want, written as constraints instead of reward terms.

Each cost is 0 or 1, so its average is a rate and its threshold is readable.
Read README.md, "Behaviour constraints", before changing anything here.
"""

import numpy as np

ATTACKER_LABEL = "attacker"
DEFENDER_LABEL = "defender"

HALFWAY_X = 52.5

# Thresholds. Every one is measured, not guessed, and none of them starts out
# already satisfied. Trailing comments are (5M checkpoint, random control).

COST_SPECS = (
    # (name, threshold d_k, what it is a rate over)
    ("pass_lost",   0.08, "passes"),        # 26.1% now, 9.9% random
    ("cross_field", 0.10, "passes"),        # 46.6% now, 38.0% random
    ("hot_potato",  0.25, "passes"),        # 97.7% now, 100% random
    ("pass_back",   0.25, "passes"),        # 37.5% now, 42.7% random
    ("offside",     0.02, "attacker-ticks"),  # 6.4% now, 1.9% random
    ("far_from_ball", 0.20, "attacker-ticks"),  # 29.4% now, 26.7% random
    # Deliberately ambitious, because its job is to stay violated and keep
    # propping up the reward's weight. At 0.85 random play nearly met it.
    ("no_success",  0.60, "episodes"),
)

COST_NAMES = tuple(name for name, _, _ in COST_SPECS)
COST_THRESHOLDS = np.array([d for _, d, _ in COST_SPECS], dtype="f4")
COST_UNITS = tuple(unit for _, _, unit in COST_SPECS)
N_COSTS = len(COST_SPECS)

IDX = {name: i for i, name in enumerate(COST_NAMES)}

# Last by convention. Its multiplier is a floor on the reward's own weight,
# which stops the policy meeting the other six by simply never passing.
BOOTSTRAP_IDX = IDX["no_success"]

# A quarter of the pitch width, above random play's 18m mean, so this catches
# the tail rather than every switch of play.
CROSS_FIELD_DY = 20.0

# The placement noise floor, so a square ball is not billed as a backward one.
BACKWARD_DX = -2.0

# Half a second. Long enough to take a touch and look up, short enough that a
# real one-touch lay-off is not caught by it.
HOT_POTATO_TICKS = 5

# The shape constraint. It stops ten attackers sprinting into the final third
# and leaving every pass a 30m diagonal.
FAR_FROM_BALL_M = 30.0


def empty_costs(n_att):
    """(cost, attempt) both (n_att, N_COSTS) float32, all zero."""
    return (np.zeros((n_att, N_COSTS), dtype="f4"),
            np.zeros((n_att, N_COSTS), dtype="f4"))


def per_tick_costs(players, ball, n_att, line):
    """Costs every attacker is assessed on every tick, whatever it did.

    Attempted on every row, so both are a plain per-attacker-tick rate.
    """
    cost, attempt = empty_costs(n_att)

    apos = np.asarray(players["position"][:n_att], dtype=float)
    ball_x = float(np.asarray(ball["position"], dtype=float)[0])

    # The same conditions as the is_offside flag already in the observation,
    # so the policy can see the thing it is being constrained on.
    offside = ((apos[:, 0] > line) & (apos[:, 0] > HALFWAY_X)
               & (apos[:, 0] > ball_x))
    if (players["team"] == DEFENDER_LABEL).sum() < 2:
        offside = np.zeros(n_att, dtype=bool)

    bpos = np.asarray(ball["position"], dtype=float)
    far = np.linalg.norm(apos - bpos, axis=1) > FAR_FROM_BALL_M

    cost[:, IDX["offside"]] = offside
    cost[:, IDX["far_from_ball"]] = far
    attempt[:, IDX["offside"]] = 1.0
    attempt[:, IDX["far_from_ball"]] = 1.0
    return cost, attempt


def release_costs(cost, attempt, row, holder_pos, target_pos, held_ticks):
    """The costs decided the moment the ball is released.

    Writes into cost and attempt on `row`, the passer's row. The positions are
    from the release tick, and held_ticks is how long the passer had the ball.
    """
    dx = float(target_pos[0] - holder_pos[0])
    dy = float(abs(target_pos[1] - holder_pos[1]))

    cost[row, IDX["cross_field"]] = float(dy > CROSS_FIELD_DY)
    cost[row, IDX["pass_back"]] = float(dx < BACKWARD_DX)
    cost[row, IDX["hot_potato"]] = float(held_ticks < HOT_POTATO_TICKS)

    for name in ("cross_field", "pass_back", "hot_potato"):
        attempt[row, IDX[name]] = 1.0
    # pass_lost is attempted here and resolved whenever the pass dies.
    attempt[row, IDX["pass_lost"]] = 1.0


def pass_lost_cost(cost, row):
    """A pass that ended in a turnover, billed to whoever played it.

    No attempt here. release_costs already booked one, and a second would
    leave the rate's denominator too high by the number of losses.
    """
    cost[row, IDX["pass_lost"]] = 1.0


def terminal_costs(cost, attempt, outcome, success_label):
    """The bootstrap constraint, billed to every row on the terminal step.

    Every attacker shares the blame for a possession that ends without a shot,
    so this one is not pinned to a single row.
    """
    cost[:, BOOTSTRAP_IDX] = float(outcome != success_label)
    attempt[:, BOOTSTRAP_IDX] = 1.0


def rates(cost_sums, attempt_sums, thresholds=None):
    """Per-constraint rates from batch sums, (K,) float64.

    A constraint with no attempts reports its own threshold, so its multiplier
    holds still. Returning 0 there would slowly train the run to ignore it.
    """
    d = COST_THRESHOLDS if thresholds is None else thresholds
    c = np.asarray(cost_sums, dtype=float)
    a = np.asarray(attempt_sums, dtype=float)
    out = np.array(d, dtype=float)
    live = a > 0
    out[live] = c[live] / a[live]
    return out


def describe(rate, lam, thresholds=None):
    """One line per constraint, for the training log."""
    d = COST_THRESHOLDS if thresholds is None else thresholds
    return "  ".join(
        f"{name} {r:.3f}/{t:.2f} l{l:.3f}"
        for name, t, r, l in zip(COST_NAMES, d, rate, lam))
