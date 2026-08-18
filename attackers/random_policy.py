import numpy as np

from physics.engine import direction_lookup, speed_lookup

N_DIRECTIONS = len(direction_lookup)
N_SPEEDS = len(speed_lookup)


def random_actions(n_att, rng, pass_prob=None):
    """Uniform draw over the three action heads -- the floor policy for the probe.

    Returns (direction_idx, speed_idx, ball_idx), the same tuple the scripted
    policy returns and the shape environment/lowblock_env.step consumes. Only
    the carrier's ball head is read; ball_action masks the rest.
    """
    direction_idx = rng.integers(0, N_DIRECTIONS, size=n_att)
    speed_idx = rng.integers(0, N_SPEEDS, size=n_att)
    if pass_prob is None:
        # Uniform over the whole ball head. This is the honest control for the
        # probe, and it releases on ~(n_att-1)/n_att of carrying ticks.
        ball_idx = rng.integers(0, n_att, size=n_att)
    else:
        ball_idx = np.where(rng.random(n_att) < pass_prob,
                            rng.integers(1, n_att, size=n_att), 0)
    return direction_idx, speed_idx, ball_idx
