# Tests for length-dependent pass speed.
#
# The constant this replaced (a flat 15 m/s) made every short pass land inside
# tti.reaction_time, which closed interception by construction for exactly the
# short recycling passes a learned attacker favours. The fit comes from
# physics/validation/pass_speed_calibration.py.
# Run: python tests/test_pass_speed.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import defenders.turnover as turnover  # noqa: E402
from physics.engine import (  # noqa: E402
    DT,
    PASS_SPEED_A,
    PASS_SPEED_B,
    PASS_SPEED_MAX,
    ball_mechanics,
    pass_speed,
)
from physics.tti import reaction_time  # noqa: E402
from schema import player_dt  # noqa: E402

# Binned medians measured off 793 Metrica passes, with the bin midpoint each was
# taken at. Tolerance is the fit's trend RMSE against these same medians (0.57
# m/s), rounded up -- this pins the calibration, not an exact functional form.
REAL_MEDIANS = [(2.5, 6.74), (6.5, 8.63), (10.0, 10.13),
                (15.0, 12.57), (21.5, 13.24), (32.5, 14.81)]
TREND_TOL = 0.8


def test_matches_the_real_speed_length_curve():
    for length, real in REAL_MEDIANS:
        got = float(pass_speed(length))
        assert abs(got - real) < TREND_TOL, (
            f"{length}m pass: fit says {got:.2f} m/s, Metrica median is {real:.2f}")


def test_speed_increases_with_length_and_caps():
    lengths = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    speeds = pass_speed(lengths)
    rising = speeds[:-1] <= speeds[1:] + 1e-9
    assert rising.all(), "longer passes are never struck slower"
    assert speeds[-1] == PASS_SPEED_MAX, "the fit is capped, not unbounded"
    assert float(pass_speed(1e9)) == PASS_SPEED_MAX


def test_short_passes_outlive_the_reaction_time():
    """The whole point of the change.

    At the old flat 15 m/s a 7m pass was airborne 0.47s, under the 0.54s
    reaction, so no defender could ever contest it however well placed.
    """
    for length in (5.0, 7.0, 10.0):
        flight = length / float(pass_speed(length))
        assert flight > reaction_time, (
            f"{length}m pass lands in {flight:.2f}s, inside the {reaction_time}s reaction")
        assert length / 15.0 < flight, "the fit must be slower than the old constant here"


def test_engine_and_turnover_agree():
    """A mismatch would leave the defence solving the wrong pursuit problem."""
    for c in ("PASS_SPEED_A", "PASS_SPEED_B", "PASS_SPEED_MAX"):
        assert getattr(turnover, c) == globals()[c], f"{c} has drifted between the two modules"
    lengths = np.array([1.0, 4.0, 9.0, 16.0, 30.0, 60.0])
    assert np.allclose(pass_speed(lengths), turnover.pass_speed(lengths))


def _two_players(start, target):
    players = np.zeros(2, dtype=player_dt)
    players["id"] = [1, 2]
    players["team"] = ["attacker", "attacker"]
    players["position"] = [start, target]
    return players


def _fly(length):
    """Release a pass of this length and count the ticks until it arrives."""
    start, target = np.array([10.0, 34.0]), np.array([10.0 + length, 34.0])
    players = _two_players(start, target)
    ball = {"state": "held", "holder_id": 1, "position": start.copy(),
            "target_id": None, "flight_start": np.zeros(2), "flight_target": np.zeros(2)}
    ball = ball_mechanics(ball, players, (1, True, 2))
    assert ball["state"] == "in_flight"

    ticks = 0
    while ball["state"] == "in_flight" and ticks < 1000:
        ball = ball_mechanics(ball, players, (None, False, None))
        ticks += 1
    return ticks * DT, ball


def test_ball_mechanics_flight_time_follows_the_fit():
    for length in (5.0, 10.0, 20.0, 40.0):
        flown, ball = _fly(length)
        expected = length / float(pass_speed(length))
        assert ball["holder_id"] == 2, "the pass must complete"
        # one tick of quantisation: the ball snaps on the tick it would overshoot
        assert abs(flown - expected) <= DT + 1e-9, (
            f"{length}m pass took {flown:.2f}s, fit says {expected:.2f}s")


def test_flight_speed_is_constant_across_the_pass():
    """Speed is read off the frozen start/target, not off the distance left, so
    the ball must not decelerate as it approaches the receiver."""
    start, target = np.array([10.0, 34.0]), np.array([40.0, 34.0])
    players = _two_players(start, target)
    ball = {"state": "held", "holder_id": 1, "position": start.copy(),
            "target_id": None, "flight_start": np.zeros(2), "flight_target": np.zeros(2)}
    ball = ball_mechanics(ball, players, (1, True, 2))

    steps = []
    while ball["state"] == "in_flight":
        prev = np.asarray(ball["position"], dtype=float).copy()
        ball = ball_mechanics(ball, players, (None, False, None))
        steps.append(float(np.linalg.norm(np.asarray(ball["position"], dtype=float) - prev)))

    # drop the final snap-to-target step, which is a partial tick by construction
    assert len(steps) > 3
    assert np.allclose(steps[:-1], steps[0]), "the ball changed speed mid-flight"
    assert abs(steps[0] / DT - float(pass_speed(30.0))) < 1e-6


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _main()
