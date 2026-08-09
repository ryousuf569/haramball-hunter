# Tests for the parts of the ball model that used to make a long pass free:
# reception, placement error, lofted-ball interception, and the pitch boundary.
#
# The exploit these close: interception was switched off outright above
# LOB_DIST, ground_duel only fires on a held ball, and ball_mechanics handed the
# ball to the intended target however far it had run. A policy that released
# every ball on the first tick of possession, always over 25m, was therefore
# untouchable by any modelled turnover -- which is what the 5M checkpoint
# learned. Run: python tests/test_pass_reception.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.lowblock_env import (  # noqa: E402
    FAILURE,
    LowBlockEnv,
    make_initial_world,
    make_ppcf_grid,
)
from environment.lowblock_env import step as world_step  # noqa: E402
from defenders.turnover import (  # noqa: E402
    INTERCEPT_REACH,
    LOB_DIST,
    LOFT_DESCENT,
    intercept_pass,
)
from physics.engine import (  # noqa: E402
    DRIBBLE_V_MAX,
    DT,
    PASS_SIGMA,
    RECEPTION_RADIUS,
    V_MAX,
    ball_mechanics,
    cap_speed,
    kinematics_integrator,
    pass_scatter,
    receiver_at,
)
from schema import player_dt  # noqa: E402


def _players(spec):
    """spec: list of (id, team, position). Velocities zero, i_p unset."""
    players = np.zeros(len(spec), dtype=player_dt)
    for i, (pid, team, pos) in enumerate(spec):
        players["id"][i] = pid
        players["team"][i] = team
        players["position"][i] = pos
    return players


def _held(holder_id, pos):
    return {"state": "held", "holder_id": holder_id, "position": np.asarray(pos, dtype="f4"),
            "target_id": None, "flight_start": np.zeros(2, dtype="f4"),
            "flight_target": np.zeros(2, dtype="f4")}


def _fly_to_arrival(ball, players):
    for _ in range(1000):
        if ball["state"] != "in_flight":
            return ball
        ball = ball_mechanics(ball, players, (None, False, None))
    raise AssertionError("pass never arrived")


def test_receiver_must_be_near_where_the_ball_lands():
    # The intended target collects it if it is still within RECEPTION_RADIUS,
    # and does not if it has run off. Before this the pass was a teleport.
    landing = np.array([50.0, 34.0])
    near = _players([(1, "attacker", [10.0, 34.0]),
                     (2, "attacker", landing + [RECEPTION_RADIUS - 0.5, 0.0]),
                     (3, "attacker", [80.0, 34.0])])
    assert receiver_at(near, landing, 2) == 2

    gone = near.copy()
    gone["position"][1] = landing + [RECEPTION_RADIUS + 5.0, 0.0]
    gone["position"][2] = landing + [1.0, 0.0]
    assert receiver_at(gone, landing, 2) == 3, (
        "a target that ran off the ball still collected it")


def test_a_defender_can_collect_a_pass_that_ran_away_from_its_target():
    landing = np.array([50.0, 34.0])
    players = _players([(1, "attacker", [10.0, 34.0]),
                        (2, "attacker", landing + [12.0, 0.0]),
                        (3, "defender", landing + [1.5, 0.0])])
    assert receiver_at(players, landing, 2) == 3


def test_pass_completes_when_the_target_holds_its_ground():
    # The straightforward case still works: nobody moves, the ball arrives, the
    # intended target has it. Guards against over-tightening RECEPTION_RADIUS.
    players = _players([(1, "attacker", [10.0, 34.0]),
                        (2, "attacker", [40.0, 34.0])])
    ball = ball_mechanics(_held(1, [10.0, 34.0]), players, (1, True, 2))
    assert ball["state"] == "in_flight"
    ball = _fly_to_arrival(ball, players)
    assert ball["holder_id"] == 2


def test_placement_error_scales_with_pass_length():
    rng = np.random.default_rng(0)
    short = np.array([pass_scatter(5.0, rng) for _ in range(4000)])
    long_ = np.array([pass_scatter(40.0, rng) for _ in range(4000)])

    assert np.allclose(short.mean(axis=0), 0.0, atol=0.05), "scatter is biased"
    assert abs(short.std() - PASS_SIGMA * 5.0) < 0.05 * PASS_SIGMA * 5.0
    assert abs(long_.std() - PASS_SIGMA * 40.0) < 0.05 * PASS_SIGMA * 40.0
    assert long_.std() > short.std() * 5


def test_no_rng_means_no_scatter():
    # ball_mechanics keeps working without a generator, so the physics tests and
    # any caller that wants a deterministic ball are unaffected.
    assert np.array_equal(pass_scatter(40.0, None), np.zeros(2))


def _lofted(flown_fraction, defender_offset):
    """A 40m pass this far through its flight, with a defender beside it."""
    length = 40.0
    start = np.array([10.0, 34.0])
    target = start + [length, 0.0]
    pos = start + [length * flown_fraction, 0.0]

    players = _players([(1, "attacker", start), (2, "attacker", target),
                        (3, "defender", pos + defender_offset)])
    players["i_p"] = 0.1     # a defender that is going to win any TTI race
    ball = {"state": "in_flight", "holder_id": None, "position": pos,
            "target_id": 2, "flight_start": start, "flight_target": target}
    return players, ball


def test_a_lofted_pass_is_contestable_on_its_way_down():
    assert LOB_DIST < 40.0, "this test needs a pass long enough to be a loft"
    rng = np.random.default_rng(0)
    players, ball = _lofted(LOFT_DESCENT + 0.2, [0.0, 0.5])
    prev = ball["position"] - [1.0, 0.0]

    assert intercept_pass(players, ball, rng, prev, dt=DT) == 3, (
        "a defender standing on a descending loft did not win it")


def test_a_lofted_pass_is_safe_early_in_its_flight():
    rng = np.random.default_rng(0)
    players, ball = _lofted(LOFT_DESCENT - 0.2, [0.0, 0.5])
    prev = ball["position"] - [1.0, 0.0]

    assert intercept_pass(players, ball, rng, prev, dt=DT) is None, (
        "a loft was intercepted before it started coming down")


def test_a_loft_has_a_smaller_corridor_than_a_ground_pass():
    # Reach is cut on a loft, so a defender that would cut out a ground pass at
    # this distance does not reach one that is dropping out of the sky.
    rng = np.random.default_rng(0)
    offset = [0.0, INTERCEPT_REACH - 0.2]
    players, ball = _lofted(LOFT_DESCENT + 0.2, offset)
    prev = ball["position"] - [1.0, 0.0]
    assert intercept_pass(players, ball, rng, prev, dt=DT) is None

    players, ball = _lofted(LOFT_DESCENT + 0.2, [0.0, 0.5])
    prev = ball["position"] - [1.0, 0.0]
    assert intercept_pass(players, ball, rng, prev, dt=DT) == 3


def test_running_into_a_line_kills_that_velocity_component():
    # Clipping position alone left a player pinned at x=105 reporting 5 m/s it
    # was not travelling, and made the boundary a free place to park.
    players = _players([(1, "attacker", [104.0, 34.0]),
                        (2, "attacker", [50.0, 67.5]),
                        (3, "attacker", [50.0, 34.0])])
    targets = np.array([[V_MAX, 0.0], [0.0, V_MAX], [V_MAX, 0.0]], dtype="f4")

    out = players
    for _ in range(20):
        out = kinematics_integrator(out, targets)

    assert out["position"][0][0] == 105.0 and out["velocity"][0][0] == 0.0, (
        f"x line: pos {out['position'][0]} vel {out['velocity'][0]}")
    assert out["position"][1][1] == 68.0 and out["velocity"][1][1] == 0.0, (
        f"y line: pos {out['position'][1]} vel {out['velocity'][1]}")
    assert np.allclose(out["velocity"][2], [V_MAX, 0.0], atol=1e-3), (
        "a player nowhere near a line lost its velocity")


def test_a_pass_collected_by_a_defender_ends_the_episode():
    # step() resolves turnovers through ground_duel and intercept_pass, neither
    # of which sees a pass that simply landed nearer a defender. Without the
    # check in step() the episode would carry on with the defence in possession.
    players, ball, attacker_ids, _rng, defender_state = make_initial_world(seed=3)
    grid = make_ppcf_grid()

    # Aim a long ball at a teammate, then have every attacker sprint away from
    # where it is going to land.
    holder_row = int(np.flatnonzero(attacker_ids == ball["holder_id"])[0])
    ball_idx = np.zeros(len(attacker_ids), dtype=int)
    ball_idx[holder_row] = 9
    vel = np.zeros((len(attacker_ids), 2), dtype="f4")
    vel[:, 0] = -V_MAX

    outcome = None
    for tick in range(80):
        players, ball, _pc, outcome = world_step(
            players, ball, attacker_ids, defender_state, tick, ppcf_grid=grid,
            attacker_velocities=vel, attacker_ball_idx=ball_idx, verbose=False)
        ball_idx = np.zeros(len(attacker_ids), dtype=int)
        if outcome is not None:
            break

    assert outcome == FAILURE, f"attackers abandoned the ball and kept it ({outcome})"
    holder = ball["holder_id"]
    assert players["team"][players["id"] == holder][0] == "defender"


def test_cap_speed_only_touches_what_is_over_the_cap():
    slow = np.array([1.0, 0.0], dtype="f4")
    assert np.allclose(cap_speed(slow, DRIBBLE_V_MAX), slow)

    fast = np.array([0.0, V_MAX], dtype="f4")
    out = cap_speed(fast, DRIBBLE_V_MAX)
    assert abs(np.linalg.norm(out) - DRIBBLE_V_MAX) < 1e-5
    assert out[1] > 0, "direction was not preserved"


def test_the_carrier_is_slower_than_a_free_runner():
    # Equal top speeds meant a chasing defender could never close on a carrier,
    # so walking the ball in from midfield was uncontested.
    assert DRIBBLE_V_MAX < V_MAX
    # Not seed 5: that one starts with an attacker already inside the shot gate
    # and ends in success on tick 0, so there is no carry to measure.
    env = LowBlockEnv(max_ticks=120, scripted_attackers=False)
    env.reset(seed=3)

    # Short enough that nobody has reached the goal line -- a player held
    # against it has had that velocity component zeroed and is not a fair
    # comparison for "did anyone reach full speed".
    east_full = np.tile([0, 2, 0], (env.n_att, 1)).astype(np.int64)
    for _ in range(15):
        _obs, _r, term, _trunc, _i = env.step(east_full)
        if term:
            break

    row = env.holder_row()
    assert row is not None, "seed 5 should still have an attacker on the ball"
    pos = np.asarray(env.players["position"][:env.n_att], dtype=float)
    speeds = np.linalg.norm(np.asarray(env.players["velocity"][:env.n_att],
                                       dtype=float), axis=1)

    assert speeds[row] <= DRIBBLE_V_MAX + 1e-3, (
        f"carrier is running at {speeds[row]:.2f} m/s")

    off_ball = (np.arange(env.n_att) != row) & (pos[:, 0] < 105.0 - 1e-6)
    assert off_ball.any(), "no off-ball attacker left to compare against"
    assert (speeds[off_ball] > DRIBBLE_V_MAX).any(), (
        f"nobody off the ball beat the dribble cap: {speeds[off_ball]}")


def test_the_cap_lifts_when_the_ball_leaves_his_feet():
    from physics.engine import DRIBBLE_SPEED
    assert 0.0 < DRIBBLE_SPEED < 1.0
    env = LowBlockEnv(max_ticks=120, scripted_attackers=False)
    env.reset(seed=3)
    row = env.holder_row()

    act = np.tile([0, 2, 0], (env.n_att, 1)).astype(np.int64)
    act[row, 2] = 4     # release it
    env.step(act)
    assert env.holder_row() != row, "the pass did not leave"

    for _ in range(20):
        _obs, _r, term, _trunc, _i = env.step(np.tile([0, 2, 0], (env.n_att, 1)).astype(np.int64))
        if term:
            return
    pos = np.asarray(env.players["position"][row], dtype=float)
    if pos[0] < 105.0 - 1e-6:
        speed = float(np.linalg.norm(env.players["velocity"][row]))
        assert speed > DRIBBLE_V_MAX, (
            f"the ex-carrier is still capped at {speed:.2f} m/s")


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
