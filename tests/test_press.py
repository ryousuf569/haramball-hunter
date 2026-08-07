# Tests for the scripted low block's press trigger.
#
# Two behaviours are pinned here, both of which the 500k baseline exposed:
#
#   1. A policy that releases the ball the tick after receiving it leaves the
#      ball in flight ~90% of ticks, and a press that only looks at
#      ball['holder_id'] is asleep for all of them. press_target keeps a
#      commitment alive across the flight.
#   2. Selecting the presser by body distance hands the press to whoever the last
#      press left nearest, so one defender chases a recycled ball across the
#      whole block. Selection by SLOT distance plus the PRESS_MAX_EXCURSION leash
#      makes it change hands instead, and PRESS_BAND_* keeps the block home when
#      the ball is not in a zone worth stepping out for.
#
# The band and leash numbers come from defenders/calibration/press_calibration.py.
# Run: python tests/test_press.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defenders.defenders import (  # noqa: E402
    GK_INDEX,
    PRESS_BAND_BEHIND,
    PRESS_BAND_FRONT,
    PRESS_LATCH_TICKS,
    PRESS_MAX_EXCURSION,
    V_MAX,
    apply_pressure_trigger,
    in_press_band,
    make_defender_state,
    press_target,
)
from schema import player_dt  # noqa: E402

N_ATT = 3
N_DEF = 11

# A resting 5-4-1 in the sim frame (block defends x=105), laid out in the row
# order compute_defender_targets uses: keeper, 5 back, 4 mid, 1 forward.
BACK_LINE_X = 88.0
SLOTS = np.array([
    [104.0, 34.0],                                                        # 0  GK
    [88.0, 14.0], [88.0, 24.0], [88.0, 34.0], [88.0, 44.0], [88.0, 54.0],  # 1-5 back
    [78.0, 19.0], [78.0, 29.0], [78.0, 39.0], [78.0, 49.0],                # 6-9 mid
    [52.0, 34.0],                                                          # 10 forward
], dtype="f4")

# Holder 8m in front of the back line (inside the band) with support 8m away. The
# nearest slot to it is row 7's, so row 7 presses unless a test moves the ball.
# Row 2 sits out in row 9's zone, close enough to row 1 to be its support when
# the ball goes there -- the trigger needs support at the DESTINATION, not at the
# holder, so a lone wide target would never be pressed at all.
DEFAULT_ATT = [[80.0, 30.0], [80.0, 38.0], [76.0, 50.0]]


def _world(holder_row=0, att_positions=None, def_positions=None):
    """Small roster with the same attackers-then-defenders layout as the env."""
    players = np.zeros(N_ATT + N_DEF, dtype=player_dt)
    players["id"] = np.arange(1, N_ATT + N_DEF + 1)
    players["team"][:N_ATT] = "attacker"
    players["team"][N_ATT:] = "defender"

    players["position"][:N_ATT] = DEFAULT_ATT if att_positions is None else att_positions
    # Block at rest on its slots unless a test moves someone off
    players["position"][N_ATT:] = SLOTS if def_positions is None else def_positions

    holder_id = int(players["id"][holder_row])
    ball = {
        "state": "held",
        "holder_id": holder_id,
        "position": players["position"][holder_row].copy(),
        "target_id": None,
        "flight_start": np.zeros(2, dtype="f4"),
        "flight_target": np.zeros(2, dtype="f4"),
    }
    return players, ball


def _in_flight(players, ball, target_row, ball_pos):
    """Put the ball mid-air toward target_row, as ball_mechanics would."""
    ball = dict(ball)
    ball["state"] = "in_flight"
    ball["flight_start"] = ball["position"].copy()
    ball["holder_id"] = None
    ball["target_id"] = int(players["id"][target_row])
    ball["flight_target"] = players["position"][target_row].copy()
    ball["position"] = np.asarray(ball_pos, dtype="f4")
    return ball


def _apply(players, ball, state, back_line_x=BACK_LINE_X):
    """Run the trigger on a zeroed target-velocity array; return defender rows."""
    def_positions = players["position"][players["team"] == "defender"]
    tv = np.zeros_like(def_positions)
    return apply_pressure_trigger(players, ball, def_positions, SLOTS, tv, V_MAX,
                                  state, back_line_x)


# --- press_target: what the press aims at ----------------------------------
def test_press_target_held():
    players, ball = _world()
    pos, key = press_target(players, ball)
    assert key == ball["holder_id"]
    assert np.allclose(pos, players["position"][0])


def test_press_target_in_flight_is_the_destination():
    """Mid-flight the press aims where the ball lands, not where it is now."""
    players, ball = _world()
    ball = _in_flight(players, ball, target_row=1, ball_pos=[80.0, 33.0])
    pos, key = press_target(players, ball)
    assert key == int(players["id"][1])
    assert np.allclose(pos, players["position"][1])
    assert not np.allclose(pos, ball["position"])


def test_press_target_none_when_defender_has_it():
    players, ball = _world()
    ball = dict(ball, holder_id=int(players["id"][N_ATT]))  # a defender
    pos, key = press_target(players, ball)
    assert pos is None and key is None


# --- the band: when the block steps out at all -----------------------------
def test_in_press_band_edges():
    assert in_press_band([BACK_LINE_X + PRESS_BAND_FRONT, 34.0], BACK_LINE_X)
    assert in_press_band([BACK_LINE_X + PRESS_BAND_BEHIND, 34.0], BACK_LINE_X)
    assert not in_press_band([BACK_LINE_X + PRESS_BAND_FRONT - 0.1, 34.0], BACK_LINE_X)
    assert not in_press_band([BACK_LINE_X + PRESS_BAND_BEHIND + 0.1, 34.0], BACK_LINE_X)


def test_no_press_when_the_ball_is_still_in_front_of_the_band():
    """Ball upfield of the midfield line: the block holds, nobody steps out."""
    x = BACK_LINE_X + PRESS_BAND_FRONT - 5.0
    players, ball = _world(att_positions=[[x, 30.0], [x, 36.0], [76.0, 50.0]])
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)
    assert state["press_latch"]["ticks_left"] == 0


def test_no_press_when_the_ball_is_already_through_the_block():
    """Past the back line it is the keeper's problem; chasing only strings the
    block out behind the ball."""
    x = BACK_LINE_X + PRESS_BAND_BEHIND + 5.0
    players, ball = _world(att_positions=[[x, 30.0], [x, 36.0], [76.0, 50.0]])
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)
    assert state["press_latch"]["ticks_left"] == 0


def test_leaving_the_band_releases_a_committed_press():
    players, ball = _world()
    state = make_defender_state(seed=0)
    _apply(players, ball, state)
    assert state["press_latch"]["ticks_left"] == PRESS_LATCH_TICKS

    # same holder, but they have carried it through the block
    through = BACK_LINE_X + PRESS_BAND_BEHIND + 5.0
    players["position"][0] = [through, 30.0]
    ball = dict(ball, position=players["position"][0].copy())
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)
    assert state["press_latch"]["ticks_left"] == 0


# --- selection: who presses ------------------------------------------------
def test_press_fires_on_a_held_ball():
    players, ball = _world()
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)

    presser = state["press_latch"]["presser"]
    assert presser == 7, "row 7's slot is the one the ball is in"
    to_holder = players["position"][0] - players["position"][N_ATT + presser]
    assert np.allclose(tv[presser] / V_MAX, to_holder / np.linalg.norm(to_holder),
                       atol=1e-6), "presser charges at full tilt"


def test_keeper_never_presses():
    players, ball = _world()
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)
    assert np.allclose(tv[GK_INDEX], 0.0)
    assert state["press_latch"]["presser"] != 0
    assert state["press_latch"]["coverer"] != 0


def test_press_is_selected_by_slot_not_by_body():
    """The regression that makes the press hand off instead of chase.

    Row 8 is parked right next to the ball -- as it would be having just chased
    the previous pass -- but the ball is in row 7's zone, so row 7 presses. Under
    body-distance selection row 8 would take this one too, and the one after.
    """
    pos = SLOTS.copy()
    pos[8] = [80.5, 30.5]
    players, ball = _world(def_positions=pos)
    state = make_defender_state(seed=0)
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 7


def test_press_hands_off_as_the_ball_moves_across_the_block():
    players, ball = _world()
    state = make_defender_state(seed=0)
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 7

    # pass out to the attacker sitting in row 9's zone
    ball = _in_flight(players, ball, target_row=2, ball_pos=[78.0, 40.0])
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 9, "row 9 owns that space now"
    assert state["press_latch"]["ticks_left"] == PRESS_LATCH_TICKS


def test_no_press_when_the_ball_is_in_nobody_s_zone():
    """In the band by x, but no slot within the leash: the block holds shape."""
    players, ball = _world(att_positions=[[80.0, 2.0], [80.0, 8.0], [76.0, 50.0]])
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)
    assert state["press_latch"]["ticks_left"] == 0


def test_no_press_without_support():
    """Isolated carrier: the block holds shape rather than stepping out."""
    players, ball = _world(att_positions=[[80.0, 30.0], [40.0, 10.0], [30.0, 60.0]])
    state = make_defender_state(seed=0)
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)
    assert state["press_latch"]["ticks_left"] == 0


def test_loose_ball_leaves_shape_untouched():
    players, ball = _world()
    state = make_defender_state(seed=0)
    ball = dict(ball, state="held", holder_id=None)
    tv = _apply(players, ball, state)
    assert np.allclose(tv, 0.0)


# --- the leash: how far a presser may be dragged ---------------------------
def test_excursion_leash_breaks_a_committed_press():
    """A carrier who runs across the block does not tow one defender with them.

    The latch would otherwise hold row 7 on the ball for its full 2.5s, which is
    exactly how a hole opens where row 7 is supposed to be.
    """
    players, ball = _world()
    state = make_defender_state(seed=0)
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 7

    # holder has carried it sideways and row 7 has chased past its leash
    players["position"][0] = [80.0, 48.0]
    players["position"][N_ATT + 7] = [79.0, 47.0]
    players["position"][1] = [80.0, 44.0]          # keep support in range
    dragged = np.linalg.norm(players["position"][N_ATT + 7] - SLOTS[7])
    assert dragged > PRESS_MAX_EXCURSION, "fixture must actually exceed the leash"

    ball = dict(ball, position=players["position"][0].copy())
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 9, "row 9 takes it on"


def test_press_within_the_leash_keeps_its_commitment():
    players, ball = _world()
    state = make_defender_state(seed=0)
    _apply(players, ball, state)
    assert state["press_latch"]["ticks_left"] == PRESS_LATCH_TICKS

    # row 7 has stepped out a little, still well inside the leash
    players["position"][N_ATT + 7] = [79.0, 29.5]
    _apply(players, ball, state)
    assert state["press_latch"]["presser"] == 7
    assert state["press_latch"]["ticks_left"] == PRESS_LATCH_TICKS - 1, "decremented, not re-armed"


# --- the latch: how long a commitment lasts --------------------------------
def test_press_stays_live_through_a_pass_flight():
    """Old behaviour: the trigger read ball['holder_id'], which is None for every
    tick of a flight, so it returned early and left every defender on its slot
    velocity for the whole ~1s the ball was in the air -- and the latch, keyed on
    that same holder_id, died on release too. The presser got one tick of
    velocity, 0.7 m/s at A_MAX, and was never seen to press.
    """
    players, ball = _world()
    state = make_defender_state(seed=0)
    _apply(players, ball, state)

    ball = _in_flight(players, ball, target_row=1, ball_pos=[80.0, 33.0])
    presser = None
    for tick in range(5):
        tv = _apply(players, ball, state)
        latch = state["press_latch"]
        assert latch["ticks_left"] > 0, f"latch died on flight tick {tick}"
        assert np.linalg.norm(tv[latch["presser"]]) > 0, f"press idle on tick {tick}"
        if presser is None:
            presser = latch["presser"]
        assert latch["presser"] == presser, f"presser changed on tick {tick}"


def test_latch_key_carries_from_flight_into_reception():
    """One commitment spans flight and reception: same key, no re-selection."""
    players, ball = _world()
    state = make_defender_state(seed=0)

    flight = _in_flight(players, ball, target_row=1, ball_pos=[80.0, 33.0])
    _apply(players, flight, state)
    key = state["press_latch"]["target_key"]
    presser = state["press_latch"]["presser"]
    ticks = state["press_latch"]["ticks_left"]
    assert key == int(players["id"][1])

    arrived = dict(ball, state="held", holder_id=int(players["id"][1]),
                   target_id=None, position=players["position"][1].copy())
    _apply(players, arrived, state)
    assert state["press_latch"]["target_key"] == key
    assert state["press_latch"]["presser"] == presser
    # decremented, not re-armed -- a reset would put it back at PRESS_LATCH_TICKS
    assert state["press_latch"]["ticks_left"] == ticks - 1


def test_coverer_holds_position_for_the_whole_commitment():
    """The coverer is latched too: without it the cover shape lasts one tick."""
    players, ball = _world()
    state = make_defender_state(seed=0)

    # both ticks are the same flight, so the latch holds rather than re-arming
    ball = _in_flight(players, ball, target_row=1, ball_pos=[80.0, 33.0])
    _apply(players, ball, state)
    coverer = state["press_latch"]["coverer"]
    assert coverer is not None

    ball = dict(ball, position=np.asarray([80.0, 34.5], dtype="f4"))
    tv = _apply(players, ball, state)
    assert state["press_latch"]["coverer"] == coverer
    # shifts at 0.6 * V_MAX, not a full-speed charge
    assert np.isclose(np.linalg.norm(tv[coverer]), 0.6 * V_MAX, atol=1e-6)


def _main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _main()
