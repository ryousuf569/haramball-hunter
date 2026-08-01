# Golden-value regression test for physics.ppcf.PPCF_grid.
# The fixture came from the pre-optimization loop (commit 0edf140), so any
# rewrite of the loop's mechanics must leave these numbers bit-identical.
# Run: python tests/test_ppcf_golden.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.ppcf import PPCF_grid  # noqa: E402
from schema import player_dt  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "golden_ppcf.npz")

# a loop-restructuring change should be bit-exact; TOL just makes failures readable
TOL = 1e-12

SCENARIOS = ["tick0", "tick0_no_ball", "tick40", "edge_cases"]


def _load(name):
    # rebuild one scenario's inputs and expected output from the fixture
    d = np.load(GOLDEN)
    positions = d[f"{name}/positions"]
    players = np.zeros(len(positions), dtype=player_dt)
    players["id"] = np.arange(1, len(positions) + 1)
    players["position"] = positions
    players["velocity"] = d[f"{name}/velocities"]
    players["team"] = d[f"{name}/team"]

    ball_pos = d[f"{name}/ball_pos"]
    ball_pos = None if ball_pos.size == 0 else ball_pos

    return players, d[f"{name}/targets"], ball_pos, d[f"{name}/result"], d[f"{name}/i_p"]


def _check_scenario(name):
    players, targets, ball_pos, expected, expected_i_p = _load(name)
    got = PPCF_grid(targets, players, ball_pos)

    assert got.shape == expected.shape, (
        f"{name}: shape {got.shape} != golden {expected.shape}")

    diff = np.abs(got - expected)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(expected), 1e-300)
    max_rel = float((diff / denom).max())
    assert max_abs <= TOL, (
        f"{name}: max abs diff {max_abs:.3e} exceeds {TOL:.0e} "
        f"(max rel {max_rel:.3e}) -- the loop change altered the physics")

    # i_p feeds intercept_pass, so a rewrite that skips it breaks interceptions silently
    if ball_pos is not None:
        assert np.array_equal(players["i_p"], expected_i_p), (
            f"{name}: players['i_p'] cache does not match golden")
    else:
        assert np.all(players["i_p"] == 0.0), (
            f"{name}: ball_pos=None must not write the i_p cache")

    return max_abs, bool(np.array_equal(got, expected))


def test_golden_tick0():
    _check_scenario("tick0")


def test_golden_tick0_no_ball():
    _check_scenario("tick0_no_ball")


def test_golden_tick40():
    _check_scenario("tick40")


def test_golden_edge_cases():
    _check_scenario("edge_cases")


def test_convergence_invariants():
    # a cell below 0.99 means the loop exited early -- what a bad cap or mask would do
    for name in SCENARIOS:
        players, targets, ball_pos, _, _ = _load(name)
        got = PPCF_grid(targets, players, ball_pos)
        totals = got.sum(axis=1)

        assert np.all(got >= 0.0), f"{name}: negative control"
        assert np.all(np.isfinite(got)), f"{name}: non-finite control"
        assert totals.min() >= 0.99, (
            f"{name}: {(totals < 0.99).sum()} cell(s) below the 0.99 "
            f"convergence threshold (min {totals.min():.6f}) -- loop exited early")
        assert totals.max() < 1.0, (
            f"{name}: cell total {totals.max():.6f} exceeds 1.0")


def test_active_mask_does_not_reorder_players():
    # column j must stay player j; the summed heatmap looks fine either way
    players, targets, ball_pos, expected, _ = _load("tick0")
    got = PPCF_grid(targets, players, ball_pos)
    is_att = players["team"] == "attacker"
    assert np.allclose(got[:, is_att], expected[:, is_att], rtol=0, atol=TOL)
    assert np.allclose(got[:, ~is_att], expected[:, ~is_att], rtol=0, atol=TOL)


def main():
    print(f"golden fixture: {GOLDEN}\n")
    failures = 0
    for name in SCENARIOS:
        try:
            max_abs, exact = _check_scenario(name)
            flag = "bit-exact" if exact else f"within {TOL:.0e}"
            print(f"  PASS  {name:14s} max_abs_diff={max_abs:.3e}  ({flag})")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name:14s} {e}")

    for fn in (test_convergence_invariants, test_active_mask_does_not_reorder_players):
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
