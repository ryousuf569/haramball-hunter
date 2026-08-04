# Structural tests for environment.reward's potential-based shaping.
# The telescoping identity is the strong one: it is exact, policy-independent,
# and a sign error, a misplaced gamma or a mishandled terminal all break it.
# Run: python tests/test_reward.py   (also works under pytest)
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace  # noqa: E402

from environment.lowblock_env import make_initial_world, make_ppcf_grid  # noqa: E402
from environment.reward import (  # noqa: E402
    FAILURE,
    SUCCESS,
    TIMEOUT,
    RewardConfig,
    build_zone_masks,
    make_pcf_state,
    phi,
    reset_potential,
    step_reward,
)
from physics.ppcf import PPCF_grid  # noqa: E402

TOL = 1e-12

CFG = RewardConfig(alpha=1.0, beta=1.0, gamma=0.99, terminal_bonus=5.0)

GRID = make_ppcf_grid()
F3_MASK, HS_MASK = build_zone_masks(GRID)


def _world(seed=11):
    players, ball, _, _, _ = make_initial_world(seed=seed)
    return players, ball


def _ppcf(players):
    return PPCF_grid(GRID, players)


def _random_rollout(players, rng, n_steps, jitter=3.0):
    # A stand-in for "several different random policies": each rng gives a
    # different sequence of states from the same s0. The identity must not care
    # which sequence it was.
    states = []
    pos = np.asarray(players["position"], dtype=float).copy()
    for _ in range(n_steps):
        pos = pos + rng.normal(0.0, jitter, size=pos.shape)
        moved = players.copy()
        moved["position"] = pos
        states.append(moved)
    return states


def _episode_return(states, cfg, outcome=SUCCESS, discount_shaping_only=True):
    # sum_t gamma^t F_t over one episode, plus the components for reporting
    players0 = states[0]
    pcf_state = make_pcf_state()
    phi_0, _ = reset_potential(players0, _ppcf(players0), F3_MASK, HS_MASK,
                               pcf_state, cfg)

    total = 0.0
    for t, players in enumerate(states[1:]):
        last = (t == len(states) - 2)
        step_outcome = outcome if last else None
        _, parts = step_reward(players, _ppcf(players), F3_MASK, HS_MASK,
                               pcf_state, cfg, outcome=step_outcome)
        shaping = parts["shaping"] if discount_shaping_only else parts["reward"]
        total += (cfg.gamma ** t) * shaping

    return total, phi_0


def test_telescoping_identity():
    # sum_t gamma^t F_t == -Phi(s0) for every policy, from the same s0
    players0, _ = _world()
    totals = []
    for seed in (0, 1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        n_steps = 3 + int(rng.integers(0, 6))  # policies differ in length too
        states = [players0] + _random_rollout(players0, rng, n_steps)
        total, phi_0 = _episode_return(states, CFG)
        totals.append(total)

        assert abs(total + phi_0) <= TOL, (
            f"seed {seed}: sum gamma^t F_t = {total:.15f} != -Phi(s0) = {-phi_0:.15f}")

    spread = max(totals) - min(totals)
    assert spread <= TOL, f"policy-dependent total (spread {spread:.3e})"
    return totals[0]


def test_telescoping_needs_terminal_zeroing():
    # the guard on the guard: without the zeroing the total is policy-dependent
    cfg = replace(CFG, zero_terminal_potential=False)
    players0, _ = _world()
    totals = []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        states = [players0] + _random_rollout(players0, rng, 5)
        total, phi_0 = _episode_return(states, cfg)
        totals.append(total)

    spread = max(totals) - min(totals)
    assert spread > 1e-6, (
        "leaving Phi(terminal) non-zero should make the total policy-dependent; "
        f"got spread {spread:.3e} -- is the flag wired up?")


def test_static_state_living_cost():
    # a state that does not change pays F = (gamma - 1) * Phi(s), i.e. negative
    players0, _ = _world()
    ppcf_result = _ppcf(players0)
    phi_s, _ = phi(players0, ppcf_result, F3_MASK, HS_MASK, CFG)

    pcf_state = make_pcf_state()
    reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)

    expected = (CFG.gamma - 1.0) * phi_s
    for _ in range(3):
        reward, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                                    pcf_state, CFG, outcome=None)
        assert abs(parts["shaping"] - expected) <= TOL, (
            f"static-state F = {parts['shaping']:.15f}, expected {expected:.15f}")
        assert reward < 0.0, "static-state reward should be a living cost"
    return phi_s, expected


def test_terminal_bonus_only_on_success():
    players0, _ = _world()
    ppcf_result = _ppcf(players0)

    got = {}
    for outcome in (SUCCESS, FAILURE, TIMEOUT):
        pcf_state = make_pcf_state()
        reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)
        _, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                               pcf_state, CFG, outcome=outcome)
        got[outcome] = parts["terminal_bonus"]
        assert parts["terminal"], f"{outcome} should be terminal"
        assert parts["phi_next"] == 0.0, f"{outcome}: terminal potential not zeroed"

    assert got[SUCCESS] == CFG.terminal_bonus, got
    assert got[FAILURE] == 0.0 and got[TIMEOUT] == 0.0, (
        f"no failure/timeout penalty by design, got {got}")


def test_components_are_reported():
    players0, _ = _world()
    ppcf_result = _ppcf(players0)
    pcf_state = make_pcf_state()
    reset_potential(players0, ppcf_result, F3_MASK, HS_MASK, pcf_state, CFG)

    reward, parts = step_reward(players0, ppcf_result, F3_MASK, HS_MASK,
                                pcf_state, CFG)
    for key in ("reward", "shaping", "terminal_bonus", "phi_prev", "phi_next",
                "pc_f3", "pc_hs", "terminal", "outcome"):
        assert key in parts, f"missing component {key}"
    assert abs(reward - (parts["shaping"] + parts["terminal_bonus"])) <= TOL


def test_phi_scale_is_order_one():
    # the calibration guard: with normalization="mean" both zone values are
    # mean pitch control, so Phi <= alpha + beta and the (gamma-1)*Phi living
    # cost stays small next to terminal_bonus.
    for seed in range(8):
        players, _ = _world(seed=seed)
        phi_s, parts = phi(players, _ppcf(players), F3_MASK, HS_MASK, CFG)
        assert 0.0 <= parts["pc_f3"] <= 1.0, parts
        assert 0.0 <= parts["pc_hs"] <= 1.0, parts
        assert phi_s <= CFG.alpha + CFG.beta + TOL, phi_s


def main():
    failures = 0
    tests = [
        test_telescoping_identity,
        test_telescoping_needs_terminal_zeroing,
        test_static_state_living_cost,
        test_terminal_bonus_only_on_success,
        test_components_are_reported,
        test_phi_scale_is_order_one,
    ]
    for fn in tests:
        try:
            out = fn()
            extra = ""
            if fn is test_telescoping_identity:
                extra = f"  sum gamma^t F_t = {out:.9f}"
            elif fn is test_static_state_living_cost:
                extra = f"  Phi = {out[0]:.6f}, (gamma-1)Phi = {out[1]:.6f}"
            print(f"  PASS  {fn.__name__}{extra}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {fn.__name__}: {e}")

    print("\n" + ("ALL PASS" if not failures else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
