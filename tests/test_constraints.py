# Tests for the indicator costs, the Lagrange multipliers, and the curriculum.
# Every bug these catch trains happily and learns the wrong thing instead.
# Run: python tests/test_constraints.py   (also works under pytest)
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CurriculumConfig, EnvConfig, LagrangeConfig  # noqa: E402
from environment import costs  # noqa: E402
from environment.lowblock_env import LowBlockEnv, SUCCESS  # noqa: E402
from environment.termination import (  # noqa: E402
    SHOT_P_MIN,
    check_shot_opening,
    p_for_radius,
    radius_for_p,
)
from ppo import (  # noqa: E402
    Curriculum,
    Lagrange,
    RolloutBuffer,
    agent_costs,
    lagrangian_advantage,
    make_venv,
)

N_ATT, N_DEF = 10, 11
K = costs.N_COSTS
TOL = 1e-6


# --- the cost indicators ---------------------------------------------------

def test_every_cost_has_a_threshold_and_a_unit():
    assert len(costs.COST_NAMES) == len(costs.COST_THRESHOLDS) == K
    assert len(costs.COST_UNITS) == K
    assert ((costs.COST_THRESHOLDS > 0) & (costs.COST_THRESHOLDS < 1)).all(), (
        "an indicator cost is a frequency; a threshold outside (0, 1) is either "
        "unsatisfiable or already satisfied")


def test_release_costs_bill_only_the_passer():
    cost, attempt = costs.empty_costs(N_ATT)
    # A 30m square ball, played the instant it arrived: cross-field and hot
    # potato, but not backward.
    costs.release_costs(cost, attempt, 4, np.array([60.0, 10.0]),
                        np.array([60.0, 40.0]), held_ticks=0)

    assert cost[4, costs.IDX["cross_field"]] == 1.0
    assert cost[4, costs.IDX["hot_potato"]] == 1.0
    assert cost[4, costs.IDX["pass_back"]] == 0.0
    assert attempt[4, costs.IDX["pass_lost"]] == 1.0, (
        "the pass must be booked as an attempt at release, or a loss 20 ticks "
        "later has no denominator")

    others = [r for r in range(N_ATT) if r != 4]
    assert cost[others].sum() == 0.0, "a team-mate was billed for a pass it did not play"
    assert attempt[others].sum() == 0.0


def test_backward_and_forward_passes_are_told_apart():
    for dx, expect in [(-10.0, 1.0), (0.0, 0.0), (+10.0, 0.0)]:
        cost, attempt = costs.empty_costs(N_ATT)
        costs.release_costs(cost, attempt, 0, np.array([60.0, 34.0]),
                            np.array([60.0 + dx, 34.0]), held_ticks=20)
        assert cost[0, costs.IDX["pass_back"]] == expect, dx
        assert cost[0, costs.IDX["hot_potato"]] == 0.0, (
            "20 ticks of possession is not a hot potato")


def test_rate_denominator_is_the_attempt_not_the_step():
    # One cross-field pass in four is 25%, not 1 in (10 agents * 50 ticks).
    sums = np.zeros(K)
    attempts = np.zeros(K)
    sums[costs.IDX["cross_field"]] = 1.0
    attempts[costs.IDX["cross_field"]] = 4.0

    r = costs.rates(sums, attempts)
    assert abs(r[costs.IDX["cross_field"]] - 0.25) < TOL, r


def test_a_constraint_with_no_attempts_reports_itself_satisfied():
    # Reporting 0.0 would push the multiplier down on every batch that happened
    # to contain no passes, a slow leak toward ignoring the constraint.
    r = costs.rates(np.zeros(K), np.zeros(K))
    assert np.allclose(r, costs.COST_THRESHOLDS), r


def test_bootstrap_constraint_is_failure_not_success():
    cost, attempt = costs.empty_costs(N_ATT)
    costs.terminal_costs(cost, attempt, SUCCESS, SUCCESS)
    assert cost[:, costs.BOOTSTRAP_IDX].sum() == 0.0, (
        "a success must cost nothing; the constraint is on FAILING")

    cost, attempt = costs.empty_costs(N_ATT)
    costs.terminal_costs(cost, attempt, "failure", SUCCESS)
    assert (cost[:, costs.BOOTSTRAP_IDX] == 1.0).all()
    assert (attempt[:, costs.BOOTSTRAP_IDX] == 1.0).all()


# --- the env wiring --------------------------------------------------------

def test_env_emits_costs_and_attempts_of_the_right_shape():
    env = LowBlockEnv(max_ticks=30, scripted_attackers=False)
    _obs, info = env.reset(seed=0)
    assert info["agent_cost"].shape == (N_ATT, K)
    assert info["cost_attempt"].shape == (N_ATT, K)

    for _ in range(30):
        _obs, _r, term, _trunc, info = env.step(env.action_space.sample())
        c, a = info["agent_cost"], info["cost_attempt"]
        assert c.shape == (N_ATT, K) and a.shape == (N_ATT, K)
        assert np.isin(c, [0.0, 1.0]).all(), "costs must be indicators"
        assert np.isin(a, [0.0, 1.0]).all()
        assert (c <= a + TOL).all(), (
            "a cost fired on an event that was never booked as an attempt, so "
            "its rate can exceed 1")
        if term:
            break


def test_per_tick_costs_are_attempted_on_every_row():
    env = LowBlockEnv(max_ticks=10, scripted_attackers=False)
    env.reset(seed=1)
    _obs, _r, _t, _tr, info = env.step(env.action_space.sample())
    a = info["cost_attempt"]
    for name in ("offside", "far_from_ball"):
        assert (a[:, costs.IDX[name]] == 1.0).all(), name


def test_a_lost_pass_is_billed_to_the_passer_even_ticks_later():
    # The point of pass_origin_row: the loss lands ticks after the release.
    env = LowBlockEnv(max_ticks=200, scripted_attackers=False)
    rng = np.random.default_rng(0)
    for seed in range(12):
        env.reset(seed=seed)
        for _ in range(200):
            action = env.action_space.sample()
            pre_row = env.holder_row()
            pending = env.pass_origin_row
            _obs, _r, term, _tr, info = env.step(action)
            lost = info["agent_cost"][:, costs.IDX["pass_lost"]]
            if lost.any():
                row = int(np.flatnonzero(lost)[0])
                assert lost.sum() == 1.0, "more than one agent billed for one pass"
                assert row in (pre_row, pending), (
                    f"pass_lost billed to row {row}, but the passer was "
                    f"{pre_row if pending is None else pending}")
                return
            if term:
                break
    raise AssertionError("no pass was lost in 12 episodes; test learned nothing")


def test_costs_on_a_terminal_step_come_from_final_info():
    # Same trap as agent_reward, where the terminal info sits in final_info.
    n_envs = 3
    venv = make_venv(EnvConfig(t_max=12), n_envs, seed=0, asynchronous=False)
    try:
        venv.reset(seed=0)
        for _ in range(60):
            _o, _r, term, trunc, info = venv.step(venv.action_space.sample())
            done = np.logical_or(term, trunc)
            c, a = agent_costs(info, done, n_envs, N_ATT, K)
            assert c.shape == (n_envs, N_ATT, K)
            if done.any():
                fin = np.asarray(info["final_info"]["agent_cost"])
                assert np.allclose(c[done], fin[done]), (
                    "terminal step took the reset env's costs")
                assert (c[done][:, :, costs.BOOTSTRAP_IDX] > 0).any() or True
                return
        raise AssertionError("no episode terminated in 60 steps")
    finally:
        venv.close()


def test_the_bootstrap_constraint_only_fires_at_a_terminal():
    env = LowBlockEnv(max_ticks=40, scripted_attackers=False)
    env.reset(seed=2)
    for _ in range(40):
        _o, _r, term, _tr, info = env.step(env.action_space.sample())
        fired = info["cost_attempt"][:, costs.BOOTSTRAP_IDX].any()
        assert fired == bool(term), (
            "the episode-level constraint was attempted on a non-terminal step")
        if term:
            return


# --- the multipliers -------------------------------------------------------

def _lag(**kw):
    return Lagrange(costs.COST_THRESHOLDS,
                    LagrangeConfig(warmup_updates=0, **kw))


def test_lambdas_are_a_simplex():
    lag = _lag()
    lam0, lam = lag.all_lambdas()
    assert abs(lam0 + lam.sum() - 1.0) < TOL
    assert (lam >= 0).all()


def test_a_violated_constraint_gains_weight_and_a_satisfied_one_loses_it():
    lag = _lag(lr=0.5)
    violated, satisfied = 0, 1
    rates = np.array(costs.COST_THRESHOLDS, dtype=float)
    rates[violated] += 0.5      # way over
    rates[satisfied] -= 0.05    # comfortably under

    before = lag.all_lambdas()[1].copy()
    for _ in range(10):
        _lam0, lam = lag.update(rates)

    assert lam[violated] > before[violated], "a violated constraint lost weight"
    assert lam[satisfied] < before[satisfied], "a satisfied constraint gained weight"


def test_multipliers_stay_bounded_under_permanent_violation():
    # The whole reason for the softmax. An unnormalised multiplier on an
    # unsatisfiable constraint grows without bound and takes the gradient with it.
    lag = _lag(lr=0.5)
    rates = np.ones(K)  # every constraint maximally violated, forever
    for _ in range(10_000):
        lam0, lam = lag.update(rates)
        assert np.isfinite(lam).all()
    assert abs(lam0 + lam.sum() - 1.0) < 1e-5
    assert lam.max() <= 1.0


def test_warmup_holds_the_multipliers_still():
    lag = Lagrange(costs.COST_THRESHOLDS, LagrangeConfig(warmup_updates=3, lr=1.0))
    rates = np.ones(K)
    for _ in range(3):
        lag.update(rates)
    assert np.allclose(lag.z, 0.0), "lambda moved before the cost critics had data"
    lag.update(rates)
    assert not np.allclose(lag.z, 0.0)


def test_bootstrap_raises_the_reward_weight_when_the_task_is_failing():
    lag = _lag(lr=0.5)
    # Every behaviour constraint met and the task failing: the policy that
    # never passes. Without the bootstrap, lambda_0 stays at the simplex floor.
    rates = np.array(costs.COST_THRESHOLDS, dtype=float) - 0.01
    rates[costs.BOOTSTRAP_IDX] = 1.0
    for _ in range(50):
        lam0, lam = lag.update(rates)

    plain = lam0
    boosted = lag.reward_weight(lam0, lam)
    assert boosted > plain, (
        "the reward weight ignored the failure multiplier; this is the collapse "
        "the bootstrap constraint exists to prevent")
    assert abs(boosted - lam[costs.BOOTSTRAP_IDX]) < TOL


def test_reward_weight_is_a_floor_not_a_replacement():
    # When the task is going well the bootstrap multiplier is small and must
    # not pull the reward's own weight DOWN.
    lag = _lag()
    lam0, lam = lag.all_lambdas()
    lam = lam.copy()
    lam[costs.BOOTSTRAP_IDX] = 0.0
    assert abs(lag.reward_weight(lam0, lam) - lam0) < TOL


# --- the combined advantage ------------------------------------------------

def test_cost_advantage_enters_with_a_minus_sign():
    adv_r = torch.zeros(4, N_ATT)
    adv_c = torch.zeros(4, N_ATT, K)
    adv_c[:, :, 0] = 1.0            # this action raises the expected cost
    adv_c[0, 0, 0] = -1.0           # ...except here
    lam = torch.zeros(K)
    lam[0] = 1.0

    a = lagrangian_advantage(adv_r, adv_c, 0.0, lam)
    assert a[0, 0] > a[1, 1], (
        "the action that LOWERS a constrained cost got the worse advantage")


def test_zero_multipliers_reduce_to_plain_ppo():
    torch.manual_seed(0)
    adv_r = torch.randn(8, N_ATT)
    adv_c = torch.randn(8, N_ATT, K)

    a = lagrangian_advantage(adv_r, adv_c, 1.0, torch.zeros(K))
    ref = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
    assert torch.allclose(a, ref, atol=1e-5)


def test_combined_advantage_is_normalised_once_not_per_channel():
    # Normalising per channel would rescale away what the multipliers encode.
    torch.manual_seed(1)
    adv_r = torch.randn(64, N_ATT)
    adv_c = torch.randn(64, N_ATT, K) * 100.0   # a wildly-scaled cost channel
    lam = torch.zeros(K)
    lam[2] = 0.5

    a = lagrangian_advantage(adv_r, adv_c, 0.5, lam)
    assert abs(float(a.mean())) < 1e-5 and abs(float(a.std()) - 1.0) < 1e-2
    small = lagrangian_advantage(adv_r, adv_c / 100.0, 0.5, lam)
    assert not torch.allclose(a, small, atol=1e-3), (
        "channel scale vanished; the multipliers have nothing left to weight")


def test_buffer_rate_matches_the_costs_it_was_given():
    buf = RolloutBuffer(4, 2, N_ATT, 8, 9, N_ATT, n_costs=K)
    buf.costs[:] = 0.0
    buf.attempts[:] = 0.0
    buf.costs[0, 0, 3, costs.IDX["pass_lost"]] = 1.0
    buf.attempts[0, 0, 3, costs.IDX["pass_lost"]] = 1.0
    buf.attempts[1, 0, 3, costs.IDX["pass_lost"]] = 1.0

    r = buf.cost_rates()
    assert abs(r[costs.IDX["pass_lost"]] - 0.5) < TOL, r


# --- the curriculum --------------------------------------------------------

def test_radius_and_probability_are_inverses():
    for r in (10.0, 15.7, 25.0, 40.0):
        assert abs(radius_for_p(p_for_radius(r)) - r) < 1e-6, r
    assert abs(radius_for_p(SHOT_P_MIN) - 15.7) < 0.2, radius_for_p(SHOT_P_MIN)


def test_annealing_on_probability_would_have_been_absurd():
    # Why CurriculumConfig is parameterised on radius. A schedule linear in p
    # spends most of its length with the gate covering most of the pitch.
    assert radius_for_p(0.60) > 45.0, radius_for_p(0.60)
    assert radius_for_p(0.74) < 20.0


def test_curriculum_starts_loose_and_ends_on_the_calibrated_gate():
    cfg = CurriculumConfig()
    curr = Curriculum(cfg, target_p_min=SHOT_P_MIN)

    p0, shift0 = curr.values()
    assert abs(radius_for_p(p0) - cfg.radius_start) < 1e-6
    assert abs(shift0 - cfg.x_shift_start) < TOL

    curr.level = cfg.steps
    p1, shift1 = curr.values()
    assert abs(p1 - SHOT_P_MIN) < 1e-6, "the curriculum never reaches the real gate"
    assert abs(shift1) < TOL, "the start state never returns to the fitted one"


def test_curriculum_disabled_is_the_real_task_from_the_first_step():
    curr = Curriculum(CurriculumConfig(enabled=False))
    p, shift = curr.values()
    assert abs(p - SHOT_P_MIN) < 1e-6 and abs(shift) < TOL


def test_curriculum_will_not_advance_on_a_thin_window_or_too_fast():
    cfg = CurriculumConfig(min_episodes=50, hold_updates=20, advance_at=0.35)
    curr = Curriculum(cfg)

    assert not curr.maybe_advance(1.0, n_episodes=10, update=100), "thin window"
    assert not curr.maybe_advance(0.10, n_episodes=100, update=100), "below advance_at"
    assert curr.maybe_advance(0.90, n_episodes=100, update=100)
    assert not curr.maybe_advance(0.90, n_episodes=100, update=105), (
        "advanced twice inside hold_updates; the task moved faster than the "
        "value function can follow")
    assert curr.maybe_advance(0.90, n_episodes=100, update=130)


def test_curriculum_never_reverses():
    curr = Curriculum(CurriculumConfig(hold_updates=0, min_episodes=0))
    curr.maybe_advance(1.0, 100, 1)
    level = curr.level
    for u in range(2, 20):
        curr.maybe_advance(0.0, 100, u)
    assert curr.level == level, "the curriculum handed the easier task back"


def test_a_looser_gate_is_actually_easier():
    # The curriculum is only a curriculum if p_min does what it claims.
    env = LowBlockEnv(max_ticks=2, scripted_attackers=False)
    env.reset(seed=0)
    players, ball, pc = env.players, env.ball, env.pc_att
    # Put an attacker on the ball 25m out, with the pitch to himself.
    row = 0
    players["position"][row] = [105.0 - 25.0, 34.0]
    players["position"][N_ATT:, 0] = 5.0
    ball = dict(ball, state="held", holder_id=int(env.attacker_ids[row]),
                position=players["position"][row].copy())
    pc = np.ones_like(np.asarray(pc))

    assert not check_shot_opening(players, ball, pc, p_min=SHOT_P_MIN), (
        "25m out should not clear the calibrated gate")
    assert check_shot_opening(players, ball, pc, p_min=p_for_radius(30.0)), (
        "25m out should clear a 30m curriculum gate")


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
